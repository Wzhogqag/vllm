// SPDX-License-Identifier: Apache-2.0
// NVSHMEM device-initiated RTT bench (Scheme A indexer round-trip).
//
// Why device-initiated: the host-put+barrier version hung -- host-driven
// comms don't fit IBGDA's device-centric model, and signal_wait_until is
// __device__-only. Here the WHOLE round-trip loop runs inside one CUDA kernel:
//
//   PE0: putmem_signal_nbi(index_q 8580B -> PE1, set sig1)      [uplink]
//        signal_wait_until(sig2 == round)                       [await topk]
//   PE1: signal_wait_until(sig1 == round)                       [await q]
//        putmem_signal_nbi(topk 8192B -> PE0, set sig2)         [downlink]
//
// No CPU in the data path, no host barrier -> no spin-hang. Timing via GPU
// %globaltimer around the loop on PE0.
//
// Init/UID bootstrap reuses the flat rix_* host wrappers (proven working).
// Kernel launched via nvshmemx_collective_launch (required for IBGDA kernels).
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>
#include "nvshmem.h"
#include "nvshmemx.h"

#define UP_BYTES   8580
#define DOWN_BYTES 8192

// --- init/UID must live in the SAME .so as the kernel so they share one
// NVSHMEM context. Mirrors the flat wrappers from rix_nvshmem.cc. ---
extern "C" {
int rix_get_uniqueid(void* out_uid_128) {
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
    int rc = nvshmemx_get_uniqueid(&uid);
    if (rc == 0) std::memcpy(out_uid_128, &uid, sizeof(uid));
    return rc;
}
int rix_init_with_uid(const void* uid_128, int myrank, int nranks) {
    nvshmemx_uniqueid_t uid;
    std::memcpy(&uid, uid_128, sizeof(uid));
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    int rc = nvshmemx_set_attr_uniqueid_args(myrank, nranks, &uid, &attr);
    if (rc != 0) return rc;
    nvshmemx_init_init_attr_ver_only(attr);
    return nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}
int  rix_my_pe() { return nvshmem_my_pe(); }
int  rix_n_pes() { return nvshmem_n_pes(); }
void rix_finalize() { nvshmemx_hostlib_finalize(); }
}

// One thread block; a single thread drives the round-trip (latency test, not
// bandwidth -- we want the per-op critical path, not many parallel ops).
__global__ void rtt_kernel(uint8_t* up_buf, uint8_t* down_buf,
                           uint8_t* up_src, uint8_t* down_src,
                           uint64_t* sig_up, uint64_t* sig_down,
                           int iters, int warmup, long long* out_ns) {
    int mype = nvshmem_my_pe();
    int peer = 1 - mype;
    long long t0 = 0;

    for (int i = 0; i < warmup + iters; i++) {
        uint64_t round = (uint64_t)(i + 1);
        if (i == warmup && mype == 0) {
            t0 = clock64();  // start timing after warmup (PE0 only)
        }
        if (mype == 0) {
            // uplink: write index_q into PE1's up_buf, set its sig_up = round
            nvshmem_putmem_signal_nbi(up_buf, up_src, UP_BYTES,
                                      sig_up, round, NVSHMEM_SIGNAL_SET, peer);
            // await downlink: PE1 will set our sig_down = round
            nvshmem_signal_wait_until(sig_down, NVSHMEM_CMP_EQ, round);
        } else {
            // await uplink from PE0
            nvshmem_signal_wait_until(sig_up, NVSHMEM_CMP_EQ, round);
            // downlink: write topk into PE0's down_buf, set its sig_down = round
            nvshmem_putmem_signal_nbi(down_buf, down_src, DOWN_BYTES,
                                      sig_down, round, NVSHMEM_SIGNAL_SET, peer);
        }
    }
    if (mype == 0) {
        long long t1 = clock64();
        *out_ns = t1 - t0;  // in GPU clocks; caller converts via clock rate
    }
}

extern "C" {

// Returns 0 on success; fills *avg_us with per-round-trip latency on PE0.
int rix_run_rtt_bench(int iters, int warmup, double* avg_us_out) {
    int mype = nvshmem_my_pe();
    // symmetric landing buffers (remote-writable) + signals
    uint8_t* up_buf   = (uint8_t*)nvshmem_malloc(UP_BYTES);
    uint8_t* down_buf = (uint8_t*)nvshmem_malloc(DOWN_BYTES);
    uint64_t* sig_up   = (uint64_t*)nvshmem_malloc(sizeof(uint64_t));
    uint64_t* sig_down = (uint64_t*)nvshmem_malloc(sizeof(uint64_t));
    // source buffers: also on the symmetric heap. With IBGDA device-initiated
    // put, the SOURCE is accessed by the NIC too; a plain cudaMalloc pointer
    // can trigger cudaErrorIllegalAddress (rc=800=100+700). Symmetric memory is
    // registered/pinned, so it's always a valid RDMA source.
    uint8_t* up_src   = (uint8_t*)nvshmem_malloc(UP_BYTES);
    uint8_t* down_src = (uint8_t*)nvshmem_malloc(DOWN_BYTES);
    long long* d_out;
    cudaMalloc(&d_out, sizeof(long long));
    cudaMemset(sig_up, 0, sizeof(uint64_t));
    cudaMemset(sig_down, 0, sizeof(uint64_t));
    cudaMemset(d_out, 0, sizeof(long long));

    nvshmem_barrier_all();  // host barrier: safe here (init phase, once)

    // Plain launch (not nvshmemx_collective_launch): collective launch uses
    // cudaLaunchCooperativeKernel which returned rc=800 (cooperative-launch
    // restriction). We don't need grid-wide sync -- one block, point-to-point
    // put/signal -- so a normal launch works with NVSHMEM device calls.
    rtt_kernel<<<1, 1>>>(up_buf, down_buf, up_src, down_src,
                         sig_up, sig_down, iters, warmup, d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return (int)(100 + e);

    nvshmem_barrier_all();

    if (mype == 0) {
        long long clocks = 0;
        cudaMemcpy(&clocks, d_out, sizeof(long long), cudaMemcpyDeviceToHost);
        // convert GPU clocks -> us via device clock rate (kHz).
        // CUDA 13 removed cudaDeviceProp::clockRate; use the attribute query.
        int clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
        double total_us = (double)clocks / (clock_khz * 1e3) * 1e6;
        *avg_us_out = total_us / iters;
    }
    nvshmem_free(up_buf); nvshmem_free(down_buf);
    nvshmem_free(sig_up); nvshmem_free(sig_down);
    nvshmem_free(up_src); nvshmem_free(down_src);
    cudaFree(d_out);
    return 0;
}

}  // extern "C"
