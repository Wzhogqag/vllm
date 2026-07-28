// SPDX-License-Identifier: Apache-2.0
// Indexer remote-communication RTT framework, built on DeepEP's IBGDA device
// primitives (which bypass the standard nvshmem_putmem_signal API that fails
// with cudaErrorIllegalAddress on this no-GDRCopy / data_direct=0 environment).
//
// Scheme A indexer round-trip, one indexer layer per iteration:
//   PE0 (local):  put uplink (index_q, UP_BYTES) -> PE1 ; amo-add PE1's up_flag
//                 spin until down_flag advances ; (one layer done)
//   PE1 (remote): spin until up_flag advances ; put downlink (topk, DOWN_BYTES)
//                 -> PE0 ; amo-add PE0's down_flag
//
// All RDMA targets (landing buffers + flags) live in ONE symmetric-heap
// allocation, sliced by offset -- this is the invariant IBGDA requires
// (rkey is looked up via addr - heap_base).
//
// UP/DOWN byte counts are template-free runtime params so the same framework
// tests Scheme A (8580/8192), Scheme B (17408/8192), or any future payload.
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include "deepep_ibgda_device.cuh"   // put_nbi_warp / amo_nonfetch_add / quiet
#include "deepep_utils.cuh"          // ld_acquire_sys_global

using namespace deep_ep;

// Symmetric buffer layout (single nvshmem_malloc, sliced by offset):
//   [0                 .. UP_CAP)          uplink landing buffer (PE1 receives)
//   [UP_CAP            .. UP_CAP+DOWN_CAP)  downlink landing buffer (PE0 receives)
//   [FLAG_UP_OFF]                          up_flag   (int, PE1 polls)
//   [FLAG_DOWN_OFF]                        down_flag (int, PE0 polls)
// Source data reuses the landing buffers of the *other* direction region on
// each PE (they're symmetric, same offsets exist on both PEs).
#define QP_ID 0

struct Layout {
    uint64_t heap;        // symmetric base returned by nvshmem_malloc
    uint64_t up_buf;      // offset 0
    uint64_t down_buf;    // offset UP_CAP
    int* up_flag;         // receiver(PE1) polls this
    int* down_flag;       // receiver(PE0) polls this
    uint64_t up_src;      // local source for uplink
    uint64_t down_src;    // local source for downlink
};

// Warp-collective copy of `bytes` from local src to a peer symmetric address.
// Mirrors DeepEP: if the peer is P2P-reachable (same machine, NVLink) do a
// direct store; otherwise use IBGDA RDMA. This is why DeepEP runs both intra-
// and inter-node -- forcing IBGDA when P2P is available crashes.
__device__ __forceinline__ void warp_put(uint64_t dst_sym, uint64_t src,
                                          int bytes, int dst_pe, int lane) {
    uint64_t p2p = nvshmemi_get_p2p_ptr(dst_sym, nvshmem_my_pe(), dst_pe);
    if (p2p != 0) {
        // same-machine: direct NVLink store, 16B per lane
        auto* d = reinterpret_cast<int4*>(p2p);
        auto* s = reinterpret_cast<const int4*>(src);
        int n = bytes / 16;
        for (int i = lane; i < n; i += 32) st_na_global(d + i, ld_nc_global(s + i));
    } else {
        nvshmemi_ibgda_put_nbi_warp(dst_sym, src, bytes, dst_pe, QP_ID, lane, 0);
    }
}

// Notify peer: bump a remote flag by 1. P2P -> direct store; else IBGDA amo.
__device__ __forceinline__ void notify(int* flag_sym, int val, int dst_pe) {
    uint64_t p2p = nvshmemi_get_p2p_ptr((uint64_t)flag_sym, nvshmem_my_pe(), dst_pe);
    if (p2p != 0) {
        st_na_release(reinterpret_cast<int*>(p2p), val);
    } else {
        nvshmemi_ibgda_quiet(dst_pe, QP_ID);
        nvshmemi_ibgda_amo_nonfetch_add(flag_sym, 1, dst_pe, QP_ID);
    }
}

// PE0 kernel: drive `iters` round-trips, time with clock64.
__global__ void rtt_pe0_kernel(Layout L, int up_bytes, int down_bytes,
                               int iters, int warmup, long long* out_clocks) {
    int lane = threadIdx.x & 31;
    long long t0 = 0;
    for (int i = 0; i < warmup + iters; i++) {
        if (i == warmup && lane == 0) t0 = clock64();
        warp_put(L.up_buf, L.up_src, up_bytes, /*dst_pe=*/1, lane);  // uplink
        __syncwarp();
        if (lane == 0) {
            notify(L.up_flag, i + 1, /*dst_pe=*/1);            // tell PE1 "q ready"
            while (ld_acquire_sys_global(L.down_flag) < (i + 1)) { }  // await topk
        }
        __syncwarp();
    }
    if (lane == 0) {
        long long t1 = clock64();
        *out_clocks = t1 - t0;
    }
}

// PE1 kernel: mirror — wait uplink, send downlink.
__global__ void rtt_pe1_kernel(Layout L, int up_bytes, int down_bytes,
                               int iters, int warmup) {
    int lane = threadIdx.x & 31;
    for (int i = 0; i < warmup + iters; i++) {
        if (lane == 0) {
            while (ld_acquire_sys_global(L.up_flag) < (i + 1)) { }   // await q
        }
        __syncwarp();
        warp_put(L.down_buf, L.down_src, down_bytes, /*dst_pe=*/0, lane);  // downlink
        __syncwarp();
        if (lane == 0) {
            notify(L.down_flag, i + 1, /*dst_pe=*/0);          // tell PE0 "topk ready"
        }
        __syncwarp();
    }
}

extern "C" {
// init wrappers (same as before, share one NVSHMEM context with the kernels)
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
int rix_my_pe() { return nvshmem_my_pe(); }
int rix_n_pes() { return nvshmem_n_pes(); }
void rix_finalize() { nvshmemx_hostlib_finalize(); }

// Run the RTT bench. up_bytes/down_bytes parameterize the payload (Scheme A/B).
int rix_run_rtt_bench(int up_bytes, int down_bytes, int iters, int warmup,
                      double* avg_us_out) {
    int mype = nvshmem_my_pe();
    const uint64_t UP_CAP = 65536, DOWN_CAP = 65536;  // caps >= any payload
    // one symmetric allocation, sliced
    uint64_t base = (uint64_t)nvshmem_malloc(UP_CAP + DOWN_CAP + 4096);
    uint64_t src  = (uint64_t)nvshmem_malloc(UP_CAP + DOWN_CAP);  // local sources
    Layout L;
    L.heap = base;
    L.up_buf = base;
    L.down_buf = base + UP_CAP;
    L.up_flag = (int*)(base + UP_CAP + DOWN_CAP);          // flag region
    L.down_flag = (int*)(base + UP_CAP + DOWN_CAP + 64);
    L.up_src = src;
    L.down_src = src + UP_CAP;
    cudaMemset((void*)L.up_flag, 0, 128);
    cudaMemset((void*)src, 0, UP_CAP + DOWN_CAP);

    long long* d_out;
    cudaMalloc(&d_out, sizeof(long long));
    cudaMemset(d_out, 0, sizeof(long long));

    nvshmem_barrier_all();

    if (mype == 0) {
        rtt_pe0_kernel<<<1, 32>>>(L, up_bytes, down_bytes, iters, warmup, d_out);
    } else {
        rtt_pe1_kernel<<<1, 32>>>(L, up_bytes, down_bytes, iters, warmup);
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return (int)(100 + e);

    nvshmem_barrier_all();

    if (mype == 0) {
        long long clocks = 0;
        cudaMemcpy(&clocks, d_out, sizeof(long long), cudaMemcpyDeviceToHost);
        int khz = 0;
        cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
        double total_us = (double)clocks / (khz * 1e3) * 1e6;
        *avg_us_out = total_us / iters;
    }
    nvshmem_free((void*)base);
    nvshmem_free((void*)src);
    cudaFree(d_out);
    return 0;
}
}  // extern "C"
