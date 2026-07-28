// SPDX-License-Identifier: Apache-2.0
// GIN 跨机 indexer 单层 RTT bench —— device 端。
//
// 2 rank(方案 A 上行 8580B / 下行 8192B),1 SM 1 warp,极简:
//     rank 0: gin.put(up_bytes) 附带 SignalInc(SIG_UP)  →  waitSignal(SIG_DOWN)
//     rank 1: waitSignal(SIG_UP)  →  gin.put(down_bytes) 附带 SignalInc(SIG_DOWN)
//
// clock64 每 iter 采样,host 端排序算 p50/p95/p99。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <nccl_device/core.h>
#include <nccl_device/gin.h>
#include <nccl_device/coop.h>

// host 端定义的一份完全对照
struct RixGinCtx {
    ncclComm_t comm;
    ncclDevComm_t dev_comm;
    void* symmetric_buffer;
    size_t symmetric_bytes;
    ncclWindow_t window;
    int rank;
    int n_ranks;
    int num_qps;
};

// 对称堆布局:两 rank 同 offset
//   [0                 .. UP_CAP)              上行 landing(rank1 收 rank0)
//   [UP_CAP            .. UP_CAP + DOWN_CAP)   下行 landing(rank0 收 rank1)
// 上限设成 4 MiB 每方向,足够 batch=256(方案 A 8580×256 ≈ 2.09 MiB)。
// symmetric_bytes(Python 参数)必须 >= UP_CAP + DOWN_CAP + 一些余量。
static constexpr size_t UP_CAP   = 4 * 1024 * 1024;
static constexpr size_t DOWN_CAP = 4 * 1024 * 1024;

// GIN signal ID(每 rank 有独立 signal 数组)
static constexpr ncclGinSignal_t SIG_UP   = 0;  // rank1 上等,rank0 触发(通过 put remoteAction)
static constexpr ncclGinSignal_t SIG_DOWN = 1;  // rank0 上等,rank1 触发

__launch_bounds__(32, 1)
__global__ void rtt_rank0_kernel(RixGinCtx ctx,
                                 int up_bytes,
                                 int iters, int warmup,
                                 long long* sample_clocks)
{
    if (threadIdx.x != 0) return;

    // GIN device handle(每 kernel 内构造,inline 展开无 overhead)
    ncclGin gin(ctx.dev_comm, /*contextIndex=*/0, NCCL_GIN_RESOURCE_SHARING_CTA);
    ncclTeam team = ncclTeamWorld(ctx.dev_comm);

    // 上行 buf: rank0 本地也在对称堆 offset 0(自己的 UP 区),send/recv 同一 offset 无所谓
    size_t up_src_off = 0;
    size_t up_dst_off = 0;
    int dst = 1;

    for (int it = 0; it < warmup + iters; ++it) {
        long long t0 = clock64();

        // 上行 put + 触发对端 SIG_UP 加 1(weak signal 语义即可,只要本 put 送达就 OK)
        gin.put(team, dst,
                ctx.window, up_dst_off,
                ctx.window, up_src_off,
                (size_t)up_bytes,
                ncclGin_WeakSignalInc{SIG_UP});

        // 等本地 SIG_DOWN >= it+1
        gin.waitSignal(ncclCoopThread{}, SIG_DOWN, (uint64_t)(it + 1));

        long long t1 = clock64();
        if (it >= warmup) sample_clocks[it - warmup] = t1 - t0;
    }
}

__launch_bounds__(32, 1)
__global__ void rtt_rank1_kernel(RixGinCtx ctx,
                                 int down_bytes,
                                 int iters, int warmup)
{
    if (threadIdx.x != 0) return;

    ncclGin gin(ctx.dev_comm, /*contextIndex=*/0, NCCL_GIN_RESOURCE_SHARING_CTA);
    ncclTeam team = ncclTeamWorld(ctx.dev_comm);

    size_t down_src_off = UP_CAP;
    size_t down_dst_off = UP_CAP;
    int dst = 0;

    for (int it = 0; it < warmup + iters; ++it) {
        // 等 rank0 上行到:本地 SIG_UP >= it+1
        gin.waitSignal(ncclCoopThread{}, SIG_UP, (uint64_t)(it + 1));

        // 下行 put + 触发 rank0 的 SIG_DOWN
        gin.put(team, dst,
                ctx.window, down_dst_off,
                ctx.window, down_src_off,
                (size_t)down_bytes,
                ncclGin_WeakSignalInc{SIG_DOWN});
    }
}

static int cmp_ll(const void* a, const void* b) {
    long long da = *(const long long*)a, db = *(const long long*)b;
    return (da > db) - (da < db);
}

extern "C" int rix_rtt_run(RixGinCtx* ctx,
                           int up_bytes, int down_bytes,
                           int iters, int warmup,
                           double* out_avg_us,
                           double* out_p50_us, double* out_p95_us, double* out_p99_us)
{
    long long* d_samples = nullptr;
    size_t sample_bytes = sizeof(long long) * iters;
    if (cudaMalloc((void**)&d_samples, sample_bytes) != cudaSuccess) return -100;
    cudaMemset(d_samples, 0, sample_bytes);

    if (ctx->rank == 0) {
        rtt_rank0_kernel<<<1, 32>>>(*ctx, up_bytes, iters, warmup, d_samples);
    } else {
        rtt_rank1_kernel<<<1, 32>>>(*ctx, down_bytes, iters, warmup);
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "[rix] kernel error: %s\n", cudaGetErrorString(e));
        cudaFree(d_samples);
        return -101;
    }

    if (ctx->rank != 0) {
        cudaFree(d_samples);
        *out_avg_us = *out_p50_us = *out_p95_us = *out_p99_us = 0;
        return 0;
    }

    long long* h = (long long*)malloc(sample_bytes);
    cudaMemcpy(h, d_samples, sample_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_samples);

    int khz = 0;
    cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
    double us_per_clock = 1.0 / ((double)khz * 1e3) * 1e6;

    qsort(h, iters, sizeof(long long), cmp_ll);
    double sum = 0;
    int stat_n = iters * 99 / 100;
    if (stat_n < 1) stat_n = iters;
    for (int i = 0; i < stat_n; ++i) sum += (double)h[i];
    *out_avg_us = sum / stat_n * us_per_clock;
    *out_p50_us = (double)h[iters * 50 / 100] * us_per_clock;
    *out_p95_us = (double)h[iters * 95 / 100] * us_per_clock;
    *out_p99_us = (double)h[iters * 99 / 100] * us_per_clock;

    free(h);
    return 0;
}
