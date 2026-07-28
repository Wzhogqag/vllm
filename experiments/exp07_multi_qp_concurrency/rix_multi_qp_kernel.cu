// SPDX-License-Identifier: Apache-2.0
// exp07: 多 QP 并发扫描 —— 模拟多主实例同时向一个 indexer 发送。
//
// N 个 CTA(block)并发发送方案 A payload(8580B up),每个 CTA 用自己的 QP id。
// 每 CTA 独立循环 iters 次,统计单次 put+wait RTT。
//
// 关键设计:
//   sender: N 个 block × 1 warp。block i 用 qp=i, signal=2*i(up), 期待 signal=2*i+1(down)
//   receiver: N 个 block × 1 warp。block i 用 qp=i, 等 signal=2*i, 回 signal=2*i+1
//
// 这样每个 CTA 有自己独立的通道,不互相依赖对方 signal;可以清楚看到"多 QP 并发"的
// 效果 —— 若 QP 足够,N 个 CTA 应该几乎同时完成一次 RTT。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <nccl_device/core.h>
#include <nccl_device/gin.h>
#include <nccl_device/coop.h>

struct RixGinCtx {
    ncclComm_t comm;
    ncclDevComm_t dev_comm;
    void* symmetric_buffer;
    size_t symmetric_bytes;
    ncclWindow_t window;
    int rank, n_ranks, num_qps;
};

// 每 block 独占的 landing buffer size。
// N=8 blocks × 2 方向,payload B=256 时单向 2.2 MB → 每 block 至少 2.5 MB。
// 保守设 4 MiB,和 exp05 一致。
static constexpr size_t PER_BLOCK_CAP = 4 * 1024 * 1024;

__launch_bounds__(32, 1)
__global__ void multi_qp_sender_kernel(RixGinCtx ctx,
                                       int up_bytes, int down_bytes,
                                       int iters, int warmup, int n_blocks,
                                       long long* per_block_clocks,
                                       int* per_block_timeouts)
{
    if (threadIdx.x != 0) return;
    int bid = blockIdx.x;
    int qp_idx = bid % ctx.num_qps;

    ncclGin gin(ctx.dev_comm, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
    ncclTeam team = ncclTeamWorld(ctx.dev_comm);
    int dst = 1;

    size_t off = (size_t)bid * PER_BLOCK_CAP;
    ncclGinSignal_t sig_up   = (ncclGinSignal_t)(2 * bid);
    ncclGinSignal_t sig_down = (ncclGinSignal_t)(2 * bid + 1);

    // 单次 RTT 超时:10 ms(足以覆盖极端 QP 争抢);超时后放弃这次 iter 计数
    // clock64 的单位是 SM cycle,取 GPU 1.5 GHz 保守估:10ms = 15M cycles;取 20M 安全
    const long long TIMEOUT_CYCLES = 20000000LL;

    long long total_clocks = 0;
    int timeout_count = 0;
    for (int it = 0; it < warmup + iters; ++it) {
        long long t0 = clock64();
        gin.put(team, dst,
                ctx.window, off, ctx.window, off,
                (size_t)up_bytes,
                ncclGin_WeakSignalInc{sig_up});

        // 手写超时 waitSignal:reader shadow ptr 不易得,直接用 gin.readSignal 轮询
        uint64_t target = (uint64_t)(it + 1);
        bool timed_out = false;
        while (true) {
            uint64_t v = gin.readSignal(sig_down, 64, cuda::memory_order_acquire);
            if (v >= target) break;
            if (clock64() - t0 > TIMEOUT_CYCLES) { timed_out = true; break; }
        }
        long long t1 = clock64();
        if (it >= warmup) {
            if (timed_out) timeout_count++;
            else total_clocks += (t1 - t0);
        }
    }
    per_block_clocks[bid] = total_clocks;
    per_block_timeouts[bid] = timeout_count;
}

__launch_bounds__(32, 1)
__global__ void multi_qp_receiver_kernel(RixGinCtx ctx,
                                         int up_bytes, int down_bytes,
                                         int iters, int warmup, int n_blocks)
{
    if (threadIdx.x != 0) return;
    int bid = blockIdx.x;
    int qp_idx = bid % ctx.num_qps;

    ncclGin gin(ctx.dev_comm, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
    ncclTeam team = ncclTeamWorld(ctx.dev_comm);
    int dst = 0;

    size_t off = (size_t)bid * PER_BLOCK_CAP;
    ncclGinSignal_t sig_up   = (ncclGinSignal_t)(2 * bid);
    ncclGinSignal_t sig_down = (ncclGinSignal_t)(2 * bid + 1);

    // 同样加超时,免得 sender 那边死了 receiver 永远等
    const long long TIMEOUT_CYCLES = 20000000LL;

    for (int it = 0; it < warmup + iters; ++it) {
        long long t0 = clock64();
        uint64_t target = (uint64_t)(it + 1);
        bool timed_out = false;
        while (true) {
            uint64_t v = gin.readSignal(sig_up, 64, cuda::memory_order_acquire);
            if (v >= target) break;
            if (clock64() - t0 > TIMEOUT_CYCLES) { timed_out = true; break; }
        }
        if (timed_out) {
            // 超时也回一个 signal,不然 sender 侧计数错乱
            gin.signal(team, dst, ncclGin_WeakSignalInc{sig_down});
            continue;
        }
        gin.put(team, dst,
                ctx.window, off, ctx.window, off,
                (size_t)down_bytes,
                ncclGin_WeakSignalInc{sig_down});
    }
}

extern "C" int rix_multi_qp_run(RixGinCtx* ctx,
                                int up_bytes, int down_bytes,
                                int iters, int warmup, int n_blocks,
                                double* out_avg_us,
                                double* out_max_us,
                                double* out_min_us,
                                int* out_total_timeouts)
{
    long long* d_clocks;
    int* d_timeouts;
    size_t clock_bytes = sizeof(long long) * n_blocks;
    size_t to_bytes = sizeof(int) * n_blocks;
    if (cudaMalloc((void**)&d_clocks, clock_bytes) != cudaSuccess) return -100;
    if (cudaMalloc((void**)&d_timeouts, to_bytes) != cudaSuccess) return -100;
    cudaMemset(d_clocks, 0, clock_bytes);
    cudaMemset(d_timeouts, 0, to_bytes);

    if (ctx->rank == 0) {
        multi_qp_sender_kernel<<<n_blocks, 32>>>(*ctx, up_bytes, down_bytes,
                                                  iters, warmup, n_blocks,
                                                  d_clocks, d_timeouts);
    } else {
        multi_qp_receiver_kernel<<<n_blocks, 32>>>(*ctx, up_bytes, down_bytes,
                                                    iters, warmup, n_blocks);
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "[rix] multi_qp kernel error: %s\n", cudaGetErrorString(e));
        cudaFree(d_clocks); cudaFree(d_timeouts);
        return -101;
    }

    if (ctx->rank == 0) {
        long long* h_clocks = (long long*)malloc(clock_bytes);
        int* h_to = (int*)malloc(to_bytes);
        cudaMemcpy(h_clocks, d_clocks, clock_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_to, d_timeouts, to_bytes, cudaMemcpyDeviceToHost);
        int khz = 0;
        cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
        double us_per_clock = 1.0 / ((double)khz * 1e3) * 1e6;

        double sum_avg = 0, max_avg = 0, min_avg = 1e18;
        int total_to = 0;
        for (int i = 0; i < n_blocks; ++i) {
            int good = iters - h_to[i];
            if (good <= 0) { total_to += h_to[i]; continue; }
            double avg_us = (double)h_clocks[i] / good * us_per_clock;
            sum_avg += avg_us;
            if (avg_us > max_avg) max_avg = avg_us;
            if (avg_us < min_avg) min_avg = avg_us;
            total_to += h_to[i];
        }
        *out_avg_us = sum_avg / n_blocks;
        *out_max_us = max_avg;
        *out_min_us = min_avg;
        *out_total_timeouts = total_to;
        free(h_clocks); free(h_to);
    } else {
        *out_avg_us = *out_max_us = *out_min_us = 0;
        *out_total_timeouts = 0;
    }
    cudaFree(d_clocks); cudaFree(d_timeouts);
    return 0;
}
