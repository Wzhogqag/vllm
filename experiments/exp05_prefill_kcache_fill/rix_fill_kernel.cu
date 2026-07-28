// SPDX-License-Identifier: Apache-2.0
// exp05: prefill K cache 灌充 —— 单向传输,不做 RTT。
//
// 三种 mode:
//   0 = BULK        1 个 put,payload = seq_len * per_token
//   1 = STREAMING   seq_len 个 put,每个 payload = per_token
//   2 = CHUNKED     seq_len/chunk 个 put,每个 payload = chunk * per_token
//
// 一次 kernel 完成一次"灌充",clock64 记总时间,host 端换算 μs。
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

static constexpr ncclGinSignal_t SIG_DONE = 0;

// mode = 0/1/2 (BULK / STREAMING / CHUNKED)
__launch_bounds__(32, 1)
__global__ void fill_sender_kernel(RixGinCtx ctx,
                                   int seq_len, int per_token_bytes,
                                   int chunk_tokens, int mode,
                                   long long* out_clocks)
{
    if (threadIdx.x != 0) return;
    ncclGin gin(ctx.dev_comm, /*qp=*/0, NCCL_GIN_RESOURCE_SHARING_CTA);
    ncclTeam team = ncclTeamWorld(ctx.dev_comm);
    int dst = 1;

    // 本地 src 用对称堆前半段(避免边界);dst 用对称堆同 offset,rank1 上收
    size_t src_off = 0;
    size_t dst_off = 0;

    long long t0 = clock64();

    if (mode == 0) {
        // BULK: 一次 put 全部 seq_len 个 token 的 K
        size_t total_bytes = (size_t)seq_len * per_token_bytes;
        gin.put(team, dst,
                ctx.window, dst_off, ctx.window, src_off,
                total_bytes,
                ncclGin_WeakSignalInc{SIG_DONE});
    } else if (mode == 1) {
        // STREAMING: 每 token 一个 put
        for (int i = 0; i < seq_len; ++i) {
            size_t off = (size_t)i * per_token_bytes;
            gin.put(team, dst,
                    ctx.window, dst_off + off, ctx.window, src_off + off,
                    (size_t)per_token_bytes,
                    ncclGin_None{});  // 除最后一个,不打 signal
        }
        // 最后一次单独打 signal,通知 rank1 全部灌充完
        gin.signal(team, dst, ncclGin_WeakSignalInc{SIG_DONE});
    } else {
        // CHUNKED: 每 chunk_tokens 个 token 一个 put
        int n_chunks = (seq_len + chunk_tokens - 1) / chunk_tokens;
        for (int c = 0; c < n_chunks; ++c) {
            int tokens_this = min(chunk_tokens, seq_len - c * chunk_tokens);
            size_t chunk_bytes = (size_t)tokens_this * per_token_bytes;
            size_t off = (size_t)c * chunk_tokens * per_token_bytes;
            gin.put(team, dst,
                    ctx.window, dst_off + off, ctx.window, src_off + off,
                    chunk_bytes,
                    ncclGin_None{});
        }
        gin.signal(team, dst, ncclGin_WeakSignalInc{SIG_DONE});
    }

    // flush 确保所有 put 完成 —— 这才是"灌充完成"的时刻
    gin.flush(ncclCoopThread{});

    long long t1 = clock64();
    *out_clocks = t1 - t0;
}

__launch_bounds__(32, 1)
__global__ void fill_receiver_kernel(RixGinCtx ctx, uint64_t target_signal_val)
{
    if (threadIdx.x != 0) return;
    ncclGin gin(ctx.dev_comm, /*qp=*/0, NCCL_GIN_RESOURCE_SHARING_CTA);
    gin.waitSignal(ncclCoopThread{}, SIG_DONE, target_signal_val);
}

extern "C" int rix_fill_run(RixGinCtx* ctx,
                            int seq_len, int per_token_bytes,
                            int chunk_tokens, int mode,
                            uint64_t signal_target,   // ← host 侧传入
                            double* out_us)
{
    long long* d_out;
    if (cudaMalloc((void**)&d_out, sizeof(long long)) != cudaSuccess) return -100;
    cudaMemset(d_out, 0, sizeof(long long));

    if (ctx->rank == 0) {
        fill_sender_kernel<<<1, 32>>>(*ctx, seq_len, per_token_bytes,
                                       chunk_tokens, mode, d_out);
    } else {
        fill_receiver_kernel<<<1, 32>>>(*ctx, signal_target);
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "[rix] fill kernel error: %s\n", cudaGetErrorString(e));
        cudaFree(d_out);
        return -101;
    }

    if (ctx->rank == 0) {
        long long clocks = 0;
        cudaMemcpy(&clocks, d_out, sizeof(long long), cudaMemcpyDeviceToHost);
        int khz = 0;
        cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
        double us_per_clock = 1.0 / ((double)khz * 1e3) * 1e6;
        *out_us = (double)clocks * us_per_clock;
    } else {
        *out_us = 0;
    }
    cudaFree(d_out);
    return 0;
}
