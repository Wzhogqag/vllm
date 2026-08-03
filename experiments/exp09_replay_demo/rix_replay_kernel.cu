// SPDX-License-Identifier: Apache-2.0
// exp09 kernel-split replay — 每层一次 host 触发,rank0/rank1 对称。
//
// 与 exp04 差别:exp04 是 rank0 一个大 kernel 内 loop 迭代次,payload 是随机字节。
// exp09 是 host 侧 for-layer 循环 61 次,每层各 launch 一个小 kernel,
// 中间在 rank1 host 上做 torch mqa 打分 + topk。
//
// 单层每 rank 的 kernel 结构:
//   rank0.launch_up:   gin.put(up_bytes, SignalInc SIG_UP)         // 送 payload
//   rank0.launch_wait: gin.waitSignal(SIG_DOWN, expected)          // 等 topk 回
//   rank1.launch_wait: gin.waitSignal(SIG_UP, expected)            // 等 payload 到
//   [rank1 host: torch bmm + weighted sum + topk → 写下行 offset]
//   rank1.launch_put:  gin.put(down_bytes, SignalInc SIG_DOWN)     // 送回 topk
//
// signal expected 计数在 host 侧持有(rank 局部单调递增),不需要跨 rank 同步 iter 编号。
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
    int rank;
    int n_ranks;
    int num_qps;
};

static constexpr size_t UP_CAP   = 4 * 1024 * 1024;

static constexpr ncclGinSignal_t SIG_UP   = 0;  // rank1 上等,rank0 触发
static constexpr ncclGinSignal_t SIG_DOWN = 1;  // rank0 上等,rank1 触发

// ============================================================
// rank0:两个小 kernel(put_up / wait_down)
// ============================================================
__launch_bounds__(32, 1)
__global__ void rank0_put_up_kernel(RixGinCtx ctx, int up_bytes)
{
    if (threadIdx.x != 0) return;
    ncclGin gin(ctx.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    ncclTeam team = ncclTeamWorld(ctx.dev_comm);
    size_t off = 0;
    int dst = 1;
    gin.put(team, dst,
            ctx.window, off,
            ctx.window, off,
            (size_t)up_bytes,
            ncclGin_WeakSignalInc{SIG_UP});
}

__launch_bounds__(32, 1)
__global__ void rank0_wait_down_kernel(RixGinCtx ctx, uint64_t expected)
{
    if (threadIdx.x != 0) return;
    ncclGin gin(ctx.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    gin.waitSignal(ncclCoopThread{}, SIG_DOWN, expected);
}

// ============================================================
// rank1:两个小 kernel(wait_up / put_down)
// ============================================================
__launch_bounds__(32, 1)
__global__ void rank1_wait_up_kernel(RixGinCtx ctx, uint64_t expected)
{
    if (threadIdx.x != 0) return;
    ncclGin gin(ctx.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    gin.waitSignal(ncclCoopThread{}, SIG_UP, expected);
}

__launch_bounds__(32, 1)
__global__ void rank1_put_down_kernel(RixGinCtx ctx, int down_bytes)
{
    if (threadIdx.x != 0) return;
    ncclGin gin(ctx.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    ncclTeam team = ncclTeamWorld(ctx.dev_comm);
    size_t off = UP_CAP;
    int dst = 0;
    gin.put(team, dst,
            ctx.window, off,
            ctx.window, off,
            (size_t)down_bytes,
            ncclGin_WeakSignalInc{SIG_DOWN});
}

// ============================================================
// Host 入口 —— 每次调用 = 一层的一个动作
// 4 个函数都异步 launch 并 sync,方便 Python 侧插入 torch 计算或 timing。
// ============================================================
extern "C" int rix_r0_put_up(RixGinCtx* ctx, int up_bytes) {
    rank0_put_up_kernel<<<1, 32>>>(*ctx, up_bytes);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[rix09] r0_put_up: %s\n", cudaGetErrorString(e)); return -101; }
    return 0;
}
extern "C" int rix_r0_wait_down(RixGinCtx* ctx, uint64_t expected) {
    rank0_wait_down_kernel<<<1, 32>>>(*ctx, expected);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[rix09] r0_wait_down: %s\n", cudaGetErrorString(e)); return -102; }
    return 0;
}
extern "C" int rix_r1_wait_up(RixGinCtx* ctx, uint64_t expected) {
    rank1_wait_up_kernel<<<1, 32>>>(*ctx, expected);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[rix09] r1_wait_up: %s\n", cudaGetErrorString(e)); return -103; }
    return 0;
}
extern "C" int rix_r1_put_down(RixGinCtx* ctx, int down_bytes) {
    rank1_put_down_kernel<<<1, 32>>>(*ctx, down_bytes);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[rix09] r1_put_down: %s\n", cudaGetErrorString(e)); return -104; }
    return 0;
}

// 便利:返回对称堆基址(host 端要 view 上面的 payload)
extern "C" void* rix_symmetric_buffer(RixGinCtx* ctx) {
    return ctx ? ctx->symmetric_buffer : nullptr;
}
