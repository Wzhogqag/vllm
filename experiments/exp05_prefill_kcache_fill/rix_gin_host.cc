// SPDX-License-Identifier: Apache-2.0
// GIN 跨机 indexer 单层 RTT bench —— 极简 host init。
//
// 复刻 DeepEP V2 `NCCLSymmetricMemoryContext`(csrc/kernels/backend/nccl.cu)但砍掉
// hybrid/LSA/CPU 混合内存。我们只需要:
//   1. 复用 Python 侧建好的 NCCL comm
//   2. ncclDevCommCreate 产 ncclDevComm_t(GIN device handle)
//   3. ncclMemAlloc + ncclCommWindowRegister 建对称内存
//
// 这里的 API 顺序、参数命名与 DeepEP V2 nccl.cu:70-140 一致。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <nccl_device/core.h>

#define NCCL_CHECK(x) do { \
    ncclResult_t _r = (x); \
    if (_r != ncclSuccess) { fprintf(stderr, "NCCL error %d at %s:%d\n", (int)_r, __FILE__, __LINE__); return -1; } \
} while(0)

#define CUDA_CHECK(x) do { \
    cudaError_t _r = (x); \
    if (_r != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_r), __FILE__, __LINE__); return -1; } \
} while(0)

// 单进程全局 GIN 上下文。dev_comm 用值类型(2.30.7 结构固定 size,不需 DeepEP 的可变长 wrapper)
struct RixGinCtx {
    ncclComm_t comm;
    ncclDevComm_t dev_comm;    // 值类型,直接嵌入
    void* symmetric_buffer;
    size_t symmetric_bytes;
    ncclWindow_t window;
    int rank;
    int n_ranks;
    int num_qps;
};

// 声明 kernel 入口(在 rtt_kernel.cu 里定义)
extern "C" int rix_rtt_run(RixGinCtx* ctx,
                           int up_bytes, int down_bytes,
                           int iters, int warmup,
                           double* out_avg_us,
                           double* out_p50_us, double* out_p95_us, double* out_p99_us);

extern "C" int rix_gin_init(int64_t nccl_comm_ptr, int rank, int n_ranks,
                            int64_t symmetric_bytes, int num_qps,
                            RixGinCtx** out_ctx) {
    RixGinCtx* ctx = new RixGinCtx();
    ctx->comm = reinterpret_cast<ncclComm_t>(nccl_comm_ptr);
    ctx->rank = rank;
    ctx->n_ranks = n_ranks;
    ctx->num_qps = num_qps;
    ctx->symmetric_bytes = static_cast<size_t>(symmetric_bytes);

    // 1) Query NCCL props,确认 GIN 可用(应当已在 exp03 里验证过 GDAKI)
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(ctx->comm, &props));
    if (rank == 0) {
        printf("[rix] NCCL props: ginType=%d railedGinType=%d deviceApi=%d nLsaTeams=%d\n",
               (int)props.ginType, (int)props.railedGinType, props.deviceApiSupport, props.nLsaTeams);
    }
    if (props.ginType != NCCL_GIN_TYPE_GDAKI) {
        fprintf(stderr, "[rix] ginType=%d not GDAKI, aborting\n", (int)props.ginType);
        delete ctx;
        return -2;
    }

    // 2) Create ncclDevComm(参考 nccl.cu:83-108,不用 hybrid,固定用 2.30 老 codepath)
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginContextCount    = num_qps;
    reqs.ginExclusiveContexts = 1;
    reqs.ginQueueDepth      = 1024;
    reqs.ginTrafficClass    = 3;   // RoCE v2
    reqs.ginSignalCount     = n_ranks + 4;
    reqs.ginConnectionType  = NCCL_GIN_CONNECTION_FULL;

    // NCCL 2.30 compile-time == runtime 必须一致(nccl.cu:104)
    int nccl_runtime;
    NCCL_CHECK(ncclGetVersion(&nccl_runtime));
    if (nccl_runtime != NCCL_VERSION_CODE) {
        fprintf(stderr, "[rix] NCCL compile=%d runtime=%d mismatch\n",
                NCCL_VERSION_CODE, nccl_runtime);
        delete ctx;
        return -3;
    }
    NCCL_CHECK(ncclDevCommCreate(ctx->comm, &reqs, &ctx->dev_comm));

    // 3) 对称堆:一段 GPU 显存,pin 给 NCCL Window 用
    NCCL_CHECK(ncclMemAlloc(&ctx->symmetric_buffer, ctx->symmetric_bytes));

    // 4) Register window (collective — 内部 barrier,两 rank 都必须到这一行)
    NCCL_CHECK(ncclCommWindowRegister(ctx->comm, ctx->symmetric_buffer,
                                       ctx->symmetric_bytes, &ctx->window,
                                       NCCL_WIN_STRICT_ORDERING));

    // 清零对称堆(既是 payload buffer 也是 signal 存放地)
    CUDA_CHECK(cudaMemset(ctx->symmetric_buffer, 0, ctx->symmetric_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        printf("[rix] GIN init ok: dev_comm ready, symmetric heap %zu bytes @ %p\n",
               ctx->symmetric_bytes, ctx->symmetric_buffer);
    }
    *out_ctx = ctx;
    return 0;
}

extern "C" int rix_gin_finalize(RixGinCtx* ctx) {
    if (!ctx) return 0;
    if (ctx->symmetric_buffer) {
        ncclCommWindowDeregister(ctx->comm, ctx->window);
        ncclMemFree(ctx->symmetric_buffer);
    }
    ncclDevCommDestroy(ctx->comm, &ctx->dev_comm);
    delete ctx;
    return 0;
}
