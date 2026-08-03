// SPDX-License-Identifier: Apache-2.0
// exp09 host — GIN symmetric context init(结构与 exp04 完全一致,extern 声明改为 exp09 4 个入口)
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

// kernel 入口(在 rix_replay_kernel.cu 里定义)
extern "C" int  rix_r0_put_up(RixGinCtx*, int);
extern "C" int  rix_r0_wait_down(RixGinCtx*, uint64_t);
extern "C" int  rix_r1_wait_up(RixGinCtx*, uint64_t);
extern "C" int  rix_r1_put_down(RixGinCtx*, int);
extern "C" void* rix_symmetric_buffer(RixGinCtx*);

extern "C" int rix_gin_init(int64_t nccl_comm_ptr, int rank, int n_ranks,
                            int64_t symmetric_bytes, int num_qps,
                            RixGinCtx** out_ctx) {
    RixGinCtx* ctx = new RixGinCtx();
    ctx->comm = reinterpret_cast<ncclComm_t>(nccl_comm_ptr);
    ctx->rank = rank;
    ctx->n_ranks = n_ranks;
    ctx->num_qps = num_qps;
    ctx->symmetric_bytes = static_cast<size_t>(symmetric_bytes);

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(ctx->comm, &props));
    if (rank == 0) {
        printf("[rix09] NCCL props: ginType=%d railedGinType=%d deviceApi=%d\n",
               (int)props.ginType, (int)props.railedGinType, props.deviceApiSupport);
    }
    if (props.ginType != NCCL_GIN_TYPE_GDAKI) {
        fprintf(stderr, "[rix09] ginType=%d not GDAKI, aborting\n", (int)props.ginType);
        delete ctx;
        return -2;
    }

    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginContextCount    = num_qps;
    reqs.ginExclusiveContexts = 1;
    reqs.ginQueueDepth      = 1024;
    reqs.ginTrafficClass    = 3;
    reqs.ginSignalCount     = n_ranks + 4;
    reqs.ginConnectionType  = NCCL_GIN_CONNECTION_FULL;

    int nccl_runtime;
    NCCL_CHECK(ncclGetVersion(&nccl_runtime));
    if (nccl_runtime != NCCL_VERSION_CODE) {
        fprintf(stderr, "[rix09] NCCL compile=%d runtime=%d mismatch\n",
                NCCL_VERSION_CODE, nccl_runtime);
        delete ctx;
        return -3;
    }
    NCCL_CHECK(ncclDevCommCreate(ctx->comm, &reqs, &ctx->dev_comm));

    NCCL_CHECK(ncclMemAlloc(&ctx->symmetric_buffer, ctx->symmetric_bytes));
    NCCL_CHECK(ncclCommWindowRegister(ctx->comm, ctx->symmetric_buffer,
                                       ctx->symmetric_bytes, &ctx->window,
                                       NCCL_WIN_STRICT_ORDERING));
    CUDA_CHECK(cudaMemset(ctx->symmetric_buffer, 0, ctx->symmetric_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        printf("[rix09] GIN init ok: symmetric heap %zu bytes @ %p\n",
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
