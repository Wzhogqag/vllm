// SPDX-License-Identifier: Apache-2.0
// GIN 后端探测器 -- 判决 exp03/后续 V2 移植路线是否可行的最小 C 程序。
//
// 由 Python launcher 通过 dlopen 调用,函数 rix_probe_gin() 参数:
//   ncclComm      : Python 层传入的已建好的 NCCL communicator (int64,cast 到 ncclComm_t)
//   my_rank       : rank id (由 launcher 传入,只影响打印)
//   n_ranks       : world size (用于打印)
//
// 打印 NCCL 版本 + props.ginType + props.railedGinType + 其他关键属性,判决:
//   ginType == GDAKI (3)   -> V2 路线绿灯,GPU-initiated RDMA 可用
//   ginType == PROXY (2)   -> CPU 代理,几十 us 起,不满足 us 级需求
//   ginType == NONE  (0)   -> NIC/驱动不支持 GIN,V2 路线死路
//
// 探测本身 <1ms,不做任何数据传输。

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <nccl.h>
#include <nccl_device/core.h>

static const char* gin_name(ncclGinType_t t) {
    switch (t) {
        case NCCL_GIN_TYPE_NONE:  return "NONE  (0) - GIN not available";
        case NCCL_GIN_TYPE_PROXY: return "PROXY (2) - CPU-proxy, ~tens of us";
        case NCCL_GIN_TYPE_GDAKI: return "GDAKI (3) - GPU-initiated RDMA (target)";
        case NCCL_GIN_TYPE_GPI:   return "GPI   (4) - alt GPU-initiated path";
        default:                  return "UNKNOWN";
    }
}

extern "C" int rix_probe_gin(int64_t comm_ptr, int my_rank, int n_ranks) {
    ncclComm_t comm = reinterpret_cast<ncclComm_t>(comm_ptr);

    // NCCL runtime version (from the .so we actually loaded)
    int v = 0;
    ncclResult_t rc = ncclGetVersion(&v);
    if (rc != ncclSuccess) {
        fprintf(stderr, "[rank %d] ncclGetVersion failed rc=%d\n", my_rank, (int)rc);
        return 100 + (int)rc;
    }

    if (my_rank == 0) {
        printf("\n========== NCCL GIN probe ==========\n");
        printf("NCCL runtime version:      %d.%d.%d  (raw=%d)\n",
               v / 10000, (v / 100) % 100, v % 100, v);
        printf("NCCL compile-time header:  %d.%d.%d  (NCCL_VERSION_CODE=%d)\n",
               NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, NCCL_VERSION_CODE);
        printf("world size:                %d\n", n_ranks);
    }

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    rc = ncclCommQueryProperties(comm, &props);
    if (rc != ncclSuccess) {
        fprintf(stderr, "[rank %d] ncclCommQueryProperties failed rc=%d\n", my_rank, (int)rc);
        return 200 + (int)rc;
    }

    // Print from all ranks so we can spot per-node asymmetry (e.g. NIC diff).
    printf("[rank %d] rank=%d/%d cudaDev=%d nvmlDev=%d\n",
           my_rank, props.rank, props.nRanks, props.cudaDev, props.nvmlDev);
    printf("[rank %d]   deviceApiSupport = %s\n",   my_rank, props.deviceApiSupport ? "true" : "false");
    printf("[rank %d]   multimemSupport  = %s\n",   my_rank, props.multimemSupport ? "true" : "false");
    printf("[rank %d]   hostRmaSupport   = %s\n",   my_rank, props.hostRmaSupport ? "true" : "false");
    printf("[rank %d]   nLsaTeams        = %d\n",   my_rank, props.nLsaTeams);
    printf("[rank %d]   ginType          = %s\n",   my_rank, gin_name(props.ginType));
    printf("[rank %d]   railedGinType    = %s\n",   my_rank, gin_name(props.railedGinType));

    if (my_rank == 0) {
        printf("\n--- verdict ---\n");
        if (props.ginType == NCCL_GIN_TYPE_GDAKI) {
            printf("  GDAKI available -- V2 route GREEN. Start porting.\n");
        } else if (props.ginType == NCCL_GIN_TYPE_PROXY) {
            printf("  Only PROXY -- CPU-mediated, ~tens of us. NOT us-level.\n");
        } else if (props.ginType == NCCL_GIN_TYPE_NONE) {
            printf("  NONE -- GIN unavailable on this NIC/driver. V2 route DEAD.\n");
        } else {
            printf("  Unknown ginType=%d\n", (int)props.ginType);
        }
        printf("=====================================\n\n");
        fflush(stdout);
    }

    return 0;
}
