# SPDX-License-Identifier: Apache-2.0
"""GIN 后端探测 launcher。

torch.distributed 起 NCCL,把 NCCL communicator 的原生指针交给 rix_probe_gin,
让它调 ncclCommQueryProperties 打印 ginType/railedGinType。

用法(两机各起 1 rank,--nnodes=2 --nproc_per_node=1,见 run_probe.sh):
    torch.distributed.run 会设 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT,
    我们只从 env 读、不硬编。
"""
import ctypes
import os
import sys

import torch
import torch.distributed as dist


def get_native_nccl_comm(group: dist.ProcessGroup) -> int:
    """从 torch ProcessGroup 里挖出底层 ncclComm_t 原生指针,cast 到 int。

    torch 2.10+ 用 ProcessGroupNCCL._comm_ptr(属性,直接返回原生 handle)。
    再往老版本可能是方法或字段,遍历几种可能。
    """
    backend = group._get_backend(torch.device("cuda"))
    for attr in ("_comm_ptr", "_get_comm_handle", "_ncclComm", "get_nccl_comm"):
        if hasattr(backend, attr):
            v = getattr(backend, attr)
            handle = v() if callable(v) else v
            return int(handle)
    raise RuntimeError(
        f"can't find NCCL comm handle on backend {type(backend).__name__}; "
        f"attrs: {[a for a in dir(backend) if 'omm' in a.lower() or 'ccl' in a.lower()]}"
    )


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 探测程序只需要 1 张卡;上层脚本已经通过 CUDA_VISIBLE_DEVICES 指定
    torch.cuda.set_device(local_rank)

    # NCCL backend,让 torch 帮我们把 NCCL communicator 建起来
    dist.init_process_group(backend="nccl")

    # 触发一次通信,确保 NCCL comm 已 lazy-init
    x = torch.zeros(1, device="cuda")
    dist.all_reduce(x)
    torch.cuda.synchronize()

    if rank == 0:
        print(f"NCCL comm initialized. world_size={world_size}", flush=True)

    comm_ptr = get_native_nccl_comm(dist.group.WORLD)

    if rank == 0:
        print(f"native NCCL comm ptr: 0x{comm_ptr:x}", flush=True)

    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "librix_gin_probe.so")
    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    lib.rix_probe_gin.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int]
    lib.rix_probe_gin.restype = ctypes.c_int

    # rank 0 先打印,其他 rank 后打(避免行乱)
    if rank == 0:
        rc = lib.rix_probe_gin(comm_ptr, rank, world_size)
    dist.barrier()
    if rank != 0:
        rc = lib.rix_probe_gin(comm_ptr, rank, world_size)
    dist.barrier()

    if rc != 0:
        print(f"[rank {rank}] rix_probe_gin returned rc={rc}", file=sys.stderr, flush=True)
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
