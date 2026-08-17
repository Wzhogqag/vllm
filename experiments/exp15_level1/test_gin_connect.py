"""exp15 地基验证:用 StatelessProcessGroup(不靠 torchrun)建 2-rank GIN comm。

这是 Level 1 最不确定的一环 —— 真 vLLM worker 要在自己 8-rank world 之外另起一个到远端
独立进程的 GIN comm。这里先脱离 vLLM 单独验证这套建连原语 + rix_gin_init 能通。

复用 vLLM in-tree 原语(weight_transfer/nccl_common.py:33 同款):
    StatelessProcessGroup.create(host, port, rank, world_size) → PyNcclCommunicator(pg, device)
    → pynccl_comm.comm (ncclComm_t = c_void_p) → 喂给 rix_gin_init

用法(两进程,可同机 loopback 也可跨机):
    rank0:  python test_gin_connect.py 0 <host_ip> <port>
    rank1:  python test_gin_connect.py 1 <host_ip> <port>
rank0 put 一个已知 pattern 到对称堆上行区并 signal,rank1 wait+读+校验+回写,rank0 收回校验。
recall 不涉及,只验:建连成功 + ginType + 一次 put/wait 往返数据无损。
"""

from __future__ import annotations

import ctypes
import os
import sys

import torch

from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "exp09_replay_demo", "librix_replay.so")
UP_CAP = 4 * 1024 * 1024
TEST_BYTES = 4096


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(os.path.abspath(LIB), mode=ctypes.RTLD_GLOBAL)
    lib.rix_gin_init.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int64, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_void_p)]
    lib.rix_gin_init.restype = ctypes.c_int
    lib.rix_gin_finalize.argtypes = [ctypes.c_void_p]
    for fn, extra in [("rix_r0_put_up", ctypes.c_int),
                      ("rix_r0_wait_down", ctypes.c_uint64),
                      ("rix_r1_wait_up", ctypes.c_uint64),
                      ("rix_r1_put_down", ctypes.c_int)]:
        f = getattr(lib, fn)
        f.argtypes = [ctypes.c_void_p, extra]
        f.restype = ctypes.c_int
    lib.rix_symmetric_buffer.argtypes = [ctypes.c_void_p]
    lib.rix_symmetric_buffer.restype = ctypes.c_void_p
    return lib


def _comm_as_int64(pynccl_comm) -> int:
    c = pynccl_comm.comm
    return int(c.value) if hasattr(c, "value") else int(c)


def main():
    rank = int(sys.argv[1])
    host = sys.argv[2]
    port = int(sys.argv[3])

    # 单卡:两进程各占一张卡(同机用不同卡,跨机各自 0)
    local_dev = int(os.environ.get("GIN_TEST_DEV", "0"))
    torch.cuda.set_device(local_dev)
    device = torch.device(f"cuda:{local_dev}")

    print(f"[rank{rank}] StatelessProcessGroup.create host={host}:{port} ...", flush=True)
    pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=2)
    pynccl = PyNcclCommunicator(pg, device=device)
    comm_int = _comm_as_int64(pynccl)
    print(f"[rank{rank}] pynccl comm ptr = 0x{comm_int:x}", flush=True)

    lib = _load_lib()
    ctx = ctypes.c_void_p(0)
    rc = lib.rix_gin_init(comm_int, rank, 2, 16 << 20, 1, ctypes.byref(ctx))
    assert rc == 0, f"[rank{rank}] rix_gin_init FAILED rc={rc}"
    sym = int(lib.rix_symmetric_buffer(ctx))
    print(f"[rank{rank}] GIN init ok, sym @ 0x{sym:x}", flush=True)

    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]

    if rank == 0:
        # 写已知 pattern 到上行区 → put → 等 rank1 回写下行区 → 校验往返
        pattern = torch.arange(TEST_BYTES, dtype=torch.uint8, device=device)
        cudart.cudaMemcpy(sym, pattern.data_ptr(), TEST_BYTES, 3)
        assert lib.rix_r0_put_up(ctx, TEST_BYTES) == 0
        assert lib.rix_r0_wait_down(ctx, 1) == 0
        got = torch.empty(TEST_BYTES, dtype=torch.uint8, device=device)
        cudart.cudaMemcpy(got.data_ptr(), sym + UP_CAP, TEST_BYTES, 3)
        # rank1 回写的应是 pattern+1
        expect = (pattern + 1).to(torch.uint8)
        ok = torch.equal(got, expect)
        print(f"[rank0] round-trip data intact = {ok}", flush=True)
        print("PASS ✓ GIN connect via StatelessProcessGroup" if ok else "FAIL ✗",
              flush=True)
    else:
        assert lib.rix_r1_wait_up(ctx, 1) == 0
        buf = torch.empty(TEST_BYTES, dtype=torch.uint8, device=device)
        cudart.cudaMemcpy(buf.data_ptr(), sym, TEST_BYTES, 3)
        out = (buf + 1).to(torch.uint8)     # 简单变换,验数据真的过来了
        cudart.cudaMemcpy(sym + UP_CAP, out.data_ptr(), TEST_BYTES, 3)
        assert lib.rix_r1_put_down(ctx, TEST_BYTES) == 0
        print("[rank1] served one round-trip", flush=True)

    lib.rix_gin_finalize(ctx)


if __name__ == "__main__":
    main()
