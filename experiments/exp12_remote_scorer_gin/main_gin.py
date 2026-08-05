"""exp12 GIN 跨机真打分 harness — 复用 exp09 的 GIN kernel-split 传输,rank1 换成真 op。

对称堆布局(简化版,只发 q_quant + weights,K history 预置 rank1):
    上行 [0 .. UP_CAP):
        [0            ..) q_quant  (B*64*128 fp8 = B*8192 B)
        [B*8192       ..) weights  (B*64 fp32   = B*256  B)
    下行 [UP_CAP .. ):
        [UP_CAP       ..) topk     (B*2048 int32 = B*8192 B)

用法(两机各一条,payload 用 exp10 抓取的 FS_*decode*.pt):
    93:  RANK=0 ... torchrun ... main_gin.py --payload <p.pt>
    90:  RANK=1 ... torchrun ... main_gin.py --payload <p.pt>
两机都需要能读到同一份 payload(scp 一份到 90,或共享盘)。
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_scorer import (  # noqa: E402
    INDEX_HEAD_DIM,
    INDEX_N_HEADS,
    INDEX_TOPK,
    RemoteIndexerScorer,
    recall_vs_native,
)

UP_CAP = 4 * 1024 * 1024
Q_BYTES_PER = INDEX_N_HEADS * INDEX_HEAD_DIM        # 8192 (fp8, 1B each)
W_BYTES_PER = INDEX_N_HEADS * 4                     # 256  (fp32)
TOPK_BYTES_PER = INDEX_TOPK * 4                     # 8192 (int32)


def _get_native_nccl_comm(group: dist.ProcessGroup) -> int:
    backend = group._get_backend(torch.device("cuda"))
    for attr in ("_comm_ptr", "_get_comm_handle", "_ncclComm", "get_nccl_comm"):
        if hasattr(backend, attr):
            v = getattr(backend, attr)
            return int(v() if callable(v) else v)
    raise RuntimeError(f"no NCCL comm handle on {type(backend).__name__}")


def _load_lib() -> ctypes.CDLL:
    # 复用 exp09 编好的 GIN kernel(2.30.7 下已验证可编译)
    lib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "exp09_replay_demo", "librix_replay.so",
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path), mode=ctypes.RTLD_GLOBAL)
    lib.rix_gin_init.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int64, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_void_p)]
    lib.rix_gin_init.restype = ctypes.c_int
    lib.rix_gin_finalize.argtypes = [ctypes.c_void_p]
    lib.rix_gin_finalize.restype = ctypes.c_int
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


_CUDART = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
_CUDART.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_size_t, ctypes.c_int]
_CUDART.cudaMemcpy.restype = ctypes.c_int


def _d2d(dst_ptr: int, src: torch.Tensor):
    n = src.numel() * src.element_size()
    rc = _CUDART.cudaMemcpy(dst_ptr, src.contiguous().data_ptr(), n, 3)
    assert rc == 0, f"cudaMemcpy(to) rc={rc}"


def _d2d_read(dst: torch.Tensor, src_ptr: int):
    rc = _CUDART.cudaMemcpy(dst.data_ptr(), src_ptr,
                            dst.numel() * dst.element_size(), 3)
    assert rc == 0, f"cudaMemcpy(from) rc={rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, help="exp10 FS_*decode*.pt")
    ap.add_argument("--layers", type=int, default=61)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--symmetric-bytes", type=int, default=16 << 20)
    ap.add_argument("--num-qps", type=int, default=1)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert world == 2

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    dist.all_reduce(torch.zeros(1, device=device))
    torch.cuda.synchronize()

    comm_ptr = _get_native_nccl_comm(dist.group.WORLD)
    lib = _load_lib()
    ctx_ptr = ctypes.c_void_p(0)
    rc = lib.rix_gin_init(comm_ptr, rank, world, args.symmetric_bytes,
                          args.num_qps, ctypes.byref(ctx_ptr))
    assert rc == 0, f"[rank{rank}] gin_init rc={rc}"
    sym = int(lib.rix_symmetric_buffer(ctx_ptr))
    dist.barrier()

    payload = torch.load(args.payload, weights_only=False, map_location="cpu")
    B = payload["inputs"]["q_quant"].shape[0]
    up_bytes = B * (Q_BYTES_PER + W_BYTES_PER)
    down_bytes = B * TOPK_BYTES_PER
    w_off = B * Q_BYTES_PER

    if rank == 0:
        # 主实例侧代理:发真实抓取的 q_quant/weights,收 topk,对拍 native。
        q_quant = payload["inputs"]["q_quant"].to(device)          # [B,64,128] fp8
        weights = payload["inputs"]["weights"].to(device).float()  # [B,64] fp32
        native_topk = payload["output"].cpu()
        valid = int(payload["score"]["seq_lens"].max().item())
        recalls = []
        up_sig = 0
        dist.barrier()
        for it in range(args.warmup + args.iters):
            for layer in range(args.layers):
                _d2d(sym + 0, q_quant.view(torch.uint8))
                _d2d(sym + w_off, weights)
                assert lib.rix_r0_put_up(ctx_ptr, up_bytes) == 0
                up_sig += 1
                assert lib.rix_r0_wait_down(ctx_ptr, up_sig) == 0
                if it >= args.warmup and layer == 0:
                    got = torch.empty(B, INDEX_TOPK, dtype=torch.int32, device=device)
                    _d2d_read(got, sym + UP_CAP)
                    recalls.append(recall_vs_native(got.cpu(), native_topk, valid))
        dist.barrier()
        mean_r = sum(recalls) / len(recalls) if recalls else 0.0
        print("=" * 60)
        print(f"[exp12] GIN cross-machine, B={B}, layers={args.layers}, "
              f"iters={args.iters}")
        print(f"[exp12] valid candidate len = {valid}")
        print(f"[exp12] recall@{INDEX_TOPK} (remote GIN vs native) = {mean_r:.4f}")
        print("PASS ✓" if mean_r >= 0.99 else "FAIL ✗")
        print("=" * 60)
    else:
        # 远端 scorer:init 建 cache(自建 allocator),每层收 payload→真 op→回传。
        scorer = RemoteIndexerScorer(payload, device)
        q_buf = torch.empty(B, INDEX_N_HEADS, INDEX_HEAD_DIM,
                            dtype=torch.uint8, device=device)
        w_buf = torch.empty(B, INDEX_N_HEADS, dtype=torch.float32, device=device)
        down_sig = 0
        dist.barrier()
        for it in range(args.warmup + args.iters):
            for layer in range(args.layers):
                down_sig += 1
                assert lib.rix_r1_wait_up(ctx_ptr, down_sig) == 0
                _d2d_read(q_buf.view(-1), sym + 0)
                _d2d_read(w_buf.view(-1), sym + w_off)
                q_fp8 = q_buf.view(torch.float8_e4m3fn)
                topk = scorer.score(q_fp8, w_buf)              # 真 op, host launch, 数据在 GPU
                _d2d(sym + UP_CAP, topk.view(torch.uint8))
                assert lib.rix_r1_put_down(ctx_ptr, down_bytes) == 0
        dist.barrier()

    lib.rix_gin_finalize(ctx_ptr)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
