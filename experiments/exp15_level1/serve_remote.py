"""exp15 远端常驻 scorer 进程(rank1)—— 与 indexer_remote_client.py 配对(纯分离协议)。

建同一个 StatelessProcessGroup GIN comm(rank=1),然后进循环:
    wait_up → 从对称堆按 client._roundtrip 的布局解出 (q, weights, k, slots, block_table,
    seq_lens, mode) → ResidentIndexerScorer.score_decode/score_prefill(直接调 vLLM 原 op)
    → 写 topk 回下行区 → put_down。一直服务到主实例结束。

纯分离:远端持有和主实例**同形同尺寸**的 index-K cache(num_blocks 覆盖主实例物理块号),
收物理 slot + 真 block_table,block_table 负责 logical→physical,topk 逐比特等于本地。

用法:  python serve_remote.py <host_ip> <port> <max_model_len> <num_blocks> [block_size]
num_blocks/block_size 必须覆盖主实例 index-K cache 的物理块号(握手时由启动脚本传入)。
"""

from __future__ import annotations

import ctypes
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resident_scorer import (  # noqa: E402
    INDEX_HEAD_DIM,
    INDEX_N_HEADS,
    ResidentIndexerScorer,
)

UP_CAP = 4 * 1024 * 1024
LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "exp09_replay_demo", "librix_replay.so")


def _load_lib():
    lib = ctypes.CDLL(os.path.abspath(LIB), mode=ctypes.RTLD_GLOBAL)
    lib.rix_gin_init.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int64, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_void_p)]
    lib.rix_gin_init.restype = ctypes.c_int
    lib.rix_r1_wait_up.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.rix_r1_wait_up.restype = ctypes.c_int
    lib.rix_r1_put_down.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rix_r1_put_down.restype = ctypes.c_int
    lib.rix_symmetric_buffer.argtypes = [ctypes.c_void_p]
    lib.rix_symmetric_buffer.restype = ctypes.c_void_p
    return lib


def main():
    host, port = sys.argv[1], int(sys.argv[2])
    max_model_len = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    # cache 物理块总数由主实例握手时随帧头(cache_nb)带来,远端按此开同尺寸 cache;
    # block_size(64)固定与主实例一致。
    REMOTE_BLOCK_SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)

    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    pg = StatelessProcessGroup.create(host=host, port=port, rank=1, world_size=2,
                                      store_timeout=1800)
    pynccl = PyNcclCommunicator(pg, device=dev)
    comm = pynccl.comm
    comm_int = int(comm.value) if hasattr(comm, "value") else int(comm)

    lib = _load_lib()
    ctx = ctypes.c_void_p(0)
    rc = lib.rix_gin_init(comm_int, 1, 2, 16 << 20, 1, ctypes.byref(ctx))
    assert rc == 0, f"rix_gin_init rc={rc}"
    sym = int(lib.rix_symmetric_buffer(ctx))
    print(f"[serve pid={os.getpid()}] GIN ready {host}:{port}, serving...", flush=True)

    cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_size_t, ctypes.c_int]

    def _read(dst, off):
        rc = cudart.cudaMemcpy(dst.data_ptr(), sym + off,
                               dst.numel() * dst.element_size(), 3)
        assert rc == 0

    scorers = {}
    sig = 0
    while True:
        sig += 1
        if lib.rix_r1_wait_up(ctx, sig) != 0:
            break
        # 布局(与 indexer_remote_client._roundtrip 一致,纯分离协议 magic=0x16):
        #   [0:32]  head int32[8]: mode, num_tok, B, nb, seq_scalar, layer_id, 0, magic
        #   q_quant(num_tok*64*128 uint8) | weights(num_tok*64 fp32)
        #   | k_rows(num_tok*128 bf16) | slots(num_tok int32)  —— slots 是物理 slot
        #   | block_table(B*nb int32) | seq_lens(B int32)
        # 尾哨兵 = 本帧序号,自旋等它到齐(GIN relaxed ordering,尾到即整帧到)。
        head = torch.empty(8, dtype=torch.int32, device=dev)
        tail = torch.empty(1, dtype=torch.int32, device=dev)
        _spins = 0
        while True:
            _read(head, 0)
            mode, num_tok, B, nb, seq_scalar, layer_id, cache_nb, magic = [
                int(x) for x in head.tolist()
            ]
            if magic == 0x16 and num_tok > 0 and B > 0 and nb > 0:
                tail_off = (32
                            + num_tok * INDEX_N_HEADS * INDEX_HEAD_DIM
                            + num_tok * INDEX_N_HEADS * 4
                            + num_tok * INDEX_HEAD_DIM * 2
                            + num_tok * 4
                            + B * nb * 4
                            + B * 4)
                _read(tail, tail_off)
                if int(tail[0].item()) == sig:
                    break
            _spins += 1
            assert _spins < 5_000_000, f"frame#{sig} never fully landed head={head.tolist()}"
        o = 32
        q = torch.empty(num_tok, INDEX_N_HEADS, INDEX_HEAD_DIM,
                        dtype=torch.uint8, device=dev)
        _read(q, o); o += q.numel()
        w = torch.empty(num_tok, INDEX_N_HEADS, dtype=torch.float32, device=dev)
        _read(w, o); o += w.numel() * 4
        k_rows = torch.empty(num_tok, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=dev)
        _read(k_rows, o); o += num_tok * INDEX_HEAD_DIM * 2
        slots = torch.empty(num_tok, dtype=torch.int32, device=dev)
        _read(slots, o); o += num_tok * 4
        block_table = torch.empty(B, nb, dtype=torch.int32, device=dev)
        _read(block_table, o); o += B * nb * 4
        seq_lens = torch.empty(B, dtype=torch.int32, device=dev)
        _read(seq_lens, o); o += B * 4
        if os.environ.get("VLLM_INDEXER_REMOTE_DEBUG_V") == "1":
            print(f"[serve frame#{sig}] mode={mode} num_tok={num_tok} B={B} nb={nb} "
                  f"cache_nb={cache_nb} L={layer_id} spins={_spins} "
                  f"slots[:4]={slots[:min(4,num_tok)].tolist()}", flush=True)

        scorer = scorers.get(layer_id)
        if scorer is None:
            # 远端 cache 与主实例同尺寸(cache_nb 由主实例握手带来),block_table 里的物理
            # 块号才有意义。block_size 从主实例配置(64)固定。
            scorer = ResidentIndexerScorer(dev, cache_nb, REMOTE_BLOCK_SIZE, max_model_len)
            scorers[layer_id] = scorer
        q_fp8 = q.view(torch.float8_e4m3fn)
        if mode == 0:       # decode
            topk = scorer.score_decode(q_fp8, k_rows, w,
                                       slots.to(torch.int64), block_table, seq_lens)
        else:               # prefill
            topk = scorer.score_prefill(q_fp8, k_rows, w,
                                        slots.to(torch.int64), block_table, seq_scalar)

        rc = cudart.cudaMemcpy(sym + UP_CAP, topk.contiguous().data_ptr(),
                               topk.numel() * topk.element_size(), 3)
        assert rc == 0
        assert lib.rix_r1_put_down(ctx, topk.numel() * topk.element_size()) == 0

    print(f"[serve pid={os.getpid()}] stopped", flush=True)


if __name__ == "__main__":
    main()
