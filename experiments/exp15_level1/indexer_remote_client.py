"""exp15 主实例侧 remote indexer 客户端(改动 2)。

进程级单例:worker load 时(首次真实 indexer 调用)lazy-init GIN comm
(StatelessProcessGroup + PyNcclCommunicator,#16 已验证),之后每层 indexer:
    抠出本步涉及的 K 整块(SoA 保序)→ 打包 q/weights/blocks 到对称堆 → GIN put →
    阻塞 wait → 读回 topk。

只在 TP-rank0 建 comm + 收发(决策 C);其余 rank 由 sparse_attn_indexer 分支里的
tp.broadcast 收结果,不进这里。

对称堆布局(上行):
    [0                     ..) q_quant   (num_tok*64*128 fp8)
    [+num_tok*8192         ..) weights   (num_tok*64 fp32)
    [+num_tok*256          ..) blocks    (nb*bs*132 uint8)  —— 整块传,保 SoA
    [+ ...                 ..) meta      (int32: num_tok, nb, seq_len, block_ids[nb])
下行:
    [UP_CAP                ..) topk      (num_tok*2048 int32)
"""

from __future__ import annotations

import ctypes
import os

import torch

INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_HEAD_WIDTH = 132
INDEX_TOPK = 2048
UP_CAP = 4 * 1024 * 1024

_CLIENT = None


def indexer_remote_enabled() -> bool:
    return os.environ.get("VLLM_INDEXER_REMOTE", "0") == "1"


def get_client():
    """进程级单例。首次调用 lazy-init GIN(阻塞 rendezvous 一次)。"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _RemoteIndexerClient()
    return _CLIENT


class _RemoteIndexerClient:
    def __init__(self):
        self.host = os.environ["VLLM_INDEXER_REMOTE_HOST"]
        self.port = int(os.environ.get("VLLM_INDEXER_REMOTE_PORT", "29920"))
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._lib = self._load_lib()
        self._cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
        self._cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_size_t, ctypes.c_int]
        # StatelessProcessGroup GIN 建连(#16 验证过的路径),rank0=vLLM 侧。
        from vllm.distributed.utils import StatelessProcessGroup
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        pg = StatelessProcessGroup.create(
            host=self.host, port=self.port, rank=0, world_size=2,
            store_timeout=1800,
        )
        self._pynccl = PyNcclCommunicator(pg, device=self.device)
        comm = self._pynccl.comm
        comm_int = int(comm.value) if hasattr(comm, "value") else int(comm)
        self._ctx = ctypes.c_void_p(0)
        rc = self._lib.rix_gin_init(comm_int, 0, 2, 16 << 20, 1,
                                    ctypes.byref(self._ctx))
        assert rc == 0, f"rix_gin_init rc={rc}"
        self._sym = int(self._lib.rix_symmetric_buffer(self._ctx))
        self._up_sig = 0
        print(f"[indexer_remote pid={os.getpid()}] GIN client ready "
              f"{self.host}:{self.port}", flush=True)

    def _load_lib(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(
            here, "..", "exp09_replay_demo", "librix_replay.so"))
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        lib.rix_gin_init.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int64, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_void_p)]
        lib.rix_gin_init.restype = ctypes.c_int
        lib.rix_r0_put_up.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.rix_r0_put_up.restype = ctypes.c_int
        lib.rix_r0_wait_down.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.rix_r0_wait_down.restype = ctypes.c_int
        lib.rix_symmetric_buffer.argtypes = [ctypes.c_void_p]
        lib.rix_symmetric_buffer.restype = ctypes.c_void_p
        return lib

    def _d2d(self, dst_ptr, src):
        rc = self._cudart.cudaMemcpy(
            dst_ptr, src.contiguous().data_ptr(),
            src.numel() * src.element_size(), 3)
        assert rc == 0

    def roundtrip_decode(self, q_quant, weights, k_rows, slots,
                         block_table, seq_lens, layer_id, cache_nb):
        """decode 往返:B 个单 token query。返回 topk [B, 2048] int32。

        q_quant [B,64,128] fp8, weights [B,64] fp32, k_rows [B,128] bf16,
        slots [B] int32(物理 slot), block_table [B, nb] int32, seq_lens [B] int32,
        cache_nb int(主实例 index-K cache 物理块总数,远端按此开同尺寸 cache)。
        """
        B = q_quant.shape[0]
        nb = block_table.shape[1]
        return self._roundtrip(
            mode=0, B=B, num_tok=B, nb=nb, seq_scalar=0, cache_nb=cache_nb,
            q_quant=q_quant, weights=weights, k_rows=k_rows, slots=slots,
            block_table=block_table, seq_lens=seq_lens, layer_id=layer_id,
        )

    def roundtrip_prefill(self, q_quant, weights, k_rows, slots,
                          block_table, seq_len, layer_id, cache_nb):
        """prefill 往返(单请求):S 个 query。返回 topk [S, 2048] int32。

        block_table [1, nb] int32, seq_len int(=S)。seq_lens 传空(prefill 用 seq_scalar)。
        """
        S = q_quant.shape[0]
        nb = block_table.shape[1]
        empty = torch.zeros(1, dtype=torch.int32, device=self.device)
        return self._roundtrip(
            mode=1, B=1, num_tok=S, nb=nb, seq_scalar=seq_len, cache_nb=cache_nb,
            q_quant=q_quant, weights=weights, k_rows=k_rows, slots=slots,
            block_table=block_table, seq_lens=empty, layer_id=layer_id,
        )

    def _roundtrip(self, mode, B, num_tok, nb, seq_scalar, cache_nb,
                   q_quant, weights, k_rows, slots, block_table, seq_lens, layer_id):
        """纯分离协议的一次往返。远端用 block_table 做 logical→physical,调原 op。

        对称堆布局(上行),尾哨兵保证整帧落地:
            [0:32]  head int32[8]: mode, num_tok, B, nb, seq_scalar, layer_id, cache_nb, magic
            q_quant (num_tok*64*128 uint8) | weights (num_tok*64 fp32)
            | k_rows (num_tok*128 bf16) | slots (num_tok int32)
            | block_table (B*nb int32) | seq_lens (B int32)
            [tail 4B] frame_seq
        下行 @UP_CAP: topk (num_tok*2048 int32)
        """
        magic = 0x16          # 协议版本:纯分离(区别于旧逻辑帧 0x15)
        head = torch.tensor(
            [mode, num_tok, B, nb, seq_scalar, layer_id, cache_nb, magic],
            dtype=torch.int32, device=self.device,
        )
        frame_seq = self._up_sig + 1
        tail = torch.tensor([frame_seq], dtype=torch.int32, device=self.device)
        o = 0
        self._d2d(self._sym + o, head); o += 8 * 4
        self._d2d(self._sym + o, q_quant.view(torch.uint8))
        o += num_tok * INDEX_N_HEADS * INDEX_HEAD_DIM
        self._d2d(self._sym + o, weights.to(torch.float32))
        o += num_tok * INDEX_N_HEADS * 4
        self._d2d(self._sym + o, k_rows.to(torch.bfloat16))
        o += num_tok * INDEX_HEAD_DIM * 2
        self._d2d(self._sym + o, slots.to(torch.int32)); o += num_tok * 4
        self._d2d(self._sym + o, block_table.to(torch.int32)); o += B * nb * 4
        self._d2d(self._sym + o, seq_lens.to(torch.int32)); o += B * 4
        self._d2d(self._sym + o, tail); o += 4
        up_bytes = o
        if os.environ.get("VLLM_INDEXER_REMOTE_HOST") and \
                os.path.exists("/tmp/vllm_indexer_remote_debug"):
            print(f"[client send#{frame_seq}] head={head.tolist()} "
                  f"up_bytes={up_bytes}", flush=True)
        assert self._lib.rix_r0_put_up(self._ctx, up_bytes) == 0
        self._up_sig += 1
        assert self._lib.rix_r0_wait_down(self._ctx, self._up_sig) == 0
        topk = torch.empty(num_tok, INDEX_TOPK, dtype=torch.int32, device=self.device)
        rc = self._cudart.cudaMemcpy(
            topk.data_ptr(), self._sym + UP_CAP,
            topk.numel() * topk.element_size(), 3)
        assert rc == 0
        return topk
