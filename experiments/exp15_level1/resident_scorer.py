"""exp15 远端常驻 scorer(方案 A:纯分离 = 复用 vLLM 原 op)。

与旧版(逻辑帧 + 自选 paged kernel)的区别:这版**直接调 torch.ops.vllm.sparse_attn_indexer
整个 op**,把 logits 计算和 topk 选择的多算子 dispatch 全部交还给 vLLM —— 远端和主实例跑
的是同一段代码、同一套 kernel、同一坐标系,topk 逐比特一致,不再手写复刻、不再自创坐标。

核心:
- cache 形状/尺寸和主实例一致:(num_blocks, block_size, 132) uint8,num_blocks 由主实例握手
  时告知(不再用 ceil(max_model_len/bs)+1)。
- 收主实例传来的**物理 slot_mapping + 真 block_table + seq_lens**(prefill 再加
  query_start_loc),远端**不 remap**。K 用 op 自己的 indexer_k_quant_and_cache 写进物理 slot,
  block_table 负责 logical→physical,输出 topk 天然在主实例坐标系。
- decode:手工构造 DeepSeekV32IndexerDecodeMetadata;prefill:调 build_prefill_chunk_metadata
  (不硬拼 Triton 生成的 cu_seqlen_ks/ke)。
- 无 vLLM 引擎:init_workspace_manager 自助初始化 workspace;override_forward_context 塞入
  {prefix: meta} 绕开 VllmConfig。

依赖(smoke_op_reuse.py 已验证独立进程可跑):DeepGEMM 已装、CUDA、SM90+。
"""

from __future__ import annotations

import torch

from vllm import _custom_ops as ops
# Import the module (not just names) to guarantee direct_register_custom_op runs
# and torch.ops.vllm.sparse_attn_indexer exists.
import vllm.model_executor.layers.sparse_attn_indexer  # noqa: F401
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillMetadata,
    build_prefill_chunk_metadata,
)
from vllm.v1.worker.workspace import init_workspace_manager

INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_HEAD_WIDTH = 132          # 128 fp8 + 4 fp32 scale
INDEX_TOPK = 2048
QUANT_BLOCK_SIZE = 128
SCALE_FMT = "ue8m0"

_WORKSPACE_READY = False


def _ensure_workspace(device: torch.device) -> None:
    global _WORKSPACE_READY
    if not _WORKSPACE_READY:
        init_workspace_manager(device)
        _WORKSPACE_READY = True


class ResidentIndexerScorer:
    """单层 index-K cache 的远端打分器,直接调 vLLM 原 op。

    每层一个实例(serve 按 layer_id 分开持有)。cache 与主实例同形:
    (num_blocks, block_size, 132)。收物理 slot + block_table + seq_lens,调 op。
    """

    def __init__(self, device: torch.device, num_blocks: int, block_size: int,
                 max_model_len: int, prefix: str = "remote.indexer.k_cache"):
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_model_len = max_model_len
        self.prefix = prefix
        self.total_seq_lens_ub = max_model_len * 40   # workspace 上界(见 indexer.py:444)
        _ensure_workspace(device)
        # vLLM 同形 cache:(num_blocks, block_size, 132) uint8。
        self.kv_cache = torch.zeros(
            num_blocks, block_size, INDEX_HEAD_WIDTH, dtype=torch.uint8, device=device
        )
        # 占位 hidden(op 不真用它,只在 profiling 分支和 shape 推断时碰;这里给最小 shape)。
        self._hidden_dummy = torch.empty(0, dtype=torch.bfloat16, device=device)

    def _run_op(self, meta: DeepseekV32IndexerMetadata, q_quant: torch.Tensor,
                k: torch.Tensor, weights: torch.Tensor,
                num_tokens: int) -> torch.Tensor:
        """构造 forward_context,调 op,返回 topk_indices_buffer[:num_tokens]。"""
        topk_buf = torch.full(
            (num_tokens, INDEX_TOPK), -1, dtype=torch.int32, device=self.device
        )
        hidden = self._hidden_dummy
        if hidden.shape[0] < num_tokens:
            hidden = torch.empty(num_tokens, 1, dtype=torch.bfloat16, device=self.device)
        fwd = ForwardContext(
            no_compile_layers={},
            attn_metadata={self.prefix: meta},
            slot_mapping={},
        )
        with override_forward_context(fwd):
            torch.ops.vllm.sparse_attn_indexer(
                hidden,
                _encode_layer_name(self.prefix),
                self.kv_cache,
                q_quant,
                None,                       # q_scale (FP8 path)
                k,                          # 当前 token 的 raw index_K,op 自己写 cache
                weights,
                QUANT_BLOCK_SIZE,
                SCALE_FMT,
                INDEX_TOPK,
                INDEX_HEAD_DIM,
                self.max_model_len,
                self.total_seq_lens_ub,
                topk_buf,
                False,                      # skip_k_cache_insert:让 op 写 K
                False,                      # use_pcp
                False,                      # use_fp4_cache
            )
        return topk_buf

    def score_decode(self, q_quant: torch.Tensor, k: torch.Tensor,
                     weights: torch.Tensor, slot_mapping: torch.Tensor,
                     block_table: torch.Tensor,
                     seq_lens: torch.Tensor) -> torch.Tensor:
        """decode:B 个单 token query。

        q_quant [B,64,128] fp8, k [B,128] bf16(当前 token 的 index_K),
        weights [B,64] fp32, slot_mapping [B] int64(物理 slot),
        block_table [B, nb] int32(逻辑→物理), seq_lens [B] int32(含当前 token 的总长)。
        """
        B = q_quant.shape[0]
        seq_lens_2d = seq_lens.to(torch.int32).view(B, 1)
        sched = get_paged_mqa_logits_metadata(seq_lens_2d, self.block_size, get_num_sms())
        decode_meta = DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table.to(torch.int32),
            seq_lens=seq_lens_2d,
            decode_lens=torch.ones(B, dtype=torch.int32, device=self.device),
            requires_padding=False,
            schedule_metadata=sched,
            global_seq_lens=None,
        )
        meta = DeepseekV32IndexerMetadata(
            seq_lens=seq_lens.to(torch.int32),
            max_seq_len=int(seq_lens.max().item()),
            slot_mapping=slot_mapping.to(torch.int64),
            num_decodes=B,
            num_decode_tokens=B,
            num_prefills=0,
            num_prefill_tokens=0,
            decode=decode_meta,
            prefill=None,
        )
        return self._run_op(meta, q_quant, k, weights, B)

    def score_prefill(self, q_quant: torch.Tensor, k: torch.Tensor,
                      weights: torch.Tensor, slot_mapping: torch.Tensor,
                      block_table: torch.Tensor, seq_len: int) -> torch.Tensor:
        """prefill(单请求):num_tok 个 query,一次性铺满。

        q_quant [S,64,128] fp8, k [S,128] bf16, weights [S,64] fp32,
        slot_mapping [S] int64(物理 slot), block_table [1, nb] int32, seq_len=S。
        """
        S = q_quant.shape[0]
        query_start_loc = torch.tensor([0, S], dtype=torch.int32, device=self.device)
        query_start_loc_cpu = torch.tensor([0, S], dtype=torch.int32)
        seq_lens_dev = torch.tensor([seq_len], dtype=torch.int32, device=self.device)
        seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int32)
        chunk = build_prefill_chunk_metadata(
            start_idx=0,
            end_idx=1,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            uncompressed_seq_lens=seq_lens_dev,
            compressed_seq_lens=seq_lens_dev,
            compressed_seq_lens_cpu=seq_lens_cpu,
            block_table=block_table.to(torch.int32),
            compress_ratio=1,
        )
        assert chunk is not None
        prefill_meta = DeepseekV32IndexerPrefillMetadata(chunks=[chunk])
        meta = DeepseekV32IndexerMetadata(
            seq_lens=seq_lens_dev,
            max_seq_len=seq_len,
            slot_mapping=slot_mapping.to(torch.int64),
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=1,
            num_prefill_tokens=S,
            decode=None,
            prefill=prefill_meta,
        )
        return self._run_op(meta, q_quant, k, weights, S)
