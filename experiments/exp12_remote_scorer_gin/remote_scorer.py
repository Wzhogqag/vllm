"""exp12 — Level 0.2 GIN 跨机远端 indexer scorer(真 op 打分)。

在 exp09 的 GIN kernel-split 骨架上,把 rank1 的**假打分**(pytorch einsum + 随机 K)
换成**真打分**:exp11 验证过的自建 allocator + 两个真 vLLM op。

拓扑(2 rank,跨机):
    rank0 (93, 发起端 / 主实例侧代理):
        - 加载 exp10 full-score 抓取的真实 decode payload
        - 每层:GIN put 送 (q_quant, weights) 到对称堆 → 等 rank1 回 topk
        - 对拍抓取的 vLLM 原生 topk → recall@2048
    rank1 (90, indexer 端 / 远端 scorer):
        - init 时从抓取数据把 index-K cache 建好(自建 allocator,物理块 0..nb-1)
        - 每层:GIN 等 payload 到 → 从对称堆读 (q_quant, weights)
                → fp8_fp4_paged_mqa_logits + persistent_topk(真 op,host launch,数据在 GPU 对称堆/显存)
                → 写回对称堆 → GIN put 送回

关键(回答"能不能 kernel 内算"):不能。两个 op 是 host-launch 的 CUDA kernel,
只能 host 侧发起;但数据全程在 GPU 显存(对称堆),host 只发指令,不落 CPU。
GIN 通信在 kernel 内,计算在 host,两者通过共享对称堆显存衔接。

用法(两机各一条):
    93:  ./run_gin.sh 0 <93_ib_ip> <payload.pt>
    90:  ./run_gin.sh 1 <93_ib_ip> <payload.pt>
"""

from __future__ import annotations

import torch

# bare import vllm —— 只为调两个 op + 布局函数,不 boot 引擎。
from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)

INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048
RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


class RemoteIndexerScorer:
    """rank1 侧的远端打分核心。init 时用自建 allocator 建好 index-K cache,
    之后每次 score(q_quant, weights) 用真 op 算 topk。

    自建 allocator 的体现:cache 的物理块号是本地 0..nb-1,与主实例(rank0)原始
    物理块号无关 —— exp11 已证明这不改变 topk(逻辑位置对齐即可)。
    """

    def __init__(self, payload: dict, device: torch.device):
        self.device = device
        score = payload["score"]
        used = score["kv_cache_used"]

        # 自建 allocator:抓到的物理块内容按 0..nb-1 顺序放进本地 cache。
        blocks = used["blocks"].to(device).contiguous()   # [nb,64,132] uint8
        orig_ids = used["orig_block_ids"].to(device)
        self.block_size = blocks.shape[1]
        self.kv_cache = kv_cache_as_quant_view(blocks, INDEX_HEAD_DIM, use_fp4_cache=False)

        # 重映射 block_table:原始物理块号 → 本地 0..nb-1
        orig_bt = score["block_table"].to(device)          # [B,max_blocks] int32
        id_map = {int(o): i for i, o in enumerate(orig_ids.tolist())}
        new_bt = torch.zeros_like(orig_bt)
        obt = orig_bt.cpu()
        for r in range(obt.shape[0]):
            for c in range(obt.shape[1]):
                new_bt[r, c] = id_map.get(int(obt[r, c].item()), 0)
        self.block_table = new_bt

        self.seq_lens = score["seq_lens"].to(device).to(torch.int32)  # [B,1]
        self.max_model_len = int(score["max_model_len"])
        self.B = self.seq_lens.shape[0]
        self.next_n = 1
        self.ctx_2d = self.seq_lens.view(self.B, self.next_n).contiguous()
        self.sched = get_paged_mqa_logits_metadata(
            self.ctx_2d, self.block_size, get_num_sms()
        )
        self.workspace = torch.empty(
            RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=device
        )
        self.max_seq_len = int(self.ctx_2d.max().item())

    def score(self, q_quant: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """真 op 打分,返回 topk [B,2048] int32。数据全程在 GPU 显存。"""
        B = q_quant.shape[0]
        q_paged = q_quant.view(B, self.next_n, INDEX_N_HEADS, INDEX_HEAD_DIM).contiguous()
        logits = fp8_fp4_paged_mqa_logits(
            (q_paged, None),
            self.kv_cache,
            weights,
            self.ctx_2d,
            self.block_table,
            self.sched,
            self.max_model_len,
            clean_logits=False,
        )
        out_idx = torch.empty(B * self.next_n, INDEX_TOPK, dtype=torch.int32,
                              device=self.device)
        torch.ops._C.persistent_topk(
            logits, self.ctx_2d, out_idx, self.workspace, INDEX_TOPK, self.max_seq_len
        )
        return out_idx


def recall_vs_native(ours: torch.Tensor, ref: torch.Tensor, valid_len: int) -> float:
    """set-overlap recall,只在有效候选范围内比。"""
    B = ours.shape[0]
    hit = denom = 0
    for i in range(B):
        a = set(x for x in ours[i].tolist() if 0 <= x < valid_len)
        b = set(x for x in ref[i].tolist() if 0 <= x < valid_len)
        hit += len(a & b)
        denom += len(b) if b else min(valid_len, INDEX_TOPK)
    return hit / max(denom, 1)
