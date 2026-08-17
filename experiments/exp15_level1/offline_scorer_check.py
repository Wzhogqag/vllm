"""exp15 离线自检:不跨机、不起 vLLM,单 GPU 直接验证 ResidentIndexerScorer 的
逻辑帧打分正确性 —— 这是把跨机全量跑之前最便宜的正确性关卡。

验证点(正是这次修复的核心):
  1. 把 K 写到**逻辑** slot 0..S-1(不是物理 64..72),paged 打分不再 illegal access。
  2. 返回的 topk 索引在**逻辑帧** 0..S-1(不是物理 64..72),证明坐标系对齐主实例。
  3. seq_len < 2048 时每个 query 选中全部历史 → topk 应恰好 = {0..pos},其余 -1 padding。
     (indexer 选 top-2048,历史不足 2048 时就是全集,顺序无关,所以判据确定。)

用法:  CUDA_VISIBLE_DEVICES=<free_gpu> python offline_scorer_check.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resident_scorer import (  # noqa: E402
    INDEX_HEAD_DIM,
    INDEX_N_HEADS,
    ResidentIndexerScorer,
)


def _fake_qkw(num_tok: int, device: torch.device):
    """造 num_tok 个 token 的 index_K(bf16 [num_tok,128])、q_fp8([num_tok,64,128]
    fp8)、weights([num_tok,64] fp32)。值不重要 —— 只要打分不 NaN 且 topk 命中全集。"""
    torch.manual_seed(0)
    k = torch.randn(num_tok, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    q = torch.randn(
        num_tok, INDEX_N_HEADS, INDEX_HEAD_DIM, device=device
    ).to(torch.float8_e4m3fn)
    w = torch.rand(num_tok, INDEX_N_HEADS, dtype=torch.float32, device=device)
    return q, k, w


def main() -> int:
    assert torch.cuda.is_available(), "need a CUDA device"
    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    S = 9                       # 和真实 baseline prompt 一样 9 token
    bs = 64
    scorer = ResidentIndexerScorer(dev, max_model_len=1024, block_size=bs)

    q, k, w = _fake_qkw(S, dev)
    logical = torch.arange(S, dtype=torch.int32, device=dev)  # 逻辑 slot 0..S-1

    # 写 K 到逻辑 slot,再打分(prefill:per-query causal = pos+1)。
    scorer.write_k_at_slots(k, logical, S)
    per_q = logical + 1
    topk = scorer.score(q, w, per_query_seq=per_q)  # [S, 2048] int32
    torch.cuda.synchronize()

    ok = True
    for pos in range(S):
        row = topk[pos]
        valid = row[row >= 0]
        got = set(int(x) for x in valid.tolist())
        expect = set(range(pos + 1))            # causal:query pos 可见 0..pos
        if got != expect:
            ok = False
            print(f"  row {pos}: got {sorted(got)[:12]}... expect {sorted(expect)}")
        # 关键:任何索引都不能落在物理帧(>=64 说明还在用物理 slot)
        if valid.numel() and int(valid.max().item()) >= S:
            ok = False
            print(f"  row {pos}: index {int(valid.max())} >= S — WRONG FRAME (physical?)")

    print(f"[offline_scorer_check] S={S} topk.shape={tuple(topk.shape)} "
          f"logical-frame-correct={ok}")

    # --- decode 增量:再 append 1 个 token 到逻辑 slot S,单 query 打分(n=1)---
    # 这验证"随 decode 追加 K"的语义:新 token 落逻辑 slot S,历史 0..S-1 已在 cache,
    # score 单行应命中全集 {0..S}。这是 per-layer-cache 修复要保证的路径。
    qd, kd, wd = _fake_qkw(1, dev)
    slot_new = torch.tensor([S], dtype=torch.int32, device=dev)
    scorer.write_k_at_slots(kd, slot_new, S + 1)
    topk_d = scorer.score(qd, wd, per_query_seq=None)   # decode:用 self.seq_len=S+1
    torch.cuda.synchronize()
    row = topk_d[0]
    valid = row[row >= 0]
    got = set(int(x) for x in valid.tolist())
    expect = set(range(S + 1))                           # decode token 见全部 0..S
    decode_ok = got == expect and (
        not valid.numel() or int(valid.max().item()) <= S
    )
    if not decode_ok:
        print(f"  decode row: got {sorted(got)[:12]}... expect {sorted(expect)}")
    print(f"[offline_scorer_check] decode append: seq_len={S + 1} "
          f"decode-correct={decode_ok}")
    ok = ok and decode_ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
