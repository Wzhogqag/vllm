"""exp15 方案A:验证重写后的 ResidentIndexerScorer(score_prefill / score_decode)。
单 GPU、无引擎。造历史 K + 当前 token,分别走 prefill 和 decode,检查 topk 落在逻辑帧
{0..S-1}(context < 2048 → 全选)。物理 slot 用 block 1 起(避开 null block),验证 op
经 block_table 映射后输出逻辑索引。

用法:  CUDA_VISIBLE_DEVICES=0 python check_scorer_v2.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/export/home/weizhongqiang.3/vllm")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 不触发主实例远端分支
_CFG = "/tmp/vllm_indexer_remote.json"
_BAK = None
if os.path.exists(_CFG):
    _BAK = _CFG + ".v2bak"
    os.rename(_CFG, _BAK)


def main() -> int:
    from vllm import _custom_ops as ops
    from resident_scorer import (
        ResidentIndexerScorer,
        INDEX_HEAD_DIM,
        INDEX_N_HEADS,
        QUANT_BLOCK_SIZE,
        SCALE_FMT,
    )

    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(0)

    BS = 64
    MAX_LEN = 1024
    num_blocks = (MAX_LEN + BS - 1) // BS + 2
    S = 40                      # prompt 长度(< BS,落物理 block 1)
    PHYS = 1                    # 起始物理块(避开 null block 0)

    scorer = ResidentIndexerScorer(dev, num_blocks, BS, MAX_LEN)

    # --- prefill:S 个 token,物理 slot 64..64+S-1,逻辑块 0→物理块 1 ---
    q_p = torch.randn(S, INDEX_N_HEADS, INDEX_HEAD_DIM, device=dev).to(torch.float8_e4m3fn)
    k_p = torch.randn(S, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=dev)
    w_p = torch.rand(S, INDEX_N_HEADS, dtype=torch.float32, device=dev)
    slot_p = torch.arange(PHYS * BS, PHYS * BS + S, dtype=torch.int64, device=dev)
    nb = (S + BS - 1) // BS + 1
    bt_p = torch.zeros(1, nb, dtype=torch.int32, device=dev)
    for j in range(nb):
        bt_p[0, j] = PHYS + j
    topk_p = scorer.score_prefill(q_p, k_p, w_p, slot_p, bt_p, S)
    torch.cuda.synchronize()

    ok_p = True
    for pos in range(S):
        valid = topk_p[pos][topk_p[pos] >= 0]
        got = sorted(int(x) for x in valid.tolist())
        if got != list(range(pos + 1)):
            ok_p = False
            print(f"  prefill row {pos}: got {got[:8]}.. expect 0..{pos}")
            break
    print(f"[check v2] prefill: rows={S} logical-frame-correct={ok_p}")

    # --- decode:接着 append 1 个 token(第 S 个),物理 slot 64+S ---
    tot = S + 1
    q_d = torch.randn(1, INDEX_N_HEADS, INDEX_HEAD_DIM, device=dev).to(torch.float8_e4m3fn)
    k_d = torch.randn(1, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=dev)
    w_d = torch.rand(1, INDEX_N_HEADS, dtype=torch.float32, device=dev)
    # 第 S 个 token 的逻辑位置 S → 逻辑块 S//BS,物理块 PHYS + S//BS,offset S%BS
    slot_d = torch.tensor(
        [(PHYS + S // BS) * BS + (S % BS)], dtype=torch.int64, device=dev
    )
    nb_d = (tot + BS - 1) // BS + 1
    bt_d = torch.zeros(1, nb_d, dtype=torch.int32, device=dev)
    for j in range(nb_d):
        bt_d[0, j] = PHYS + j
    seq_d = torch.tensor([tot], dtype=torch.int32, device=dev)
    topk_d = scorer.score_decode(q_d, k_d, w_d, slot_d, bt_d, seq_d)
    torch.cuda.synchronize()

    valid = topk_d[0][topk_d[0] >= 0]
    got = sorted(int(x) for x in valid.tolist())
    ok_d = got == list(range(tot))
    print(f"[check v2] decode: total={tot} num_valid={valid.numel()} "
          f"max={got[-1] if got else None} logical-frame-correct={ok_d}")
    if not ok_d:
        print(f"  got[:20]={got[:20]} ... expect 0..{tot-1}")

    ok = ok_p and ok_d
    print(f"[check v2] {'PASS — rewritten scorer reuses op correctly' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        if _BAK and os.path.exists(_BAK):
            os.rename(_BAK, _CFG)
    sys.exit(rc)
