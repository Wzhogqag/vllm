"""exp11 — remote indexer scorer 原型(离线对拍)。

验证分离方案的核心声明:一个 bare `import vllm` 的独立进程,用**自建 allocator**
(物理 block id 与主实例不同)+ 真实的两个 vLLM 打分 op,能算出和 vLLM 原生
**一致的 top-2048**。

数据来自 exp10 的 full-score 抓取(FS_*decode*.pt),含:
    inputs.q_quant [1,64,128] fp8, inputs.weights [1,64] fp32
    score.kv_cache_used = {blocks [nb,64,132] uint8, orig_block_ids}
    score.block_table [1,16] int32(引用原始物理块号)
    score.seq_lens [1,1] int32
    output = vLLM 原生 topk [1,2048] int32  ← ground truth

自建 allocator 的体现(L-a):把抓到的物理块重新编号成 0,1,2,...(和原始不同),
重映射 block_table,再调 paged 打分 op。若 recall≈1.0,证明物理 id 奇偶性无关,
远端自建分配正确。

用法:
    python replay_scorer.py <path-to-FS_*decode*.pt>
"""

from __future__ import annotations

import sys

import torch

# bare import vllm —— 只为调这两个 op,不 boot 引擎/模型/调度器。
from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)

INDEX_TOPK = 2048
RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def _recall(ours: torch.Tensor, ref: torch.Tensor, valid_len: int) -> float:
    """两组 topk 索引的 set-overlap recall,只在有效范围内比(padding 不算)。
    ours/ref: [B, 2048] int32(CPU)。valid_len: 该行真实候选数(<2048 时只比有效部分)。
    """
    B = ours.shape[0]
    hit = 0
    denom = 0
    for i in range(B):
        n = min(valid_len, INDEX_TOPK)
        a = set(x for x in ours[i].tolist() if 0 <= x < valid_len)
        b = set(x for x in ref[i].tolist() if 0 <= x < valid_len)
        hit += len(a & b)
        denom += min(n, len(b)) if b else n
    return hit / max(denom, 1)


def replay_paged_self_alloc(payload: dict) -> dict:
    """L-a:自建 allocator(块重新编号)+ paged 打分 op,对拍原生 topk。"""
    dev = torch.device("cuda")
    sig = payload["sig"]
    score = payload["score"]
    inp = payload["inputs"]

    q_quant = inp["q_quant"].to(dev)          # [B,64,128] fp8
    weights = inp["weights"].to(dev)          # [B,64] fp32
    B = q_quant.shape[0]
    next_n = 1

    used = score["kv_cache_used"]
    blocks = used["blocks"].to(dev)           # [nb,64,132] uint8 (原始物理块内容)
    orig_ids = used["orig_block_ids"].to(dev)  # [nb] 原始物理块号
    orig_bt = score["block_table"].to(dev)    # [B,max_blocks] int32(引用原始块号)
    seq_lens = score["seq_lens"].to(dev).to(torch.int32)  # [B,1]
    max_model_len = int(score["max_model_len"])
    block_size = blocks.shape[1]

    # ---- 自建 allocator:把原始物理块号重新编号成 0,1,2,... ----
    # 新 cache 只放这 nb 个块,顺序即 orig_ids 的顺序 → 新块号 = 0..nb-1。
    # ---- 自建 allocator:把原始物理块号重新编号 ----
    # 为了真正证明"物理 id 无关",这里故意用一个 NON-trivial 排列(倒序),
    # 让新块号明确不同于原始的连续 0..nb-1;并把块内容按同一排列搬到新位置。
    nb = blocks.shape[0]
    perm = list(range(nb))[::-1]              # 倒序:原块 i → 新位置 perm.index(i)
    # new_pos[i] = i 号原块在新 cache 里的物理下标
    new_pos = [0] * nb
    for new_idx, old_local in enumerate(perm):
        new_pos[old_local] = new_idx
    # 按新排列搬块:new_cache_3d[new_idx] = blocks[old_local]
    new_cache_3d = torch.empty_like(blocks)
    for new_idx, old_local in enumerate(perm):
        new_cache_3d[new_idx] = blocks[old_local]
    new_cache_3d = new_cache_3d.contiguous()
    # DeepGEMM paged op 要 4D view [nb,64,1,132]。用 vLLM 自己的函数保证布局一致。
    head_dim = int(sig.get("head_dim", 128)) if isinstance(sig, dict) else 128
    new_cache = kv_cache_as_quant_view(new_cache_3d, head_dim, use_fp4_cache=False)

    # 原始物理块号 → 新物理块号:先 orig_id → 本地下标(0..nb-1),再 → 新排列位置。
    orig_to_local = {int(o): i for i, o in enumerate(orig_ids.tolist())}
    id_map = {oid: new_pos[local] for oid, local in orig_to_local.items()}
    # 重映射 block_table:每个原始块号换成新物理块号(未用槽保持 0)
    new_bt = torch.zeros_like(orig_bt)
    obt = orig_bt.cpu()
    for r in range(obt.shape[0]):
        for c in range(obt.shape[1]):
            v = int(obt[r, c].item())
            new_bt[r, c] = id_map.get(v, 0)

    # paged op 要 2D context_lens (B,next_n)
    ctx_2d = seq_lens.view(B, next_n).contiguous()
    sched = get_paged_mqa_logits_metadata(ctx_2d, block_size, get_num_sms())

    # q reshape 成 [B,next_n,H,D]
    q_paged = q_quant.view(B, next_n, q_quant.shape[1], q_quant.shape[2]).contiguous()

    logits = fp8_fp4_paged_mqa_logits(
        (q_paged, None),
        new_cache,
        weights,
        ctx_2d,
        new_bt,
        sched,
        max_model_len,
        clean_logits=False,
    )  # [B*next_n, max_model_len] fp32

    # ---- topk ----
    out_idx = torch.empty(B * next_n, INDEX_TOPK, dtype=torch.int32, device=dev)
    workspace = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)
    max_seq_len = int(ctx_2d.max().item())
    torch.ops._C.persistent_topk(
        logits, ctx_2d, out_idx, workspace, INDEX_TOPK, max_seq_len
    )

    ref = payload["output"].cpu()             # vLLM 原生 topk
    ours = out_idx.cpu()
    valid = int(ctx_2d.max().item())
    rec = _recall(ours, ref, valid)

    return {
        "recall": rec,
        "valid_len": valid,
        "B": B,
        "num_blocks_self_alloc": nb,
        "orig_block_ids": orig_ids.tolist(),
        "remapped_to": [id_map[int(o)] for o in orig_ids.tolist()],
        "ours_head": ours[0, :8].tolist(),
        "ref_head": ref[0, :8].tolist(),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: python replay_scorer.py <FS_*decode*.pt>")
        sys.exit(1)
    path = sys.argv[1]
    payload = torch.load(path, weights_only=False, map_location="cpu")
    assert payload["sig"]["bucket"] == "decode", "用 decode 抓取文件(paged 路径最干净)"

    print(f"[replay] loaded {path}")
    print(f"[replay] layer={payload['sig']['layer_name']} "
          f"num_rows={payload['sig']['num_rows']}")

    res = replay_paged_self_alloc(payload)
    print("=" * 60)
    print(f"self-allocated block ids: {res['orig_block_ids']} → {res['remapped_to']}")
    print(f"valid candidate len (seq): {res['valid_len']}")
    print(f"ours topk head: {res['ours_head']}")
    print(f"ref  topk head: {res['ref_head']}")
    print(f"RECALL@{INDEX_TOPK} (within valid range): {res['recall']:.4f}")
    print("=" * 60)
    if res["recall"] >= 0.99:
        print("PASS ✓ — self-allocator + real ops reproduce vLLM-native topk")
    else:
        print("FAIL ✗ — recall below 0.99, investigate")


if __name__ == "__main__":
    main()
