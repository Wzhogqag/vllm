"""exp13 — prefill 路径离线打分器(对拍 native topk)。

exp11 只验了 decode 路径(paged op:fp8_fp4_paged_mqa_logits + persistent_topk)。
prefill 走的是**完全不同的一对 op**,这里补上打分面的另一半:

    ① logits: fp8_fp4_mqa_logits    —— 非 paged、varlen,按 cu_seqlen_ks/ke 逐行因果
    ② topk  : ops.top_k_per_row_prefill

关键洞察(为什么能纯离线、不用 block_table):
    抓取的 prefill payload 里 block_table/seq_lens 都是 None,但 **slot_mapping 抓到了**
    (本 forward 每个 token 写进 cache 的物理槽号)。首个 prefill 无历史上下文,query 在
    逻辑位置 i 的因果候选就是逻辑位置 [0, i] 的 K,而逻辑位置 j 的 K 就在 slot_mapping[j]。
    所以直接把 full cache 拍平、按 slot_mapping gather 出 [N,132],就是这次 prefill 的
    全部候选 K —— 无需 block_table。这正是 vLLM 里 cp_gather_indexer_k_quant_cache 干的事
    (gather 进连续 workspace),我们用 slot_mapping 离线复现。

    132 字节/token = 128 fp8 值 + 4 字节 fp32 scale(见 _gather_workspace_shapes:
    FP8 path = (T,head_dim) fp8 + (T,4) uint8 fp32 scales)。

    cu_seqlen_ks[i]=0, cu_seqlen_ke[i]=i+1(因果:query i 看 key [0,i])。
    native topk 的有效候选数逐行 = 1..N,正好印证这个因果结构。

用法:
    python prefill_scorer.py <path-to-FS_*prefill*.pt>
    python prefill_scorer.py --sweep <run_dir>     # 扫该目录下所有 prefill 层
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

# bare import vllm —— 只为调这两个 prefill op,不 boot 引擎/模型/调度器。
from vllm import _custom_ops as ops
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048
CACHE_ENTRY_BYTES = 132  # 128 fp8 值 + 4 fp32 scale


def recall_vs_native(ours: torch.Tensor, ref: torch.Tensor) -> float:
    """逐行 set-overlap recall。prefill 每行有效候选数不同(因果三角),
    所以按每行各自的有效集合(ref 里 >=0 的索引)来比,padding(-1)不计。
    """
    M = ours.shape[0]
    hit = denom = 0
    for i in range(M):
        a = {x for x in ours[i].tolist() if x >= 0}
        b = {x for x in ref[i].tolist() if x >= 0}
        hit += len(a & b)
        denom += len(b)
    return hit / max(denom, 1)


def reconstruct_candidate_k(payload: dict, device: torch.device):
    """用 slot_mapping 从 full cache gather 出连续候选 K,拆成 (fp8 值, fp32 scale)。

    Returns:
        k_values: [N,128] float8_e4m3fn
        k_scale:  [N] float32
    """
    score = payload["score"]
    kv_cache = score["kv_cache_used"]
    assert torch.is_tensor(kv_cache), (
        "prefill payload 的 kv_cache_used 应是 full cache tensor [nb,bs,132]"
    )
    slot_mapping = score["slot_mapping"].to(device).long()  # [N] flat token 槽号

    kv_flat = kv_cache.to(device).view(-1, CACHE_ENTRY_BYTES)  # [nb*bs,132] uint8
    gathered = kv_flat.index_select(0, slot_mapping).contiguous()  # [N,132]

    k_values = gathered[:, :INDEX_HEAD_DIM].view(torch.float8_e4m3fn)  # [N,128]
    k_scale = gathered[:, INDEX_HEAD_DIM:].view(torch.float32).squeeze(-1)  # [N]
    return k_values.contiguous(), k_scale.contiguous()


def score_prefill(payload: dict, device: torch.device) -> dict:
    inp = payload["inputs"]
    q_quant = inp["q_quant"].to(device)   # [M,64,128] fp8
    weights = inp["weights"].to(device).float()  # [M,64] fp32
    M = q_quant.shape[0]

    k_values, k_scale = reconstruct_candidate_k(payload, device)
    N = k_values.shape[0]

    # 因果 key 范围:query i 看 key [0, i+1)。首个 prefill 逻辑位置 = 行号。
    cu_seqlen_ks = torch.zeros(M, dtype=torch.int32, device=device)
    cu_seqlen_ke = torch.arange(1, M + 1, dtype=torch.int32, device=device)

    logits = fp8_fp4_mqa_logits(
        (q_quant, None),           # FP8 Q:scale 折进 weights
        (k_values, k_scale),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
    )  # [M,N] fp32

    topk = torch.full((M, INDEX_TOPK), -1, dtype=torch.int32, device=device)
    ops.top_k_per_row_prefill(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk,
        M,
        logits.stride(0),
        logits.stride(1),
        INDEX_TOPK,
    )

    ref = payload["output"].cpu()
    ours = topk.cpu()
    rec = recall_vs_native(ours, ref)
    return {
        "recall": rec,
        "M": M,
        "N": N,
        "ours_row_last": [x for x in ours[-1].tolist() if x >= 0][:8],
        "ref_row_last": [x for x in ref[-1].tolist() if x >= 0][:8],
    }


def _layer_of(path: str) -> str:
    base = os.path.basename(path)
    return base.split("_")[1] if base.startswith("FS_L") else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="FS_*prefill*.pt,或 --sweep 时是 run 目录")
    ap.add_argument("--sweep", action="store_true", help="扫目录下所有 prefill 层")
    args = ap.parse_args()

    device = torch.device("cuda")

    if args.sweep:
        files = sorted(glob.glob(os.path.join(args.path, "FS_L*_prefill_s0_rank0.pt")))
        assert files, f"no prefill payloads under {args.path}"
        recalls = []
        for f in files:
            payload = torch.load(f, weights_only=False, map_location="cpu")
            res = score_prefill(payload, device)
            recalls.append(res["recall"])
            print(f"  {_layer_of(f)}: recall={res['recall']:.4f} "
                  f"(M={res['M']}, N={res['N']})")
        mean_r = sum(recalls) / len(recalls)
        n_pass = sum(1 for r in recalls if r >= 0.99)
        print("=" * 60)
        print(f"[exp13] prefill 打分对拍:{len(files)} 层")
        print(f"[exp13] mean recall = {mean_r:.4f}, {n_pass}/{len(files)} 层 >= 0.99")
        print("PASS ✓" if n_pass == len(files) else "PARTIAL / FAIL ✗")
        print("=" * 60)
        return

    payload = torch.load(args.path, weights_only=False, map_location="cpu")
    assert payload["sig"]["bucket"] == "prefill", "用 prefill 抓取文件"
    print(f"[exp13] loaded {args.path}")
    print(f"[exp13] layer={payload['sig']['layer_name']} "
          f"num_rows={payload['sig']['num_rows']}")
    res = score_prefill(payload, device)
    print("=" * 60)
    print(f"query rows M = {res['M']}, candidate K rows N = {res['N']}")
    print(f"ours last-row topk head: {res['ours_row_last']}")
    print(f"ref  last-row topk head: {res['ref_row_last']}")
    print(f"RECALL (prefill, per-row causal) = {res['recall']:.4f}")
    print("=" * 60)
    print("PASS ✓ — prefill 打分路径复现 native topk"
          if res["recall"] >= 0.99 else "FAIL ✗ — recall < 0.99,查布局/因果范围")


if __name__ == "__main__":
    main()
