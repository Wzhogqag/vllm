"""exp14 — 4 个打分 op 的 compute 微基准(补全延迟预算)。

背景:exp04/09 只测了**传输** RTT(GIN ~4μs/层 @ B≥8)。但每层的真实预算 =
传输 + **op compute**,而 op compute 从没单独测过。"占 decode 2%" 的结论只算了传输
那一半。这里把打分本身的 GPU 耗时补上。

测 4 个 op(host-launch CUDA kernel,数据在 GPU):
    decode:  fp8_fp4_paged_mqa_logits  +  torch.ops._C.persistent_topk
    prefill: fp8_fp4_mqa_logits        +  ops.top_k_per_row_prefill

sweep:
    decode:  B ∈ {1,8,16,64} × context_len ∈ {2048,4096,16384}(对齐 exp09 seq-lens)
    prefill: prompt_len ∈ {2048,4096,16384}

用 CUDA event 计时(GPU 侧,含 launch overhead —— 这正是我们要的,因为生产也是
host 逐层 launch)。合成输入(compute 耗时只由 shape 决定,与值无关,dense kernel)。

用法:
    CUDA_VISIBLE_DEVICES=<free> python op_bench.py [--iters 50] [--out results.json]
"""

from __future__ import annotations

import argparse
import json

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)

H = 64            # index heads
D = 128           # head dim
TOPK = 2048
BLOCK_SIZE = 64
ENTRY_BYTES = 132  # 128 fp8 + 4 fp32 scale
WS_BYTES = 1024 * 1024
N_LAYERS = 61


def _time_op(fn, iters: int, warmup: int) -> float:
    """返回单次调用的 GPU 耗时(ms),CUDA event 计时。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def build_decode(B: int, ctx_len: int, dev: torch.device):
    next_n = 1
    blocks_per_seq = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    nb = blocks_per_seq * B
    cache_3d = torch.randint(0, 255, (nb, BLOCK_SIZE, ENTRY_BYTES),
                             dtype=torch.uint8, device=dev)
    kv_cache = kv_cache_as_quant_view(cache_3d, D, use_fp4_cache=False)
    q_paged = torch.randn(B, next_n, H, D, device=dev).to(torch.float8_e4m3fn)
    weights = torch.randn(B, H, dtype=torch.float32, device=dev)
    ctx_2d = torch.full((B, next_n), ctx_len, dtype=torch.int32, device=dev)
    block_table = torch.arange(nb, dtype=torch.int32, device=dev).view(B, blocks_per_seq)
    sched = get_paged_mqa_logits_metadata(ctx_2d, BLOCK_SIZE, get_num_sms())
    max_model_len = ctx_len
    workspace = torch.empty(WS_BYTES, dtype=torch.uint8, device=dev)
    out_idx = torch.empty(B * next_n, TOPK, dtype=torch.int32, device=dev)
    max_seq_len = ctx_len

    def logits_fn():
        return fp8_fp4_paged_mqa_logits(
            (q_paged, None), kv_cache, weights, ctx_2d, block_table, sched,
            max_model_len, clean_logits=False,
        )

    logits = logits_fn()

    def topk_fn():
        torch.ops._C.persistent_topk(
            logits, ctx_2d, out_idx, workspace, TOPK, max_seq_len
        )

    return logits_fn, topk_fn


def build_prefill(L: int, dev: torch.device):
    """单序列 prefill,query len = key len = L,因果。"""
    q = torch.randn(L, H, D, device=dev).to(torch.float8_e4m3fn)
    k_values = torch.randn(L, D, device=dev).to(torch.float8_e4m3fn)
    k_scale = torch.randn(L, dtype=torch.float32, device=dev)
    weights = torch.randn(L, H, dtype=torch.float32, device=dev)
    cu_ks = torch.zeros(L, dtype=torch.int32, device=dev)
    cu_ke = torch.arange(1, L + 1, dtype=torch.int32, device=dev)
    topk = torch.full((L, TOPK), -1, dtype=torch.int32, device=dev)

    def logits_fn():
        return fp8_fp4_mqa_logits(
            (q, None), (k_values, k_scale), weights, cu_ks, cu_ke,
            clean_logits=False,
        )

    logits = logits_fn()

    def topk_fn():
        ops.top_k_per_row_prefill(
            logits, cu_ks, cu_ke, topk, L,
            logits.stride(0), logits.stride(1), TOPK,
        )

    return logits_fn, topk_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    dev = torch.device("cuda")

    ctx_lens = [2048, 4096, 16384]
    batches = [1, 8, 16, 64]
    results = {"decode": [], "prefill": [], "num_sms": get_num_sms()}

    print("=" * 74)
    print("DECODE 路径:fp8_fp4_paged_mqa_logits + persistent_topk")
    print(f"{'B':>4} {'ctx':>7} | {'logits ms':>10} {'topk ms':>9} "
          f"{'层 μs':>9} {'×61 ms':>8}")
    print("-" * 74)
    for ctx in ctx_lens:
        for B in batches:
            lf, tf = build_decode(B, ctx, dev)
            t_log = _time_op(lf, args.iters, args.warmup)
            t_top = _time_op(tf, args.iters, args.warmup)
            per_layer_us = (t_log + t_top) * 1000
            total_ms = (t_log + t_top) * N_LAYERS
            results["decode"].append({
                "B": B, "ctx_len": ctx, "logits_ms": t_log, "topk_ms": t_top,
                "per_layer_us": per_layer_us, "total_61_ms": total_ms,
            })
            print(f"{B:>4} {ctx:>7} | {t_log:>10.4f} {t_top:>9.4f} "
                  f"{per_layer_us:>9.2f} {total_ms:>8.3f}")

    print("=" * 74)
    print("PREFILL 路径:fp8_fp4_mqa_logits + top_k_per_row_prefill(单序列)")
    print(f"{'L':>7} | {'logits ms':>10} {'topk ms':>9} {'层 μs':>9} {'×61 ms':>8}")
    print("-" * 74)
    for L in ctx_lens:
        lf, tf = build_prefill(L, dev)
        t_log = _time_op(lf, args.iters, args.warmup)
        t_top = _time_op(tf, args.iters, args.warmup)
        per_layer_us = (t_log + t_top) * 1000
        total_ms = (t_log + t_top) * N_LAYERS
        results["prefill"].append({
            "prompt_len": L, "logits_ms": t_log, "topk_ms": t_top,
            "per_layer_us": per_layer_us, "total_61_ms": total_ms,
        })
        print(f"{L:>7} | {t_log:>10.4f} {t_top:>9.4f} "
              f"{per_layer_us:>9.2f} {total_ms:>8.3f}")
    print("=" * 74)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp14] wrote {args.out}")


if __name__ == "__main__":
    main()
