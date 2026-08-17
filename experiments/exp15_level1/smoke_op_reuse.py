"""exp15 方案A 最小验证:在独立进程(无 vLLM 引擎)里把 torch.ops.vllm.sparse_attn_indexer
整个 op 跑通(decode 路径)。这是重写 resident_scorer 前的决定性实验 —— 证明"复用原 op"这条
路能走通:init_workspace_manager + 手工构造 metadata + override_forward_context + 调 op。

跑通判据:op 不抛异常,返回的 topk_indices_buffer 被填成合理值(前 seq_len 个是有效索引、
落在 [0, seq_len) 内,其余 -1)。因为 context 只有几十 < 2048,decode token 选中全部历史,
topk 应是 {0..seq_len-1}。

用法:  CUDA_VISIBLE_DEVICES=0 python smoke_op_reuse.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/export/home/weizhongqiang.3/vllm")

# 关掉远端分支(否则 op body 会走 _indexer_remote_maybe)
for k in ("VLLM_INDEXER_REMOTE",):
    os.environ.pop(k, None)
# 确保配置文件不触发远端分支
_CFG = "/tmp/vllm_indexer_remote.json"
_BAK = None
if os.path.exists(_CFG):
    _BAK = _CFG + ".smokebak"
    os.rename(_CFG, _BAK)


def main() -> int:
    import vllm.model_executor.layers.sparse_attn_indexer as S  # noqa: 触发 op 注册
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.utils.torch_utils import _encode_layer_name
    from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    dev = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(0)

    # --- op 常量(DeepSeek-V3.2 indexer)---
    N_HEADS = 64
    HEAD_DIM = 128
    HEAD_WIDTH = 132          # 128 fp8 + 4 fp32 scale
    TOPK = 2048
    BS = 64                   # block_size
    QUANT_BS = 128
    SCALE_FMT = "ue8m0"
    MAX_MODEL_LEN = 1024
    prefix = "model.layers.0.self_attn.indexer.k_cache"

    # workspace 必须先 init(topk=2048 → use_persistent_topk 会用它)
    init_workspace_manager(dev)

    # --- 单请求 decode:1 个 query token,历史 seq_len 个 ---
    seq_len = 40              # < BS,落在物理 block 1(逻辑上不重要,这里就用物理 slot)
    B = 1
    next_n = 1

    # cache:(num_blocks, block_size, 132) uint8。给足 block 覆盖物理块号。
    num_blocks = (MAX_MODEL_LEN + BS - 1) // BS + 2
    kv_cache = torch.zeros(num_blocks, BS, HEAD_WIDTH, dtype=torch.uint8, device=dev)

    # 用一个真实 block:物理 block 1(避开 block 0),token 落 slot 64..64+seq_len-1
    phys_block = 1
    slots_hist = torch.arange(
        phys_block * BS, phys_block * BS + seq_len, dtype=torch.int64, device=dev
    )
    # 先把历史 K 灌进 cache(用同一个 op,模拟主实例已写好的 cache)
    from vllm import _custom_ops as ops
    k_hist = torch.randn(seq_len, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    ops.indexer_k_quant_and_cache(k_hist, kv_cache, slots_hist, QUANT_BS, SCALE_FMT)

    # 当前 decode token 的输入
    hidden = torch.randn(B, 7168, dtype=torch.bfloat16, device=dev)  # 不真用,占位
    q_quant = torch.randn(
        B, N_HEADS, HEAD_DIM, device=dev
    ).to(torch.float8_e4m3fn)
    weights = torch.rand(B, N_HEADS, dtype=torch.float32, device=dev)
    # decode 当前 token 的 K(要写进 slot 64+seq_len)
    k_cur = torch.randn(B, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    slot_cur = torch.tensor([phys_block * BS + seq_len], dtype=torch.int64, device=dev)

    total_len = seq_len + 1   # 含当前 token
    # block_table:逻辑块 → 物理块。逻辑块 0 → 物理块 1(整条请求都在物理块 1,够装 65 token > 64?)
    # 65 token 需要 2 个块:物理块 1(0..63)+ 物理块 2(64)。
    n_logical_blocks = (total_len + BS - 1) // BS
    block_table = torch.zeros(B, n_logical_blocks, dtype=torch.int32, device=dev)
    for j in range(n_logical_blocks):
        block_table[0, j] = phys_block + j
    # 把当前 token 也写好(它落逻辑块 1 offset 0 = 物理块 2 slot 128)
    slot_cur = torch.tensor(
        [(phys_block + (seq_len // BS)) * BS + (seq_len % BS)],
        dtype=torch.int64, device=dev,
    )

    # decode metadata(2D seq_lens (B,next_n))
    seq_lens_2d = torch.tensor([[total_len]], dtype=torch.int32, device=dev)
    sched = get_paged_mqa_logits_metadata(seq_lens_2d, BS, get_num_sms())
    decode_meta = DeepSeekV32IndexerDecodeMetadata(
        block_table=block_table,
        seq_lens=seq_lens_2d,
        decode_lens=torch.tensor([1], dtype=torch.int32, device=dev),
        requires_padding=False,
        schedule_metadata=sched,
        global_seq_lens=None,
    )
    meta = DeepseekV32IndexerMetadata(
        seq_lens=seq_lens_2d.view(-1),
        max_seq_len=total_len,
        slot_mapping=slot_cur,          # 当前 token 的物理 slot
        num_decodes=B,
        num_decode_tokens=B,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode_meta,
        prefill=None,
    )

    topk_buf = torch.full((B, TOPK), -1, dtype=torch.int32, device=dev)

    fwd = ForwardContext(
        no_compile_layers={}, attn_metadata={prefix: meta}, slot_mapping={}
    )
    with override_forward_context(fwd):
        torch.ops.vllm.sparse_attn_indexer(
            hidden,
            _encode_layer_name(prefix),
            kv_cache,
            q_quant,
            None,               # q_scale
            k_cur,              # k(当前 token,op 会写进 slot_cur)
            weights,
            QUANT_BS,
            SCALE_FMT,
            TOPK,
            HEAD_DIM,
            MAX_MODEL_LEN,
            MAX_MODEL_LEN * 40,  # total_seq_lens(workspace 上界)
            topk_buf,
            False,              # skip_k_cache_insert
            False,              # use_pcp
            False,              # use_fp4_cache
        )
    torch.cuda.synchronize()

    row = topk_buf[0]
    valid = row[row >= 0]
    got = sorted(int(x) for x in valid.tolist())
    expect = list(range(total_len))     # decode token 见全部 0..total_len-1
    ok = got == expect
    print(f"[smoke] seq_len(hist)={seq_len} total={total_len} "
          f"num_valid={valid.numel()} min={got[0] if got else None} "
          f"max={got[-1] if got else None}")
    print(f"[smoke] topk == full logical set {{0..{total_len-1}}}? {ok}")
    if not ok:
        print(f"[smoke]   got[:20]={got[:20]}")
        print(f"[smoke]   expect={expect}")
    print(f"[smoke] {'PASS — op reuse works in standalone process' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        if _BAK and os.path.exists(_BAK):
            os.rename(_BAK, _CFG)
    sys.exit(rc)
