# exp13 — prefill 路径离线打分器(对拍 native topk)

## 结论(2026-08-05)

**PASS ✓(机制层面)** —— prefill 打分路径的**另一对 op**(`fp8_fp4_mqa_logits` +
`top_k_per_row_prefill`)在 bare `import vllm` 进程里离线复现 vLLM 原生 topk,
**61/61 层 recall = 1.0000**。

这补上了 exp11 只验了 decode 路径的缺口 —— 打分面的另一半也成立。

## 验证了什么 / 没验证什么(重要,别过度解读)

| 项 | 状态 |
|----|------|
| prefill 的两个 op 能 bare import 独立调 | ✓ |
| 用 slot_mapping 从 full cache gather 候选 K(无需 block_table)| ✓ |
| 132B/token 拆分 = 128 fp8 值 + 4B fp32 scale | ✓ 布局对 |
| 因果 key 范围 cu_seqlen_ks=0 / ke=i+1 | ✓ 与 native 有效集合逐行一致 |
| **top-k 判别性选择(候选 > 2048 时谁进谁出)** | ✗ **未压到** |

**未验证的原因(与 exp10「short-prefill = no signal」同因)**:抓取用的是 64-token
短 prompt,每个 query 行的因果候选数 N ≤ 64 < topk=2048,所以 top-k 直接返回**全部**
候选 —— recall=1.0 是"全选全中",证明了**管路正确**(op 签名 / K 重建 / 因果掩码 /
fp8 布局),但**没有压到候选饱和(N>2048)时的判别性排序**。

要真正压 prefill 的 top-k 选择,需要一份 **prompt 长度 > 2048** 的 prefill 抓取
(exp10 记忆:>2048 token 时 topk 才真正筛选,topk_max≈2159)。当前 fullscore run 是
64-token,不满足。**这是 exp13 的已知 TODO**,需要一次新的长 prompt full-score 抓取
(要 8 卡跑 V3.2,与 decode 抓取同法,只是把 prompt 加长)。

## 怎么跑

```bash
# 单层
CUDA_VISIBLE_DEVICES=<free> ../../.venv/bin/python prefill_scorer.py \
  ../exp10_vllm_shadow/run_20260804-185915_CST_fullscore/FS_L00_prefill_s0_rank0.pt

# 扫全部 61 层
CUDA_VISIBLE_DEVICES=<free> ../../.venv/bin/python prefill_scorer.py --sweep \
  ../exp10_vllm_shadow/run_20260804-185915_CST_fullscore
```

## 关键实现点(与 decode 路径的差异)

- **非 paged**:prefill 用 `fp8_fp4_mqa_logits`(varlen),不是 decode 的
  `fp8_fp4_paged_mqa_logits`;不需要 paged 调度 metadata、不需要 block_table。
- **K 从哪来**:vLLM 里是 `cp_gather_indexer_k_quant_cache` 把 paged cache gather 进连续
  workspace。离线复现用抓到的 `slot_mapping`(每 token 写 cache 的物理槽号)直接
  `kv_flat.index_select(0, slot_mapping)`。首个 prefill 无历史,逻辑位置 = 行号。
- **因果范围**:`cu_seqlen_ks[i]=0, cu_seqlen_ke[i]=i+1`(query i 看 key [0,i])。
- **topk buffer 预填 -1**:`top_k_per_row_prefill` 只 scatter 有效索引,其余保持 -1 sentinel。

## 关联

- 数据来源:`../exp10_vllm_shadow/`(full-score 抓取,`bucket=prefill`)
- decode 路径对照:`../exp11_remote_scorer/`(paged op,recall 1.0)
- 打分零拆分依据:`../data/INDEXER_SPLIT_SCORING.md`
