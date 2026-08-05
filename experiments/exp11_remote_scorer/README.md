# exp11 — remote indexer scorer 原型(离线对拍)

## 结论(2026-08-04)

**PASS ✓** —— 一个 bare `import vllm` 的独立进程,用**自建 allocator**(物理 block id
故意与主实例不同)+ 真实的 vLLM 打分 op,在真实抓取数据上算出和 vLLM 原生**完全一致**
的 top-2048(recall@2048 = 1.0000,L00/L02/L04 三层验证)。

这验证了 indexer 分离方案的核心声明:**物理 block id 奇偶性无关,逻辑位置对齐即可**。
远端可以自建分配,不必镜像主实例的 block_pool。

## 验证了什么

| 项 | 结论 |
|----|------|
| 两个打分 op 能 bare import vllm 独立调 | ✓ 不 boot 引擎/模型/调度器 |
| 自建 allocator(块号 [0,1,2]→[2,1,0] 倒序)| ✓ topk 不变 |
| 真实数据(非合成)| ✓ 用 exp10 full-score 抓取 |
| 对拍 vLLM 原生 topk | ✓ recall = 1.0000 |

## 怎么跑

1. **抓数据**(需要 8 卡跑 V3.2):在 `../exp10_vllm_shadow/` 用 full-score 模式起短 prompt:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   VLLM_DUMP_FULL_SCORE=1 VLLM_DUMP_DECODE_STEPS=1 \
   VLLM_DUMP_DIR=$(pwd)/run_XXX ../../.venv/bin/python launch_with_hook.py \
     --model /models/DeepSeek-V3.2 --tensor-parallel-size 8 --enforce-eager \
     --max-num-seqs 4 --max-model-len 1024 --port 8000
   # 发一个短请求(~64 token prompt, max_tokens>=4)→ 落 FS_*.pt
   ```
2. **对拍**(单卡即可,不 load 模型):
   ```bash
   CUDA_VISIBLE_DEVICES=1 ../../.venv/bin/python replay_scorer.py \
     ../exp10_vllm_shadow/run_XXX/FS_L00_decode_s0_rank0.pt
   ```

## 数据契约(full-score 抓取的 decode 文件)

`{sig, inputs:{hidden_states,q_quant,k,weights}, output:native_topk, score:{...}}`,其中
`score` 含 paged 打分 op 所需:`kv_cache_used={blocks[nb,64,132]uint8, orig_block_ids}`、
`block_table[B,16]`、`seq_lens[B,1]`、`max_model_len`、`scale_fmt='ue8m0'`。

## 关键实现点(踩过的坑)

- **4D view**:DeepGEMM paged op 要 `[nb,64,1,132]`,抓到的是 3D `[nb,64,132]`。用 vLLM 自己的
  `kv_cache_as_quant_view(kv_cache, 128, use_fp4_cache=False)`(= `unsqueeze(-2)`)转,保证布局一致。
- **context_lens 2D**:paged op 要 `(B, next_n)` 形状,`clean_logits=False`。
- **persistent_topk**:要 1MiB uint8 workspace,k=2048。
- topk 是逻辑 token 位置,与物理块摆放无关 —— 这正是"物理 id 无关"的来源。

## 下一步(Level 0.2)

打分核心 + 自建 allocator 已证明。下一步把 payload 走 GIN 传输(exp04/exp09 的通信层),
做成真正的跨进程/跨机远端 scorer,vLLM 仍用原生结果(影子模式),对拍真实负载下的 recall + 延迟。

## 关联

- 数据来源:`../exp10_vllm_shadow/dump_hook.py`(full-score 模式)
- 设计依据:`../data/INDEXER_SPLIT_SCORING.md`(打分零拆分 + 输入清单 + 自建 allocator 决策)
- 通信层:`../exp09_replay_demo/`(GIN + payload 布局,pytorch 参照)
