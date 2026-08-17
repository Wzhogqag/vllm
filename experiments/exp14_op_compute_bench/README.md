# exp14 — 打分 op compute 微基准

## 结论(2026-08-05)

补上此前延迟预算里缺失的一半:**op compute**。exp04/09 只测了传输(GIN ~4μs/层),
从没测打分本身的 GPU 耗时。结果 —— **decode 打分 compute ≈ 23μs/层,比传输(4μs)还大**。

## 数据(H200 SM90,50 iters,CUDA event 计时,合成输入)

### decode 路径:`fp8_fp4_paged_mqa_logits` + `persistent_topk`

| B | ctx | logits μs | topk μs | 层合计 μs | ×61 层 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 2048 | 13.6 | 9.8 | 23.4 | 1.43 |
| 16 | 2048 | 13.1 | 9.9 | 23.1 | 1.41 |
| 64 | 2048 | 13.4 | 6.6 | 20.1 | 1.22 |
| 16 | 4096 | 12.9 | 14.4 | 27.3 | 1.67 |
| 16 | 16384 | 12.9 | 15.7 | 28.6 | 1.75 |
| 64 | 16384 | 37.9 | 12.4 | 50.3 | 3.07 |

decode logits 在这些规模基本 **launch-bound**(B、ctx 变化几乎不动 ~13μs),
只有 B=64×ctx=16384 才真正吃到 compute(38μs)。

### prefill 路径:`fp8_fp4_mqa_logits` + `top_k_per_row_prefill`(单序列因果)

| prompt_len | logits μs | topk μs | 层合计 μs | ×61 层 ms |
|---:|---:|---:|---:|---:|
| 2048 | 37.4 | 7.4 | 44.8 | 2.74 |
| 4096 | 133.3 | 49.9 | 183.2 | 11.17 |
| 16384 | 2045.2 | 735.9 | 2781.1 | **169.6** |

prefill logits **O(L²)**(M×N 因果三角):L 翻 4 倍,耗时 ~16 倍。长 prompt 的 prefill
打分是重 compute。

## 怎么解读(关键,别误算)

1. **compute 不是分离的"额外开销"**:打分 compute 在**不分离时也要花**(主实例本地
   inline 跑同样的 op)。分离到同款 H200 远端,compute 是一笔"平账";分离**净增**的
   只有传输 RTT(~4μs/层)。所以"分离 overhead 占 decode ~2%"的结论**仍成立**
   —— 那 2% 指的是净增的传输。

2. **但 compute 绝对值决定 pooling 天花板(exp14 的新发现)**:单次 decode step 的远端
   打分 compute ≈ 1.4ms(61 层 @ B=16 ctx=2048)。decode step ~10ms → 一个主实例让远端
   忙 ~14%。**即一块 indexer GPU 靠 compute 约 7 个主实例就饱和**。exp07 的
   "8 主实例 : 1 indexer"是**纯传输**结论;compute 给出的天花板(~7)与之接近 ——
   两者一致,pooling 上限 ~7-8:1 是 compute 和传输**共同**限定的,不是只看传输。

3. **prefill 长 context compute 很重**(16384 → 170ms/61层),但 prefill 本身就是几百 ms
   级、且不在 decode 热路径,占比仍小;真正要注意的是它会**长时间独占远端 GPU**,
   与并发 decode 抢资源 —— pooling 调度需要把 prefill 打分和 decode 打分分开考虑。

## 怎么跑

```bash
CUDA_VISIBLE_DEVICES=<free> ../../.venv/bin/python op_bench.py --iters 50 --out results.json
```

## 关联

- 传输侧预算:`../data/README.md`(exp04/09,GIN RTT)—— exp14 补 compute 侧
- op 出处 / 零拆分:`../data/INDEXER_SPLIT_SCORING.md`
- 打分正确性:`../exp11_remote_scorer/`(decode)、`../exp13_prefill_scorer/`(prefill)
