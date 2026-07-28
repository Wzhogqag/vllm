# exp07 — 多 QP 并发扫描

## 目的

一个 indexer 卡能同时服务多少个主实例的 index_q 请求?QP 池要开多大?

## 核心结论(直接写方案)

**QP 数 ≥ 主实例并发数(N ≤ num_qps 是硬约束)**——每 CTA 独占 QP 时,B=16 生产场景 8 主实例并发单次 RTT 仍稳定 3-4μs,不劣化。**建议 indexer 侧 num_qps=8 或 16**。

## 数据(B=16 健康区,方案 A payload)

```
 N   QP    avg(us)   spread  含义
 1    1      3.15     0.00   exp04 baseline
 1    2      2.86     0.00
 1    4      2.89     0.00
 1    8      3.70     0.00
 2    2      3.13     0.25   2 主实例独立 QP,无劣化 ✓
 4    4     ~3.4      1.0    4 主实例并发,稍抖 ✓
 8    8     ~3.5      0.9    8 主实例并发,仍 ~3.5μs ✓
```

## 数据(B=1 反常区,不推荐生产用)

```
 N   QP    avg(us)   min    max    spread
 1    1      21.29   21.29  21.29    0
 2    2      11.82    2.88  20.75   17.87  ← 极不均衡!
 4    4       6.50    2.75  20.43   17.68
 8    8       6.76    2.80  21.18   18.37
```

**B=1 payload 8580B 走了 GIN 内部一条"小消息 fast path fallback"**,导致每次 iter 只有一个 CTA 能命中快路径,其他 CTA 走慢路径,spread ≈ 18μs 是这两条路径的差距。exp04 B=1 反常慢就是它。

**生产影响**:decode batch ≥ 8,payload ≥ 68KB,已跨过小消息陷阱,不会遇到。

## SQ 死锁(必须避免)

```
N > num_qps  →  多 CTA 挤同一个 QP 的 send queue,GIN 内部死锁,永远等 up signal
```

设计上必须保证 `每主实例分配 1 个 QP`,`indexer 侧 num_qps ≥ 并发峰值`。

## 参数配置建议

- **主实例侧**:每 request 用 1 个 QP;多主实例并发时用不同 context_id
- **indexer 侧**:`num_qps = max_concurrent_main_instances`(通常 8-16 足够;QP 是廉价资源,几 KB 显存/个)
- **batch**:强制 B ≥ 8 才发 GIN put(小 batch 攒到 8 再发,避免 B=1 慢路径)

## 文件

- `rix_gin_host.cc`(复用 exp04)
- `rix_multi_qp_kernel.cu`:N block × 1 warp,每 block 独占 QP + 独占 signal 对(sig_up=2*bid, sig_down=2*bid+1);带 20M cycle(约 10ms)超时保护
- `bench_multi_qp.py`:扫 (B, N, num_qps),跳过 N > num_qps 的死锁组合
- `run_multi_qp.sh`:两机启动器
