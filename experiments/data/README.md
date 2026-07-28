# Remote Indexer 实验数据汇总

**目的**:DeepSeek V3.2 remote lightning indexer 分离方案的实验数据一手来源。所有关键结果集中在本目录,写方案文档/写代码时从这里查。

**日期**:2026-07-28
**硬件**:H200 单机 8 卡 × 2(A=I14-90,B=I14-92),CX-7 400G RoCE
**软件栈**:torch 2.11.0+cu130 / NCCL 2.30.7 / NCCL GIN GDAKI backend

---

## 一图表看全套结果

| 场景 | 单层 μs | 61 层 ms | 备注 |
|---|---:|---:|---|
| **参考:H800 老基线** | 70 | 4.3 | 换机前 |
| **参考:跨机 NCCL send/recv** | 130-167 | ~10 | 完全不可用 |
| **参考:同机 NVLink IPC (cudaMemcpyPeer)** | 8 | 0.5 | exp01 |
| **参考:裸 ib_write_lat 单向(CPU 内存)** | 4 (单向) | 0.25 | 硬件地板 |
| GIN GDAKI 跨机 RTT B=1 | 20.77 | 1.267 | exp04,小消息 slow path |
| **GIN GDAKI 跨机 RTT B=4** | **3.87** | **0.236** | exp04,launch-bound 甜蜜区 |
| GIN GDAKI 跨机 RTT B=16 | 3.88 | 0.237 | exp04 |
| GIN GDAKI 跨机 RTT B=64 | 5.86 | 0.358 | exp04,尾抖开始 |
| GIN GDAKI 跨机 RTT B=256 | 29.33 | 1.789 | exp04,已 bandwidth-bound |
| GIN GDAKI 同机 2 卡 RTT B=16 | 3.81 | 0.232 | exp06,和跨机几乎一样 |

**跨机 decode 主体开销占比**(以 8-15ms/decode step 计):

- B=16:0.237 ms / 10 ms ≈ **2.4%** ← 目标场景,零负担
- B=1:1.267 ms / 10 ms ≈ 12.7% ← 需要 batching 规避

---

## Prefill K cache 灌充(exp05)

payload = seq_len × 132 B/token(vllm 源码坐实,`DeepseekV32IndexerCache`)

| seq_len | 净数据 | BULK p50 | CHUNKED-64 p50 | STREAMING p50 | BULK 有效带宽 |
|---:|---:|---:|---:|---:|---:|
| 2k | 264 KiB | **20.3 μs** | 75.2 μs | 6.17 ms | 13.3 GB/s |
| 8k | 1.03 MiB | **37.5 μs** | 356.6 μs | 25.77 ms | 28.8 GB/s |
| 32k | 4.13 MiB | **105.0 μs** | 1389.5 μs | 104.57 ms | 41.2 GB/s |
| 128k | 16.5 MiB | **372.8 μs** | 5903.9 μs | (skip 太慢) | 46.4 GB/s |

**BULK 打到 46.4 GB/s = CX-7 400G 单向 92% 硬件线速**。**逐 token STREAMING 是生产禁止模式**(8k 灌充 25.8 ms,快赶上一次 prefill 主体)。

---

## 多主实例并发同 indexer(exp07)

前提:每主实例 1 QP,indexer 侧 num_qps ≥ 并发峰值(N ≤ num_qps 是硬约束)。

**健康区 B=16**(方案 A payload = 16 × 8580B):

| N 并发主实例 | num_qps | 单次 RTT μs | spread | 状态 |
|---:|---:|---:|---:|---|
| 1 | 1 | 3.15 | 0.00 | exp04 baseline |
| 2 | 2 | 3.13 | 0.25 | 无劣化 ✓ |
| 4 | 4 | ~3.4 | 1.0 | 稍抖 ✓ |
| 8 | 8 | ~3.5 | 0.9 | 稍抖 ✓ |

**结论**:**8 个主实例并发共用一个 indexer,单次 RTT 仍 3-4μs**。

**B=1 反常**:spread ~18μs(每次 iter 只有一个 CTA 走 GIN 快路径,其他走慢路径)。这解释了 exp04 B=1 反常慢 20.77μs 的根因。生产 decode batch≥8 时不会遇到。

---

## 关键 launch-bound / bandwidth-bound 拐点

- **Launch overhead(GIN GDAKI put+wait)**:~3.8 μs / 单次 put+wait(RTT 一半 ~2μs)
- **单向 put 净启动**:~2-3 μs(exp05 CHUNKED 单 put 约 2.7μs,比 RTT 一半还低——因为没有 waitSignal fence)
- **Bandwidth-bound 起点**:payload ≥ ~500 KB(64 tokens × 8580B),此时时间 ~= payload/50GBs
- **完全带宽饱和**:payload ≥ 16 MB(BULK 128k),94% 硬件线速

---

## 生产设计的硬约束

| 约束 | 数据支持 | 出处 |
|---|---|---|
| **NCCL ≥ 2.30.4** 且必须 GDAKI 后端 | props.ginType=3 (GDAKI) | exp03 |
| **每主实例 1 QP,indexer num_qps ≥ 峰值并发** | N > num_qps SQ 死锁 | exp07 |
| **batch B ≥ 4-8** 触发 GIN put | B=1 慢路径 spread 18μs | exp04+07 |
| **灌充必须 BULK,不能逐 token** | STREAMING 700x 慢于 BULK | exp05 |
| **同机部署要 kernel 内手写 P2P shortcut** | GIN 不做 auto shortcut,同机也走 IB 栈 | exp06 |

---

## 各实验 JSON 索引

- [exp04_rtt_batch_sweep.json](exp04_rtt_batch_sweep.json) — 跨机稳态 decode RTT × batch sweep
- [exp05_prefill_kcache_fill.json](exp05_prefill_kcache_fill.json) — Prefill 灌充 3 mode × 4 seq_len
- [exp06_intranode_rtt.json](exp06_intranode_rtt.json) — 同机 2 卡 GIN RTT(对照)
- [exp07_multi_qp.json](exp07_multi_qp.json) — 多主实例并发 × QP 数扫描

## 各实验源码/README 索引

- [exp01_remote_indexer_rtt/](../exp01_remote_indexer_rtt/) — NCCL / IPC / raw peer 基线
- [exp02_nvshmem_rtt/](../exp02_nvshmem_rtt/) — V1 NVSHMEM/IBGDA 死路存档
- [exp03_nccl_gin_probe/README.md](../exp03_nccl_gin_probe/README.md) — GIN 后端探测
- [exp04_gin_indexer_rtt/README.md](../exp04_gin_indexer_rtt/README.md) — 单层 RTT + batch sweep(主结果)
- [exp05_prefill_kcache_fill/README.md](../exp05_prefill_kcache_fill/README.md) — 灌充 3 mode
- [exp06_gin_intranode_rtt/README.md](../exp06_gin_intranode_rtt/README.md) — 同机对照
- [exp07_multi_qp_concurrency/README.md](../exp07_multi_qp_concurrency/README.md) — 多 QP 并发
