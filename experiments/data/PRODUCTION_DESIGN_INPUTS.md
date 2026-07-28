# Remote Indexer 生产设计输入(Data-backed)

> 本文档是 remote indexer 分离方案的**决策依据清单**。每条都有实验数据支撑,指到具体 JSON 文件的 key path。

**日期**:2026-07-28
**基线硬件假设**:H200 8 卡节点 + CX-7 400G RoCE(生产同款)。规格变化需重新 bench。

---

## 1. 时延预算(每条实测)

### 1.1 稳态 decode 的通信开销

方案 A(投影本地,index_q 送远端 = 8580B/token,top-k 回本地 = 8192B/token)。

| batch size | 单层 RTT | 61 层 |
|---:|---:|---:|
| B=1 单 token | 20.77 μs ❌ | 1.267 ms(占 decode 12%,不推荐) |
| **B=8+** | **~4 μs** ✓ | **~0.24 ms(占 decode ~2%)** |
| B=64 | 5.86 μs | 0.358 ms |
| B=256 | 29.33 μs(带宽饱和) | 1.789 ms |

**数据源**:`exp04_rtt_batch_sweep.json → results[].rtt_avg_us`

**设计推论**:
- **必须强制 batching**:请求进 vLLM 后攒到 B ≥ 8 再发起 indexer 通信
- decode 主循环加个 20μs 的 8-token buffer 逻辑,吞吐几乎无损失

### 1.2 新 request 绑定 indexer 的 TTFT 开销

一次性把 prefill K cache 从主实例灌到 indexer(BULK 模式必选)。

| context 长度 | 灌充耗时 |
|---:|---:|
| 8k tokens(短) | 37.5 μs |
| 32k tokens(中) | 105.0 μs |
| 128k tokens(长) | 372.8 μs |

**数据源**:`exp05_prefill_kcache_fill.json → results[mode=BULK]`

**设计推论**:
- TTFT 增量对所有 context 长度都 <1ms,**用户无感知**
- Prefill 主体计算通常几十 ms 到几秒,通信开销 <1%
- 迁移 request 到另一个 indexer 也用这条 BULK 路径,成本相同

### 1.3 主实例 kernel 里怎么发

- **禁止逐 token put**:STREAMING 模式测出 8k = 25.8 ms(比 BULK 慢 700x)
- **必须一次性大 put**:先在主实例侧攒完整 K cache buffer,一次 gin.put 送

**数据源**:`exp05_prefill_kcache_fill.json → results[mode=STREAMING]`

---

## 2. 资源配置(硬要求)

### 2.1 每 indexer 卡的 QP 池大小

```
num_qps = max_concurrent_main_instances_on_this_indexer
```

**具体建议:8 或 16**(每 QP ~KB 显存 + 1024 depth,资源廉价)。

**数据源**:`exp07_multi_qp.json`
- N=8 主实例并发 + num_qps=8 → 单次 RTT 稳定 3-4μs,不劣化
- N > num_qps → **SQ 死锁**(硬故障,kernel 死自旋)

### 2.2 主实例侧 QP 分配

每个主实例(或者更精确:每个"独立通信流")分配 **1 个 GIN context**。
- 一个 vLLM 实例可能同时是多个 indexer 的客户端 → 每个 (main, indexer) 对独立 comm 独立 QP
- 主实例内多 GPU(TP)本来就有 NCCL comm 组,GIN 是独立于 TP 的一层

### 2.3 对称堆大小

灌充 buffer + 每层 index_q/top-k 循环 buffer。生产建议:

```
symmetric_bytes ≥ max_context_length × 132  +  61 × B_peak × 8580  +  安全 margin
                = 128k × 132  +  61 × 32 × 8580  +  1MB
                ≈ 16.5 MB   +  16.8 MB          +  1MB
                ≈ 35 MB per indexer
```

---

## 3. 部署形态

### 3.1 跨机 indexer 分离(主路径)

**通信**:GIN GDAKI(GPU 直接 ring mlx5 doorbell)
**延迟**:steady 3-4 μs / layer @ B=16
**代码**:见 exp04/rix_rtt_kernel.cu,主 kernel 用 `gin.put + WeakSignalInc + waitSignal`

### 3.2 同机 indexer 分离(常见配置)

**⚠️ 陷阱**:GIN 不做自动 NVLink shortcut!
- exp06 实测:同机 2 卡跑 exp04 kernel,延迟和跨机一样(~4 μs @ B=16),**没走 NVLink**
- **DeepEP V2 是应用层显式路径**(参考 `deep_ep/include/deep_ep/common/handle.cuh:75`)

**生产必须实现双路径**:
```cuda
if (gin.is_nvlink_accessible(peer)) {
    // NVLink shortcut: ncclGetLsaPointer + st.global
    // 预期 <1 μs / layer
} else {
    // 跨机 GIN: gin.put + WeakSignalInc
    // 预期 ~4 μs / layer
}
```

**数据源**:`exp06_intranode_rtt.json`(同机数据同跨机)

### 3.3 Request 迁移(indexer 之间)

**语义**:stop-and-copy —— 主实例暂停 request 一步 decode → 源 indexer BULK 送 K cache 到目标 → 主实例切 GIN comm → 恢复

**开销**:
- 32k context 迁移 = 105 μs
- 128k context 迁移 = 373 μs
- 加上一次控制面 RPC (~几十 μs)
- **总 <1 ms 用户无感**

**数据源**:`exp05_prefill_kcache_fill.json → results[mode=BULK]`

---

## 4. 环境准入检查(生产上线前)

按顺序,任一失败即不满足条件。

### 4.1 NCCL 版本

```bash
python -c "import ctypes; l=ctypes.CDLL('libnccl.so.2'); \
  v=ctypes.c_int(); l.ncclGetVersion(ctypes.byref(v)); \
  print(v.value)"
```
**要求**:≥ 23004(即 2.30.4+)

**为什么**:2.28.9 的 libnccl 没有 `ncclCommQueryProperties` 符号,V2 GIN API 不可用。

### 4.2 GDAKI 后端

跑 exp03 探测:
```bash
cd experiments/exp03_nccl_gin_probe
./run_probe.sh 0 <A_ib1_ip>   # A
./run_probe.sh 1 <A_ib1_ip>   # B
```

**要求**:
```
[rank 0]   ginType          = GDAKI (3)
[rank 0]   railedGinType    = GDAKI (3)
```

**为什么**:
- `PROXY (2)`:CPU 代理路径,延迟几十 μs,**不满足生产**
- `NONE (0)`:NIC/驱动/固件不支持,**路径死路**

### 4.3 网卡 PIX 亲和

每个 GPU 应有一个 PIX 亲和 mlx5 网卡(通过 `nvidia-smi topo -m` 验证)。生产的 `NCCL_IB_HCA` 必须指向亲和的 HCA,否则 GDAKI 走 NODE/SYS 路径延迟劣化。

exp04 `run_bench.sh` 有自动推导逻辑(基于 sysfs PCI 路径最长共享前缀),可直接搬到生产。

---

## 5. 已知限制 / TODO

| 事项 | 状态 | 影响 |
|---|---|---|
| B=1 单 token slow path(20.77μs) | 已定位到 GIN 小消息 fast path fallback,未查到关闭开关 | 强制 batching ≥ 8 规避 |
| CUDA Graph 消除 host launch overhead | 未测 | 潜在 3-4μs → <2μs 优化空间 |
| 同机 NVLink shortcut 手写实现 | 未做(参考 DeepEP V2) | 同机部署时必须补 |
| K cache prefix dedup(和 vLLM prefix cache 联动) | 未做 | 长 prompt 重复场景可省灌充时间 |
| Indexer 挂掉的 fault tolerance | 未设计 | 生产上线前必须补(热备 or 降级) |
| Symmetric heap 与 vLLM KV cache 共存 | 未测 | 需要显式调 `gpu_memory_utilization` 留出空间 |
| Weight loading 分离(indexer 权重 vs 主实例权重) | 未做 | 需要动 vLLM weight loader |

---

## 6. 附录:实验 → 数据文件对照

| 实验 | JSON | 用途 |
|---|---|---|
| exp04 | [exp04_rtt_batch_sweep.json](exp04_rtt_batch_sweep.json) | steady decode RTT × batch |
| exp05 | [exp05_prefill_kcache_fill.json](exp05_prefill_kcache_fill.json) | prefill 灌充 3 mode × 4 seq_len |
| exp06 | [exp06_intranode_rtt.json](exp06_intranode_rtt.json) | 同机 vs 跨机对照 |
| exp07 | [exp07_multi_qp.json](exp07_multi_qp.json) | 多主实例并发 × QP 数 |

想复现或深挖,进各 `exp*/` 目录看 `README.md` + `run_*.sh`。
