# Prefill vs Decode 两阶段通信方案分析

> 前置读物:[README.md](README.md)(实验数据) + [PRODUCTION_DESIGN_INPUTS.md](PRODUCTION_DESIGN_INPUTS.md)(设计约束)。本文回答**同一 batch 的 token 在 DeepSeek V3.2 的 61 层里怎么和远端 indexer 交互**。

## 一、V3.2 主 forward 的层间约束(源码坐实)

`vllm/model_executor/models/deepseek_v2.py:1466-1493`:主 forward 是**严格串行 for 循环**:

```python
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    ...
    hidden_states, residual = layer(positions, hidden_states, residual, ...)
```

- 上一层输出是下一层输入,**无法层间流水/预取**
- 每层内部:MLA attention + MoE FFN + indexer 分支
- 每层的 indexer 分支需要:
  1. 算本层 K(shape=(N_tokens, 128))
  2. **写入本层 K cache**(`indexer_k_quant_and_cache`,`sparse_attn_indexer.py:398`)
  3. 用本层 index_q 对本层历史 K 打分,得 top-k(用于本层 MLA)
- 只有 top-k 拿回来,本层 attention 才能继续,才能到下一层

**结论:每层都有一次"完整通信 = K cache append + index_q RTT",无法合并跨层**。

## 二、Prefill 阶段的通信

假设 request seq_len=S(prefill 一次算完 S 个 token)。

### K/Q 的 shape 对称性(MQA 语义,先讲清楚)

Indexer 是 MQA(Multi-Query Attention)结构:

```
每 token 的 Q (index_q):  n_heads=64 × head_dim=128 = 8192 dim/token(fp8 后 8192 B)
每 token 的 K (index_k):  1 × head_dim=128 = 128 dim/token(fp8 后 128 B + 4 B scale = 132 B)
每 token 的 weights:      n_heads=64 个标量(bf16 128 B 或 fp32 256 B,视配置)
```

**Q 每 head 独立,K 全 head 共享**(所有 64 个 q head 都拿这一份 K 打分)。源码坐实:
- `wq_b = ReplicatedLinear(1536, 128×64=8192)`(每 head 独立)—— `deepseek_v2.py:669`
- `wk_weights_proj = MergedColumnParallelLinear(hidden_size, [128, 64])`(K 单份 128 dim + weights 64 标量)—— `deepseek_v2.py:678-685`
- `IndexerCache.num_kv_heads=1`,`head_size=132`(128 fp8 + 4 fp32 scale packed)—— `deepseek_v2.py:634`

### 方案 A 上行 8580B/token 已包含 K append

之前 memory 里坐实的方案 A 单 token 上行:
```
index_q_fp8 (8192)  +  index_weights (256)  +  index_k (132)  =  8580 B/token
```

**index_k 132B 就是要 append 到远端 k_cache 的那一份**,已经在 8580B/token 里,**不需要额外一次 K append 传输**。

一层的实际通信 = **一次 up put(合并 Q + weights + K,per-token 8580B)+ 一次 down put(top-k,per-token 8192B)= 一次 RTT**。这也是 exp04 测的东西。

### 每层通信语义

```
主实例:
├── 算 K(S × 132B)+ index_q(S × 8192B)+ weights(S × 256B)
├── 拼到 up buffer(S × 8580B),一次 gin.put + WeakSignalInc
│   ← 远端 kernel 拿到后:先把 K append 到 k_cache[layer_i](offset 由 slot_mapping 决定),
│      再用 index_q 对整个 k_cache[layer_i][:seq_so_far] 打分
├── waitSignal 等 top-k(S × 8192B)传回
└── 用 top-k 做本层 MLA attention,下一层
```

**总量**:每层就是**一次 RTT**,不是两次单独传输。

### Prefill 单层延迟估算(基于 exp04/05 数据)

以 seq_len=8k 为例:
- Up payload:8192 × 8580 = 67.5 MiB
- Down payload:8192 × 8192 = 64 MiB
- 单向估算(BULK 46 GB/s):up ~1.5 ms,down ~1.4 ms
- **单层 RTT ~2.9 ms**(估算,exp04 只测到 B=256 = 2.2 MiB,大 payload RTT 未实测)

其他 seq_len(估):

| seq_len | up (MiB) | down (MiB) | 单层 RTT 估 | 61 层总和 | 相对 prefill 主体 |
|---|---|---|---|---|---|
| 2k | 16.8 | 16.0 | ~0.7 ms | ~44 ms | 中等 |
| 8k | 67.5 | 64.0 | ~2.9 ms | ~178 ms | 显著 |
| 32k | 270.0 | 256.0 | ~11.5 ms | ~700 ms | 已经接近 prefill 主体 |
| 128k | 1080.0 | 1024.0 | ~46 ms | ~2.8 s | 超过典型 prefill 计算,不可接受 |

**⚠️ 未测的关键数字**:exp04 只到 B=256(2.2 MiB up),prefill 一层就是 67.5 MiB up,**大 payload RTT 是外推,不是实测**。这是**最大不确定性**,需要 exp09 补测。

### Prefill 的设计选择

**方案 P1:直接照实现,每层一次 RTT**

按上面的估算,61 层 178 ms @ 8k。**32k 已经吃力,128k 崩**。

**方案 P2:主实例本地算 top-k(即"prefill 保留本地 sparse attention")**

对 prefill 阶段:反正 prefill 计算主体也很重,不做 sparse 直接 dense attention 都行(V3.2 config 允许某些层跳过 indexer)。**prefill 用本地,decode 用远端**——这是个可考虑的**混合部署**方案。

**方案 P3:接受 prefill 慢一些**

Prefill 是 request 生命周期一次性的,慢 100-200ms 用户不敏感(只影响 TTFT)。**长 context 迁移到独立 prefill 池,decode 池才用远端 indexer**——和 disaggregated prefill/decode 天然契合。

**我倾向 P3**:让 remote indexer 只服务 decode,prefill 走本地(或专用 prefill 池)。理由后面讲。

## 三、Decode 阶段的通信

Decode 每 step 生成 1 个 token,batch size = B 表示 B 个并发 request 同步 decode。

### 每层要做什么(每 step 每层)

```
主实例:
├── 算本层 K(shape=(B, 132B))+ index_q(shape=(B, 8192B))+ weights(B, 256B)
├── 拼到 up buffer(B × 8580B),一次 gin.put + WeakSignalInc
│   ← 远端 kernel:先 K append 到 k_cache[layer_i],再用 index_q 打分选 top-k
├── waitSignal 等 top-k(B × 8192B)传回
├── 用 top-k 做本层 MLA attention
└── 下一层
```

**每层一次 RTT,和 exp04 测的完全一样**。

### Decode 单层延迟估算(B=16 典型)

| 传输 | payload | 延迟 |
|---|---|---|
| **一次上行 put(合并 K + Q + weights)** | B × 8580 = 137 KiB | put + waitSignal 单向 ~2 μs |
| **一次下行 put(top-k 回)** | B × 8192 = 131 KiB | ~2 μs |
| **单层 RTT 合计** | | **3.88 μs(exp04 实测)** |

上行 payload 结构:
- K(B × 132)= 2.1 KiB —— **要 append 到远端 k_cache**
- index_q(B × 8192)= 128 KiB
- weights(B × 256)= 4.1 KiB
- **合计 B × 8580 = 137 KiB**

方案 A 的**关键设计**就是把 K append + index_q + weights **一次 put 送到远端**,不拆成两次;远端 kernel 拿到后自己做"K append + 打分"两步。这是 exp04 已经测的语义,不需要修正。

### 层数总和(B=16)

| B | 单层 RTT | 61 层 ms |
|---|---|---|
| 1 | 20.77 μs | 1.27 ms(slow path,禁用) |
| 4 | 3.87 μs | 0.24 ms |
| **16** | **3.88 μs** | **0.24 ms** ← 典型甜蜜区 |
| 64 | 5.86 μs | 0.36 ms |
| 256 | 29.33 μs | 1.79 ms(bandwidth-bound) |

**Decode 场景通信开销 61 层 0.24 ms,占 decode 主体(8-15ms)1.6-3%,几乎零成本。**

### Decode 各 batch size 全景(基于 exp04)

见上表。B ≥ 4 时都在 launch-bound 甜蜜区,B ≥ 64 开始出现带宽挤压,B=256 已 bandwidth-bound。

## 四、Prefill vs Decode 对比

| 维度 | Prefill(seq_len=8k) | Decode(B=16) |
|---|---|---|
| 每层 payload 上行(合并 K + Q + weights) | 67.5 MiB | 137 KiB |
| 每层 payload 下行(top-k) | 64 MiB | 131 KiB |
| 单层延迟制约 | **Bandwidth-bound**(几十 MiB → ms) | **Launch-bound**(几百 KiB → μs) |
| 每层 RTT | ~2.9 ms(估) | **3.88 μs(实测)** |
| 61 层总和 | ~178 ms(估) | **0.24 ms(实测)** |
| 相对主计算占比 | 中等(15-30%) | 小(1.6-3%) |

**性质完全不同**:
- **Prefill 是 bandwidth 之战**:payload 大,通信时间 ~= payload / 网卡带宽。CX-7 400G 单向 50 GB/s 是硬地板
- **Decode 是 launch overhead 之战**:payload 小,通信时间 ~= 3-5μs 固定 launch 常数

## 五、生产部署建议

### 5.1 我的推荐:decode-only 远端 indexer

**架构**:
- **Prefill 池**:主实例内本地做 indexer(不分离),或者用现有 disaggregated prefill 单独一组机器
- **Decode 池**:主实例连远端 indexer,只把 decode 的 K cache 和 index_q 走 GIN

**理由**:
1. Prefill 的 payload 已经在 bandwidth-bound 区,分离**节省的显存 vs 增加的延迟**权衡不划算
2. Decode 显存压力真正大(每 request 长期占用 K cache),分离价值最高
3. 和业界 disaggregated prefill/decode 架构天然兼容
4. Prefill→decode 切换时把 K cache 一次性传给 decode 池的 indexer——用 exp05 BULK 单次拷贝语义,耗时估算 exp05 数据可查

### 5.2 每层通信要合并到一次 up + 一次 down

- 上行 put:包含 K cache append(B×132B)+ index_q(B×8580B)
- 下行 put(远端回):top-k(B×8192B)
- **一次 RTT 完成一层的所有通信**,不要拆成两次 put

主实例侧 kernel 结构:
```cuda
__global__ void indexer_forward_kernel(...) {
    // 每层调用:
    // 1. 本地算 K + index_q,拼到对称堆连续 buffer
    // 2. gin.put(合并 buffer, WeakSignalInc{up[layer_i]})
    // 3. waitSignal(down[layer_i]) 等 top-k
    // 4. top-k 已在本地对称堆的下行区,继续 MLA
}
```

### 5.3 Prefill(如果非要分离)

要么用 **P2 混合方案**(prefill 保留本地 sparse attention,decode 走远端 indexer),要么接受 178ms+/8k 通信开销(其中大 payload RTT 未实测,真实值可能更高)。

## 六、还没测但方案落地前需要补的实验

按重要性排序:

1. **exp09(必测):大 payload RTT** —— exp04 只到 B=256(2.2 MiB),但 prefill 场景 8k 单层 up 就 67.5 MiB。**大 payload RTT 是否真的是 bandwidth-bound?p99 如何?** 直接决定 prefill 方案。

2. **exp11(必测):61 层串行 full trace** —— 主 kernel 里循环 61 次 (put + waitSignal + 计算模拟),测真实 61 层总和。验证 "3.88 μs × 61 = 0.24 ms" 是否成立(可能有跨层累积开销)。

3. **exp08(计划中,仍相关):K cache 迁移分布** —— 100 次 BULK 大 buf 拷,拿 p95/p99。

4. **exp12(next):同机 P2P shortcut** —— 手写 st.global 路径实测,和跨机 GIN 对照。exp06 已证 GIN 不做 shortcut,这个必须补。

## 七、结论

**Decode 分离方案已经数据充足,可以进入实现阶段**。exp04 单层 RTT 3.88 μs @ B=16 就是 "K append + Q + weights + top-k" 完整语义,不需要额外传输。

**Prefill 分离方案缺关键数据**(大 payload RTT 只有外推),而且理论上更倾向"不分离"(bandwidth-bound 场景下分离受益低)。

**下一步**:
- 立即可做:exp09(大 payload RTT)+ exp11(61 层 full trace)双验证
- 中期:vLLM 分支接入,先做 decode-only
- 长期:如果要扩到 prefill,基于 exp09 数据做决策
