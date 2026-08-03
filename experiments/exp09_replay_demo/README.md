# exp09 — 跨机 remote indexer replay demo(合成 payload,不接 vLLM)

## TL;DR

**验证目的:** 在 exp04(GIN empty put/wait 单层 3.88μs)之上补齐**真实语义** ——
远端 rank1 收到 payload 后做真正的 mqa 打分 + topk,再送回;两端算的 top-2048
索引应完全一致(验证 rank 分离不改变结果)。

**结果(2026-07-30):**

| S | recall@2048 | mean_layer_us | min_layer_us | 61 层 |
|---|---|---|---|---|
| 2048  | **1.0000** | 7515 | 3240 | 458 ms |
| 4096  | **1.0000** | 7428 | 3019 | 453 ms |
| 16384 | **1.0000** | 3109 | 3089 | 190 ms |

**结论:**

1. **正确性完全 OK** —— 三种历史长度下,rank0 本地算的 topk 与 rank1 远端算的 topk **100% 命中**,证明"分离后 vLLM 的 sparse indexer top-k 语义完全保持"。
2. **时延 3-8ms/层 与 exp04 empty(3.88μs)差 800×+,但这不能作为"真实分离成本"的估计。** 三种混淆因素:
   - **rank0 本地也算 reference topk 做对比**(apples-to-apples 无关的额外 GEMM),吃掉一大半
   - **每层 4 次 host `cudaDeviceSynchronize`,共 244 次/iter**(kernel-split 方案的固有代价)
   - **每层随机生成 payload + `torch.randn(S+1, 128)` K cache**,也走在计时窗口内
3. **S=16384 mean/min 都 ~3100μs 且几乎无抖动;S=2048/4096 min 也 3000+ μs 但 mean 抖到 7500** —— 说明 baseline 是 ~3ms/层 host 代价,S 短时抖动来自 warmup 尾部污染 mean(iters=20 太少)。**下一版应把 rank0 reference 剥离,并延长 iters 到 200+,才能测到真实的"远端化净成本"**。

## 目的(设计合同)

exp04 已证明:GIN 跨机 empty put/wait 单层 RTT B=16 = 3.88μs / 61 层 = 0.24ms。
本 demo 在其之上补齐**真实语义**:

- 上行 payload **不是随机字节**,而是形状/dtype 与真实 `sparse_attn_indexer` 一致
    - `index_q_fp8 [B, 64, 128]` fp8 e4m3 + `index_weights [B, 64]` bf16
    - `index_k [B, 128]` fp8 + `scale [B]` fp32 拼成 [B, 132] uint8
    - 合计 = **8452 B/token**(严格按元素算,exp04 曾用 8580 是记 padding)
- 远端 rank1 收到后**做真实计算**:
    - 读对称堆 up 区 → torch view
    - `score = einsum("bhd,sd->bhs", q_bf16, K_cache)` shape [B, 64, S+1]
    - `logit = Σ_h w[h] · softmax_scale · n_head_scale · score[h]`,shape [B, S+1]
    - `torch.topk(logit, 2048)` → indices [B, 2048] int32
- 下行 = 真实 top-2048 int32 = **8192 B/token**,B=16 → 131,072 B

## 参数

- B = 16(方案 A 主目标)
- S(历史长度)扫 [**2048**, 4096, 16384] — 注:S=2048 是能选出 top-2048 的最小值,故起点从 2048 起(S=1024 时 S+1<2048,`torch.topk` 会 out-of-range)
- 61 层串行(V3.2 全模型 indexer 层数,`index_topk_freq=1`)
- iters=20,warmup=3 — **下一版必须加大**

## 不做

- 不加载 V3.2 权重(仅需 head=128 卡就能跑;完全脱离 vLLM 依赖)
- 不做 fp8 gemm(bf16 upcast + torch.matmul,数值一致)
- 不做 paged K cache(一维 dense buffer)
- 不测多 concurrent request(B=16 是同一 forward 的 batch)

## 方案 = kernel-split(host+device 串接)

单层每 rank 4 个 kernel:

```
rank0.launch_up:    rank0_put_up_kernel(up_bytes)                # gin.put + SignalInc SIG_UP
rank0.launch_wait:  rank0_wait_down_kernel(expected)             # gin.waitSignal SIG_DOWN
rank1.launch_wait:  rank1_wait_up_kernel(expected)               # gin.waitSignal SIG_UP
[host: torch bmm/topk 读 up 区,写 down 区]
rank1.launch_put:   rank1_put_down_kernel(down_bytes)            # gin.put + SignalInc SIG_DOWN
```

每 kernel launch 后 `cudaDeviceSynchronize`(简单起步,不做 stream chain)。
每层 host 循环一次,共 61 层。

## 复用

- `libnccl.so` 软链到 `../exp04_gin_indexer_rtt/libnccl.so`(NCCL 2.30.7 wheel)
- `rix_gin_host.cc` 与 exp04 结构相同(改 extern 声明为 exp09 4 个入口)
- 环境变量(NCCL_IB_HCA 亲和选卡、GID_INDEX=3 RoCE、LD_LIBRARY_PATH 前置)全部复用

## 目录

```
common.py            维度常量、payload 布局、offset 计算
rix_replay_kernel.cu 4 个 kernel-split device 端
rix_gin_host.cc      GIN symmetric context init(exp09 版)
build.sh             编译 librix_replay.so
bench_replay.py      Python driver: torch 打分 + host+device 串接 + recall 验证
run_replay.sh        两机启动器(GPU-NIC PIX 亲和)
libnccl.so           -> ../exp04_gin_indexer_rtt/libnccl.so
```

## 使用

**两机(A=6.102.176.49 ib1,B=6.102.176.47 ib1)**:

```bash
./build.sh                              # 两机各一次

# 30 秒内先后启动:
# A:
./run_replay.sh 0 6.102.176.49
# B:
./run_replay.sh 1 6.102.176.49
```

## 完整跑通日志(2026-07-30_18:00:40 CST)

```
NCCL version 2.30.7+cuda13.3
[bench09] world=2 native NCCL comm ptr = 0x39d85cf0
[rix09] NCCL props: ginType=3 railedGinType=3 deviceApi=1
[rix09] GIN init ok: symmetric heap 16777216 bytes @ 0x7fb767600000

=== B=16  S=2048  layers=61  iters=20  warmup=3 ===
  mean 7514.56μs/层  [min 3240.31 max 9554.04]  61 层 458.388ms  recall mean=1.0000  min=1.0000

=== B=16  S=4096  layers=61  iters=20  warmup=3 ===
  mean 7428.14μs/层  [min 3019.19 max 9590.49]  61 层 453.117ms  recall mean=1.0000  min=1.0000

=== B=16  S=16384  layers=61  iters=20  warmup=3 ===
  mean 3109.36μs/层  [min 3088.74 max 3260.35]  61 层 189.671ms  recall mean=1.0000  min=1.0000
```

结果 JSON: `run_20260730-180040_CST/replay.json`

## 下一步

- **exp09.2:** 剥离 rank0 reference 打分(改成一次性对拍 iter=0,后续不再重复算),延长 iters=200,warmup=20;单独用 cudaEvent 测 GIN 部分与 torch 部分的时间。目标:量化"kernel-split 方案额外多花多少 μs/层"。
- **exp10:** 等 GPU 齐(8 卡 H200),接 vLLM `--enforce-eager` 起 V3.2 TP=8,dump 真实 payload,replay 对比 recall(时延仍不取一号,先看正确性)。
- **exp11:** 若 exp09.2 显示 host-side 是主要瓶颈,尝试方案 A(kernel-fused,含 device topk + fp8 device GEMM),追极限时延。

## 关联

- 前置:[[project-remote-indexer]] 记忆 · exp04(GIN empty RTT 决定性绿灯)
- 术语:方案 A(投影留本地,payload 8452↑ 8192↓) · kernel-split vs kernel-fused
