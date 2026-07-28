# exp05 — prefill K cache 灌充耗时

## 目的

方案 A remote indexer 的 K cache 存在 indexer 侧。**新 request 首次绑定 indexer 时,主实例要把 prefill 阶段生成的 K cache 全灌进去**——这一步的耗时直接影响 TTFT(首 token 延迟)。

## Payload 精确数据(vllm 源码坐实)

从 `vllm/model_executor/models/deepseek_v2.py:696-701`:

```python
self.k_cache = DeepseekV32IndexerCache(
    head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4,
    #        = 128 + 128//128 * 4 = 128 + 4 = 132
    dtype=torch.uint8,
    ...
)
```

**每 token 132 字节**(128B fp8 payload + 4B fp32 scale,packed 到 uint8),head-broadcast(不 × n_heads=64,所有 query head 共用同一份 K)。

- 8k context:1.03 MiB
- 32k context:4.13 MiB
- 128k context:16.5 MiB

## 三种灌充 mode 对应真实场景

- **BULK**:一次性大 put(整条 K cache 一次送),对应"从别的 indexer 迁移过来"或"预生成后灌"
- **STREAMING**:每 token 一个小 put(132B × seq_len 次),对应"prefill 每 token 生成后立即送"
- **CHUNKED**:每 chunk N token 一个 put(N×132B × seq_len/N 次),中间态,模拟"每层生成一批"

## 关键预测

- **BULK 接近带宽极限**(CX-7 400G IB 单向 ~50 GB/s),16.5 MB → ~330 μs
- **STREAMING 纯 launch-bound**(每 put 3-4μs × seq_len),8k → 32 ms 太慢,**是反面教材**
- **CHUNKED 64 tokens**:8192/64 = 128 次 put × ~5μs = 640 μs 可接受

预测正确的话,方案设计就得**强制 CHUNKED 灌充**,不能 STREAMING。

## 文件

- `rix_gin_host.cc` — 同 exp04
- `rix_fill_kernel.cu` — 三种 mode 分支的单向灌充 kernel
- `bench_fill.py` — 扫 seq_len × mode 组合
- `run_fill.sh` — 两机启动器
- `build.sh` — 编译

## 使用

```bash
./build.sh
# A 机:
./run_fill.sh 0 <A_ib1_ip>
# B 机:
./run_fill.sh 1 <A_ib1_ip>
```

默认参数:
- `--seq-lens 2048,8192,32768,131072`
- `--modes 0,2,1`(BULK → CHUNKED → STREAMING)
- `--chunk-tokens 64`
- `--iters 20`(每组合 20 次取 p50/p95/p99)

STREAMING mode 在 seq_len > 32k 时自动跳过(避免几十秒等待)。
