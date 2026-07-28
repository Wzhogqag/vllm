# exp04 — 跨机 NCCL GIN GDAKI indexer 单层 RTT bench

## 结论(TL;DR)

DeepSeek V3.2 remote lightning indexer 跨机通信路线**决定性绿灯**——用 NCCL 2.30 新引入的 **GIN GDAKI** 后端(GPU 直接敲 mlx5 doorbell,和 IBGDA 同层但走 NCCL 官方 API),在 CX-7 400G RoCE 上单层单 token(方案 A,上行 8580B / 下行 8192B)RTT **avg 20.77μs / p50 21.08 / p99 21.66**;61 层串行 **1.267 ms**。

**Batch sweep 决定性发现**:B=16 时 61 层串行仅 **0.237 ms**(比 B=1 快 5.4x),对 decode 主体 8-15ms 只占 **1.6-3%**——**远端 indexer 在 batch decode 场景下几乎零成本**。

## 全课题回顾

从"H200 900 GB/s NVLink 能否把 indexer 远端化的往返压下来"到 GIN 拿到最终数据,踩了几个坑:

| 阶段 | 方案 | 单层 RTT | 61 层 | 结论 |
|---|---|---|---|---|
| H800 基线 | 跨机 NCCL send/recv | 70μs | 4.3ms | 太慢(占 decode 20%+) |
| exp01(H200) | 跨机 NCCL send/recv | 130-167μs | ~10ms | 更慢,不可行 |
| exp01 | ib_write_lat 单向(CPU 内存) | 4μs 单向 | 0.25ms | 硬件地板,证明网卡行 |
| exp01 | 同机 NVLink IPC | 8μs | 0.5ms | 同机没问题,但用不上 |
| exp02 | V1 NVSHMEM 手写 kernel | 死 | 死 | rc=800 + IBGDA fail 卡两天 |
| exp02 | V1 IBGDA(拷 DeepEP `ibgda_device.cuh`) | 死 | 死 | ibgda_nic_mem_gpu_map failed,gdrdrv 缺失 |
| exp03 | 探测 NCCL 2.30 GIN 后端 | — | — | 绿灯:`ginType=GDAKI(3)` 且跨机通 |
| **exp04(本目录)** | **NCCL GIN GDAKI put + WeakSignalInc + waitSignal** | **20.77μs (B=1) / 3.88μs (B=16)** | **1.27ms / 0.24ms** | **可行,决定性数据** |

## Batch sweep 完整表(方案 A payload = B × 8580B 上 / B × 8192B 下)

```
   B  up(B)     down(B)   avg(μs) p50    p95    p99    GB/s   61 层(ms)
   1   8580      8192     20.77  21.08  21.40  21.66   0.81      1.267
   4  34320     32768      3.87   3.88   4.00   4.05  17.34      0.236
  16 137280    131072      3.88   3.89   4.04   4.07  69.19      0.237
  64 549120    524288      5.86   3.99  12.01  12.34 183.09      0.358
 256 2196480  2097152     29.33  44.36  46.01  50.11 146.38      1.789
```

三段结构:

- **B=1 反常慢(20.77μs)**:未查明,候选是 GIN 对极小非对齐 message 的固定 fence 常数
- **B ∈ [4, 16] 是 launch-bound 平台(3.88μs)**:GIN put + waitSignal 的 GPU-side 固定 overhead ~3.8μs,payload 4x 时间不变
- **B ∈ [64, 256] 开始 bandwidth-bound**:B=256 双向 146 GB/s ≈ 单向 73 GB/s,逼近 CX-7 400G IB 单向 50 GB/s 硬上限(注意 IB 400G ≈ 50 GB/s 是单向理论)

## 环境要求

- NCCL **≥ 2.30.4**(2.28.9 无 `ncclCommQueryProperties`,V2 硬依赖此 API)
- 我们方案:装 `nvidia-nccl-cu13>=2.30.4` 到 `.venv`,运行时 `LD_LIBRARY_PATH` 前置 `.venv/.../nvidia/nccl/lib` 让 torch dlopen 新 .so(系统 `libnccl.so.2.28.9` 不动)
- 网卡:CX-7(或支持 GDAKI 的 mlx5)+ 内核 `nvidia_peermem` 加载
- `ncclCommQueryProperties` 返回 `props.ginType == NCCL_GIN_TYPE_GDAKI (3)`——**这是硬门槛**,如果返回 `PROXY(2)` 或 `NONE(0)`,延迟会掉 5-10x

## 文件

```
rix_gin_host.cc          # NCCLSymmetricMemoryContext 精简版(ncclMemAlloc + ncclCommWindowRegister + ncclDevCommCreate)
rix_rtt_kernel.cu        # 两个 __launch_bounds__(32, 1) 单 warp kernel;gin.put + WeakSignalInc + waitSignal
build.sh                 # header-only 编译,不 dlink libnccl_device.bc(不需 JIT)
bench_rtt.py             # Python launcher,ctypes 调 3 个 C API,batch sweep 支持
run_bench.sh             # 两机启动器(自动挑 GPU + PIX-affinity HCA + LD_LIBRARY_PATH 前置)
librix_gin_rtt.so        # 编译产物
```

## 使用方法

**两机 rank 0/1,ib1 IP 传 A 机地址**:

```bash
# 编(两机各一次)
./build.sh

# 跑(两机 30 秒内先后启动,第二参数都是 A 的 ib1 IP)
# A 机:
./run_bench.sh 0 <A_ib1_ip>
# B 机:
./run_bench.sh 1 <A_ib1_ip>

# 默认参数(方案 A + 5 个 batch):
#   --up-per-token 8580 --down-per-token 8192
#   --batch-sizes 1,4,16,64,256
#   --iters 1000 --warmup 50 --layers 61 --symmetric-bytes 16M
```

## 关键实现要点(踩过的坑)

- **`ncclDevComm_t` 是值类型 struct**,不是 DeepEP 那种 `jit::NoRefPtr` 包装。2.30.7 结构固定 size,直接 `ncclDevCommCreate(comm, &reqs, &ctx->dev_comm)` 传 struct 地址即可。
- **不需要 `-rdc=true` 也不需要 dlink `libnccl_device.bc`**——那是 DeepEP JIT 走 IR 的路径。所有 GIN device API 都是 `NCCL_DEVICE_INLINE`,header-only 编译即可。
- **`_comm_ptr` 有时是属性有时是方法**(torch 2.11 跨机场景是 method):`v() if callable(v) else v`。
- **DOWN_CAP 参考值**:目前编译期常量 4 MiB(覆盖 B ≤ 256,方案 A);symmetric_bytes 运行时参数 16 MiB。
- **payload 对齐**:8580 非 128 对齐,B=1 反常慢可能与此相关;生产可 pad 到 8704B 试试。

## 下一步(未做)

- 搞清 B=1 vs B=4 的 5.4x 反常突降根因(padding? memory_order? StrongSignal?)
- 若接入 vLLM,需要把 GIN put/wait 嵌到 `vllm/models/deepseek_v32/nvidia/attention.py` 里 indexer 分支,同时对 `fused_qkv_a_proj` 和 `fused_norm_rope` 做拆分(远端化 = 拆融合 kernel 的代价)
