# exp03 — NCCL GIN 后端探测

## 目的

在决定移植 DeepEP V2 的 NCCL Gin 底子写 RTT bench 之前,**先探清本环境的 GIN 后端类型**:

- `ginType == GDAKI (3)` → GPU 直接敲 mlx5 doorbell,和 IBGDA 同物理机制,us 级可及 → V2 路线**绿灯**
- `ginType == PROXY (2)` → CPU 代理,几十 μs 起 → 不满足 us 级需求
- `ginType == NONE (0)` → GIN 不可用,V2 路线**死路**

## 结果(2026-07-28)

两机跨机跑(A + B 各起 1 rank,`torch.distributed --nnodes=2 --nproc_per_node=1`):

```
NCCL runtime version:      2.30.7
NCCL compile-time header:  2.30.7
world size:                2
[rank 0]   deviceApiSupport = true
[rank 0]   multimemSupport  = false
[rank 0]   hostRmaSupport   = true
[rank 0]   nLsaTeams        = 2
[rank 0]   ginType          = GDAKI (3) - GPU-initiated RDMA (target)
[rank 0]   railedGinType    = GDAKI (3) - GPU-initiated RDMA (target)
--- verdict ---
  GDAKI available -- V2 route GREEN. Start porting.
```

**决定性利好**——两机 CX-7 + 驱动 + PCIe 拓扑都能 GPU 直接控制网卡。

后续 RTT bench 见 [exp04_gin_indexer_rtt/](../exp04_gin_indexer_rtt/README.md)。

## 环境要求

- **NCCL ≥ 2.30.4**——`ncclCommQueryProperties` 这个 API 在 2.28.9 不存在
- 我们方案:`uv pip install "nvidia-nccl-cu13>=2.30.4"` 装到 .venv(装到 2.30.7),运行时 `LD_LIBRARY_PATH` 前置让 torch 加载新 .so;系统 `libnccl.so.2.28.9` 不动

## 文件

```
probe_gin.cc        # 极简 C:ncclCommQueryProperties 打印 props.ginType 等
build.sh            # g++ header-only 编译
bench_probe.py      # torch.distributed 起 NCCL,挖 _comm_ptr 交给 probe .so
run_probe.sh        # 两机启动器(自动挑 GPU + HCA + LD_LIBRARY_PATH 前置)
librix_gin_probe.so # 编译产物
```

## 使用方法

```bash
./build.sh
# A 机:
./run_probe.sh 0 <A_ib1_ip>
# B 机(30 秒内):
./run_probe.sh 1 <A_ib1_ip>
```

10 秒内出结果。**这个探测本身零传输开销**,可作为任何后续 GIN 项目的准入检查。
