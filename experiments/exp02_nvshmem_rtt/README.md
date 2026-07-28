# exp02: NVSHMEM 跨机 RTT 基准

延续 exp01 的结论:跨机 NCCL send/recv 单层 ~130μs(其中 ~126μs 是 NCCL 软件栈开销,
RDMA 硬件裸延迟只要 ~4μs)。本实验用 NVSHMEM 绕开 NCCL,看能压到多低。

## 与 exp01 的关系

- exp01(`../exp01_remote_indexer_rtt/`):NCCL(同机/跨机)+ raw peer copy + IPC。
- exp02(本目录):NVSHMEM。方案 A payload 沿用 exp01(上行 8580B / 下行 8192B / 61 层)。
- 两个实验**物理隔离**,各自独立目录,互不依赖代码。

## ⚠️ 第一版是「上界」,不是最终延迟

NVSHMEM 的 `wait_until` / `signal_wait_until` 是 `__device__`(只能在 CUDA kernel 内调),
host 侧没有。所以纯 host + ctypes **做不了 put_signal 点对点接力**,只能用
`putmem_on_stream` + `barrier_all` 做往返。`barrier_all` 是集合同步,其开销会混入 RTT,
**测出的数偏高,是一个上界**。

- 若上界已明显低于 NCCL 130μs → 值得写第二版(device-initiated .cu kernel,
  用 put_signal + wait_until 走 ibgda)去逼近 ~4μs 硬件地板。
- 若上界仍接近 130μs → host 路径没优势,直接上 device 版。

第一版的真正目的:**跑通环境**(UID bootstrap + RoCE 传输 + 对称内存),并拿一个
可与 NCCL 对比的量级。

## 文件

| 文件 | 作用 |
|------|------|
| `nvshmem_ctypes.py` | ctypes 封装 NVSHMEM host API(NVSHMEM 无 Python 绑定)+ UID bootstrap |
| `bench_nvshmem_put.py` | put + barrier 往返 RTT bench |
| `run_nvshmem.sh` | 两机启动器(探空闲卡 / RoCE env / torchrun 传 UID / PGID 隔离) |

## 怎么跑(两台机)

两机各一 rank,机器 A 当 rendezvous master(用 A 的 IB IP,如 6.102.176.49)。
**两边都传 A 的 IP。**

```bash
# 机器 A(node 0)
cd experiments/exp02_nvshmem_rtt
./run_nvshmem.sh 0 6.102.176.49

# 机器 B(node 1)
cd experiments/exp02_nvshmem_rtt
./run_nvshmem.sh 1 6.102.176.49
```

结果写到 A 的 `run_<北京时间>_CST/nvshmem_put.json`。停止卡住的运行:
`kill -TERM -$(cat run_*/pid_node0.txt)`。

## RoCE / NVSHMEM 环境变量(run_nvshmem.sh 已内置,可 env 覆盖)

| 变量 | 默认 | 说明 |
|------|------|------|
| `NVSHMEM_HCA_LIST` | mlx5_0 | 用哪张 RDMA 网卡(避开 DOWN 的 mlx5_7) |
| `NVSHMEM_IB_GID_INDEX` | 3 | GID index 3 = RoCE v2(见 exp01 坐实) |
| `NVSHMEM_REMOTE_TRANSPORT` | ibrc | 传输后端;第二版可试 ibgda |
| `NVSHMEM_SYMMETRIC_SIZE` | 64MiB | 对称堆大小(默认约 1GiB,本 bench 几 KB 够,调小避免白占显存) |

## 状态

- [ ] 第一版(host put + barrier)跨机跑通 —— 待两机都有空闲卡时实测
- [ ] 第二版(device kernel put_signal + wait_until, ibgda)—— 视第一版结果决定
