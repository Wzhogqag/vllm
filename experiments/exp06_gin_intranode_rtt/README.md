# exp06 — 同机 NVLink GIN RTT

## 结论(反直觉,重要)

**GIN 本身不做同机 P2P shortcut**——同机 2 卡跑 exp04 相同的 `gin.put + waitSignal` kernel,延迟和跨机几乎一样(甚至 B=1 时更差)。要利用 NVLink 极速通路,必须**在应用 kernel 里显式判断 P2P 可达并走 st.global 分支**,DeepEP V2 就是这么做的。

## 数据(与 exp04 跨机对照)

```
              exp04 跨机    exp06 同机     差异
B=1           20.77 μs      27.46 μs      同机反而慢 6.7 μs
B=4            3.87          3.74          同机快 0.13
B=16           3.88          3.81          同机快 0.07
B=64           5.86          5.83          几乎相同
B=256         29.33         28.97          几乎相同
```

## 为什么

预期:同机走 NVLink 应该 <1μs。实测:仍走完整 GIN 栈(WQE 拼装 + 内部路由)。

grep DeepEP V2 源码坐实(`deep_ep/include/deep_ep/common/handle.cuh:75`):
- V2 显式定义 `is_nvlink_accessible` 判断
- 若 P2P 可达,用 `ncclGetLsaPointer` 拿远端对称虚拟地址,直接 `st.global` / `ptx::red_add_rel_gpu`(不走 GIN put)
- 只有远端不可达才 fallback 到 `gin.put`

我们的 kernel(和 exp04 一样)无差别用 `gin.put`,所以同机时 GIN 仍然走完整 IB path,没走 NVLink shortcut。

## 对方案设计的影响

**修正之前的判断**:GIN "自动检测走 NVLink" 是错的。

生产方案里:
- **同机 indexer 分离**(1 卡主实例 + 1 卡 indexer 同机):**必须手写 P2P shortcut**,不能靠 GIN。参考 exp01 IPC(~8μs)或直接 `cudaMalloc peer access + st.global`(理论 <1μs)。
- **跨机 indexer 分离**:直接 GIN GDAKI(3.88μs @ B=16,exp04 已验证)。
- **可移动 indexer**:需要在主实例 kernel 里保留"同机 P2P 快路径 + 跨机 GIN 慢路径"两条分支,DeepEP V2 就是这么做的。

## 复用 exp04 代码

`rix_gin_host.cc` / `rix_rtt_kernel.cu` / `bench_rtt.py` / `build.sh` 完全从 exp04 复制。只新增 `run_intranode.sh`(单机 2 卡,`--nnodes=1 --nproc_per_node=2`,自动挑同机 2 张空闲卡)。

## 使用

```bash
./build.sh
./run_intranode.sh   # 单机自动挑 2 卡
```
