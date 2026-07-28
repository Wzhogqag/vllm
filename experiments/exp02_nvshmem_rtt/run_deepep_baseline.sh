#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# 目的:在本环境上跨机跑 DeepEP 官方 tests/test_low_latency.py,拿一个
#       ground-truth 结论 —— "本机器 IBGDA 跨机通不通"。
#
# 用法(两机各起一份):
#   A 机:  ./run_deepep_baseline.sh 0 <A_IB_IP>
#   B 机:  ./run_deepep_baseline.sh 1 <A_IB_IP>
#
# 两机第二个参数都传 A 机 ib1 的 IP(A=6.102.176.49)。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

NODE_RANK="${1:?usage: $0 NODE_RANK A_IB_IP  (NODE_RANK is 0 or 1)}"
MASTER_IB_IP="${2:?need machine A IB IP as arg 2, same on both nodes}"
MASTER_PORT="${MASTER_PORT:-29800}"          # 与 exp02 其他脚本错开
PY=../../.venv/bin/python
# test_low_latency.py + utils.py 已复制到本目录(exp02 自包含,B 机不用访问 guojinrong6)
DEEPEP_TESTS=.

# -- 每机各占 1 张空闲卡(--num-processes 1)。DeepEP init_dist 用
#    WORLD_SIZE=nnodes, RANK=node_rank(见 tests/utils.py:14-36)。
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 2000 {print $1; exit}')
if [[ -z "${FREE_GPU:-}" ]]; then
  echo "ERROR: no free GPU on this node" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="$FREE_GPU"

# -- 与 run_comm.sh 相同的 GPU-NIC PIX 亲和推导(longest shared PCI prefix)。
if [[ -z "${NVSHMEM_HCA_LIST:-}" ]]; then
  gpu_bus=$(nvidia-smi -i "$FREE_GPU" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null \
    | tr 'A-F' 'a-f' | sed -E 's/^0+([0-9a-f]{4}:)/\1/')
  gpu_path=$(readlink -f "/sys/bus/pci/devices/$gpu_bus" 2>/dev/null)
  best_hca=""; best_score=0
  for dev in /sys/class/infiniband/*; do
    hca=$(basename "$dev")
    [[ "$hca" == *bond* ]] && continue
    st=$(cat "$dev"/ports/1/state 2>/dev/null); [[ "$st" == *ACTIVE* ]] || continue
    dpath=$(readlink -f "$dev/device" 2>/dev/null)
    score=$(awk -v a="$gpu_path" -v b="$dpath" 'BEGIN{
      na=split(a,A,"/"); nb=split(b,B,"/"); s=0;
      for(i=1;i<=na&&i<=nb;i++){ if(A[i]==B[i]) s++; else break }
      print s }')
    if (( score > best_score )); then best_score=$score; best_hca="$hca"; fi
  done
  if [[ -n "$best_hca" ]]; then
    export NVSHMEM_HCA_LIST="${best_hca}:1"
  else
    export NVSHMEM_HCA_LIST="mlx5_0:1"
    echo "WARN: no affinitized HCA for GPU$FREE_GPU, fallback mlx5_0" >&2
  fi
fi
echo "selected GPU=$FREE_GPU  HCA=$NVSHMEM_HCA_LIST"

# -- NVSHMEM & DeepEP env(和 run_comm.sh 一致,让 DeepEP 走它熟悉的 IBGDA 路径)
NVSHMEM_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_IB_ENABLE_IBGDA="${NVSHMEM_IB_ENABLE_IBGDA:-1}"
export NVSHMEM_IBGDA_NIC_HANDLER="${NVSHMEM_IBGDA_NIC_HANDLER:-gpu}"  # DeepEP 默认让 auto/gpu
export NVSHMEM_IBGDA_NUM_RC_PER_PE="${NVSHMEM_IBGDA_NUM_RC_PER_PE:-1}"
export NVSHMEM_QP_DEPTH="${NVSHMEM_QP_DEPTH:-1024}"
export NVSHMEM_MAX_TEAMS="${NVSHMEM_MAX_TEAMS:-7}"
export NVSHMEM_DISABLE_NVLS="${NVSHMEM_DISABLE_NVLS:-1}"
export NVSHMEM_CUMEM_GRANULARITY="${NVSHMEM_CUMEM_GRANULARITY:-536870912}"
export NVSHMEM_IB_GID_INDEX="${NVSHMEM_IB_GID_INDEX:-3}"       # RoCE v2
export NVSHMEM_IB_ADDR_FAMILY="${NVSHMEM_IB_ADDR_FAMILY:-AF_INET}"
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-536870912}"  # 512 MiB(DeepEP 需要比 exp02 大)
export NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-WARN}"                  # INFO 太吵,先 WARN

# DeepEP init_dist 期望的三件套(见 utils.py:14-36):
#   MASTER_ADDR / MASTER_PORT / WORLD_SIZE(=nnodes) / RANK(=node_rank)
export MASTER_ADDR="$MASTER_IB_IP"
export MASTER_PORT="$MASTER_PORT"
export WORLD_SIZE=2
export RANK="$NODE_RANK"

# 让 gloo/NCCL bootstrap 走 IB 网卡
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib1}"

TS=$(TZ='Asia/Shanghai' date '+%Y%m%d-%H%M%S')
RUN_DIR="run_deepep_${TS}_CST"
mkdir -p "$RUN_DIR"
echo "node=$NODE_RANK master=$MASTER_IB_IP:$MASTER_PORT GPU=$FREE_GPU HCA=$NVSHMEM_HCA_LIST"

# --num-processes 1  :本机只 1 张卡
# --num-tokens 128   :DeepEP 默认;跑通即可
# --hidden 7168      :DeepSeek 默认(和 indexer 无关,只是拿来测通)
# --num-topk 8       :默认
setsid "$PY" "$DEEPEP_TESTS/test_low_latency.py" \
  --num-processes 1 \
  --num-tokens 128 \
  --hidden 7168 \
  --num-topk 8 \
  >"$RUN_DIR/node${NODE_RANK}.log" 2>&1 &
PGID=$!
echo "$PGID" > "$RUN_DIR/pid_node${NODE_RANK}.txt"
echo "launched pgid=$PGID  log: $RUN_DIR/node${NODE_RANK}.log"
echo "stop with:  kill -TERM -$PGID"
wait "$PGID" && echo "DONE node $NODE_RANK" || echo "node $NODE_RANK FAILED (see log)"
