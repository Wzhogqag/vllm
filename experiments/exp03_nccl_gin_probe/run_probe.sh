#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# NCCL GIN 后端探测启动器 — 两机各起 1 rank。
#
# 用法(两机第二个参数都传 A 的 ib1 IP):
#   A 机: ./run_probe.sh 0 6.102.176.49
#   B 机: ./run_probe.sh 1 6.102.176.49
#
# 只探测 props.ginType/railedGinType,不做数据传输。<10 秒出结果。
set -euo pipefail
cd "$(dirname "$0")"

NODE_RANK="${1:?usage: $0 NODE_RANK A_IB_IP  (NODE_RANK is 0 or 1)}"
MASTER_IB_IP="${2:?need machine A IB IP as arg 2, same on both nodes}"
MASTER_PORT="${MASTER_PORT:-29900}"
PY=../../.venv/bin/python

# 挑空闲卡(used<2000MiB)
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 2000 {print $1; exit}')
if [[ -z "${FREE_GPU:-}" ]]; then
  echo "ERROR: no free GPU on this node" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="$FREE_GPU"

# GPU-NIC PIX 亲和推导(和 exp02 一致)
if [[ -z "${NCCL_IB_HCA:-}" ]]; then
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
  export NCCL_IB_HCA="${best_hca:-mlx5_0}"
fi

# 强制 .venv 里的新 NCCL 2.30.7 优先加载(否则 torch 会拉 /usr/lib 的 2.28.9)
NVLIB=../../.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH="$(readlink -f "$NVLIB"):${LD_LIBRARY_PATH:-}"

# RoCE 设置(和 exp01 跨机 NCCL 一致)
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

TS=$(TZ='Asia/Shanghai' date '+%Y%m%d-%H%M%S')
RUN_DIR="run_probe_${TS}_CST"
mkdir -p "$RUN_DIR"
echo "node=$NODE_RANK master=$MASTER_IB_IP:$MASTER_PORT GPU=$FREE_GPU HCA=$NCCL_IB_HCA"
echo "using NCCL from: $(readlink -f "$NVLIB")"

setsid "$PY" -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_IB_IP" --master_port="$MASTER_PORT" \
  bench_probe.py \
  >"$RUN_DIR/node${NODE_RANK}.log" 2>&1 &
PGID=$!
echo "$PGID" > "$RUN_DIR/pid_node${NODE_RANK}.txt"
echo "launched pgid=$PGID  log: $RUN_DIR/node${NODE_RANK}.log"
echo "stop with:  kill -TERM -$PGID"
wait "$PGID" && echo "DONE node $NODE_RANK" || echo "node $NODE_RANK FAILED (see log)"
