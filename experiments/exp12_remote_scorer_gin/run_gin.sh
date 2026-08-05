#!/usr/bin/env bash
# exp12 GIN 跨机真打分启动器,两机各 1 rank。
#
# 用法:
#   93 (rank0, 发起端): ./run_gin.sh 0 <93_ib_ip> <payload.pt>
#   90 (rank1, 远端):   ./run_gin.sh 1 <93_ib_ip> <payload.pt>
#
# payload 用 exp10 抓取的 FS_*decode*.pt;两机都要能读到同一份(scp 一份到 90)。
set -euo pipefail
cd "$(dirname "$0")"

NODE_RANK="${1:?usage: $0 NODE_RANK MASTER_IB_IP PAYLOAD}"
MASTER_IB_IP="${2:?need master(93) ib IP}"
PAYLOAD="${3:?need payload .pt path}"
MASTER_PORT="${MASTER_PORT:-29912}"
PY=../../.venv/bin/python

# 挑空闲卡
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 2000 {print $1; exit}')
[[ -z "${FREE_GPU:-}" ]] && { echo "ERROR: no free GPU" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$FREE_GPU"

# GPU-NIC PIX 亲和选 HCA(与 exp04/09 同实现)
if [[ -z "${NCCL_IB_HCA:-}" ]]; then
  gpu_bus=$(nvidia-smi -i "$FREE_GPU" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null \
    | tr 'A-F' 'a-f' | sed -E 's/^0+([0-9a-f]{4}:)/\1/')
  gpu_path=$(readlink -f "/sys/bus/pci/devices/$gpu_bus" 2>/dev/null)
  best_hca=""; best_score=0
  for dev in /sys/class/infiniband/*; do
    hca=$(basename "$dev"); [[ "$hca" == *bond* ]] && continue
    st=$(cat "$dev"/ports/1/state 2>/dev/null); [[ "$st" == *ACTIVE* ]] || continue
    dpath=$(readlink -f "$dev/device" 2>/dev/null)
    score=$(awk -v a="$gpu_path" -v b="$dpath" 'BEGIN{
      na=split(a,A,"/"); nb=split(b,B,"/"); s=0;
      for(i=1;i<=na&&i<=nb;i++){ if(A[i]==B[i]) s++; else break } print s }')
    (( score > best_score )) && { best_score=$score; best_hca="$hca"; }
  done
  export NCCL_IB_HCA="${best_hca:-mlx5_0}"
fi

# 关键:锁定 .venv 的 NCCL 2.30.7(系统 /usr/lib 是 2.28.9,别加载错)
NVLIB=../../.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH="$(readlink -f "$NVLIB"):${LD_LIBRARY_PATH:-}"

export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

TS=$(TZ='Asia/Shanghai' date '+%Y%m%d-%H%M%S')
RUN_DIR="run_${TS}_CST"
mkdir -p "$RUN_DIR"
echo "node=$NODE_RANK master=$MASTER_IB_IP:$MASTER_PORT GPU=$FREE_GPU HCA=$NCCL_IB_HCA"
echo "nccl runtime: $($PY -c "import ctypes;l=ctypes.CDLL('$(readlink -f $NVLIB)/libnccl.so.2');v=ctypes.c_int();l.ncclGetVersion(ctypes.byref(v));print(f'{v.value//10000}.{(v.value//100)%100}.{v.value%100}')")"

setsid "$PY" -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_IB_IP" --master_port="$MASTER_PORT" \
  main_gin.py --payload "$PAYLOAD" --layers 61 --iters 10 --warmup 3 \
  >"$RUN_DIR/node${NODE_RANK}.log" 2>&1 &
PGID=$!
echo "$PGID" > "$RUN_DIR/pid_node${NODE_RANK}.txt"
echo "launched pgid=$PGID  log: $RUN_DIR/node${NODE_RANK}.log"
echo "stop: kill -TERM -$PGID"
wait "$PGID" && echo "DONE node $NODE_RANK" || echo "node $NODE_RANK FAILED (see log)"
