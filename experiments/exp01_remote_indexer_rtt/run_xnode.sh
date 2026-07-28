#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Cross-machine RTT bench launcher. Run on BOTH machines.
#
#   Node 0 (machine A, rank 0):  ./run_xnode.sh 0 <A_IB_IP>
#   Node 1 (machine B, rank 1):  ./run_xnode.sh 1 <A_IB_IP>
#
# <A_IB_IP> is machine A's IB-network IP (the rendezvous master); BOTH nodes
# pass the SAME value (A's IP). Example: 6.102.176.49
#
# The bench traffic goes over RoCE via NCCL_IB_*; only the torchrun rendezvous
# (a tiny TCP handshake) uses this IP. We bind rendezvous to the IB net too so
# there is no dependency on the business network being routable between nodes.
set -euo pipefail
cd "$(dirname "$0")"

NODE_RANK="${1:?usage: run_xnode.sh NODE_RANK A_IB_IP  (NODE_RANK is 0 or 1)}"
MASTER_IB_IP="${2:?need machine A IB IP as arg 2, same value on both nodes}"
MASTER_PORT="${MASTER_PORT:-29600}"
PY=../../.venv/bin/python

# --- pick a free GPU on THIS node (used < 2000 MiB) and pin to it ---
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 2000 {print $1; exit}')
if [[ -z "${FREE_GPU:-}" ]]; then
  echo "ERROR: no free GPU (used<2000MiB) on this node" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="$FREE_GPU"

# --- RoCE / NCCL config (this cluster: link_layer=Ethernet => RoCE) ---
# mlx5_7 is PORT_DOWN on both nodes; use an ACTIVE HCA. mlx5_0 is active.
# GID index 3 is the usual RoCEv2 GID; override via env if ibv_devinfo -v differs.
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# Force IB transport (fail loudly instead of silently falling back to TCP).
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# Rendezvous + any NCCL bootstrap socket goes over the IB IP interface.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib1}"
# Surface what NCCL actually chose (INFO) so we can confirm it used IB not TCP.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

TS=$(TZ='Asia/Shanghai' date '+%Y%m%d-%H%M%S')
RUN_DIR="run_xnode_${TS}_CST"
mkdir -p "$RUN_DIR"
echo "node_rank=$NODE_RANK master=$MASTER_IB_IP:$MASTER_PORT GPU=$FREE_GPU"
echo "NCCL_IB_HCA=$NCCL_IB_HCA GID=$NCCL_IB_GID_INDEX IFNAME=$NCCL_SOCKET_IFNAME"

OUT_ARG=""
[[ "$NODE_RANK" == "0" ]] && OUT_ARG="--out $RUN_DIR/xnode_rtt.json"

setsid "$PY" -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_IB_IP" --master_port="$MASTER_PORT" \
  bench_xnode_rtt.py $OUT_ARG \
  >"$RUN_DIR/node${NODE_RANK}.log" 2>&1 &
PGID=$!
echo "$PGID" > "$RUN_DIR/pid_node${NODE_RANK}.txt"
echo "launched pgid=$PGID; log: $RUN_DIR/node${NODE_RANK}.log"
echo "  stop with: kill -TERM -$PGID"
wait "$PGID" && echo "DONE node $NODE_RANK" || echo "node $NODE_RANK FAILED (see log)"
