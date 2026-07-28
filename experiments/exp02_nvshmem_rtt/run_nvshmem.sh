#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# NVSHMEM host-put RTT bench launcher. Run on BOTH machines.
#   Node 0 (machine A):  ./run_nvshmem.sh 0 <A_IB_IP>
#   Node 1 (machine B):  ./run_nvshmem.sh 1 <A_IB_IP>
# Both nodes pass the SAME value (machine A's IB IP) as the rendezvous master.
set -euo pipefail
cd "$(dirname "$0")"

NODE_RANK="${1:?usage: run_nvshmem.sh NODE_RANK A_IB_IP  (NODE_RANK is 0 or 1)}"
MASTER_IB_IP="${2:?need machine A IB IP as arg 2, same on both nodes}"
MASTER_PORT="${MASTER_PORT:-29700}"
PY=../../.venv/bin/python

# pick a free GPU on THIS node
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 2000 {print $1; exit}')
if [[ -z "${FREE_GPU:-}" ]]; then
  echo "ERROR: no free GPU on this node" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="$FREE_GPU"

# --- NVSHMEM env ---
NVSHMEM_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:${LD_LIBRARY_PATH:-}"
# Transport: IBGDA (GPU directly rings the NIC doorbell, no CPU proxy). We use
# it because ibrc's CPU-proxy path hit LOC_PROT_ERR (status 4) at first RDMA --
# that path needs GDRCopy (/dev/gdrdrv), which is absent. IBGDA has no proxy so
# it should bypass that failure, and it is the low-latency transport we want
# anyway. Toggle back to ibrc by setting NVSHMEM_IB_ENABLE_IBGDA=0.
export NVSHMEM_IB_ENABLE_IBGDA="${NVSHMEM_IB_ENABLE_IBGDA:-1}"
# IBGDA NIC handler: auto tries GPU (needs GPU UAR mapping -- failed here:
# ibgda_nic_mem_gpu_map failed) then falls back to cpu (needs GDRCopy -- absent,
# caused a spin-hang). cpu_host_memory puts the doorbell/WQ in HOST memory and
# needs NEITHER GPU UAR mapping NOR GDRCopy. Data still moves via GPU-mem RDMA;
# only the control path is on host, so latency is a bit higher but it RUNS.
export NVSHMEM_IBGDA_NIC_HANDLER="${NVSHMEM_IBGDA_NIC_HANDLER:-cpu_host_memory}"
export NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST:-mlx5_0:1}"
# RoCE GID selection (GID 3 = RoCE v2, verified in exp01 via ib_write_lat).
export NVSHMEM_IB_GID_INDEX="${NVSHMEM_IB_GID_INDEX:-3}"
export NVSHMEM_IB_ADDR_FAMILY="${NVSHMEM_IB_ADDR_FAMILY:-AF_INET}"
# small symmetric heap: this bench needs only a few KB (default is ~1GiB)
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-67108864}"  # 64 MiB
# torch.distributed rendezvous over the IB net
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib1}"
# INFO so we can confirm which transport/GID NVSHMEM actually picked
export NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-INFO}"

TS=$(TZ='Asia/Shanghai' date '+%Y%m%d-%H%M%S')
RUN_DIR="run_${TS}_CST"
mkdir -p "$RUN_DIR"
echo "node=$NODE_RANK master=$MASTER_IB_IP:$MASTER_PORT GPU=$FREE_GPU HCA=$NVSHMEM_HCA_LIST"

OUT_ARG=""
[[ "$NODE_RANK" == "0" ]] && OUT_ARG="--out $RUN_DIR/nvshmem_put.json"

setsid "$PY" -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_IB_IP" --master_port="$MASTER_PORT" \
  bench_nvshmem_put.py $OUT_ARG \
  >"$RUN_DIR/node${NODE_RANK}.log" 2>&1 &
PGID=$!
echo "$PGID" > "$RUN_DIR/pid_node${NODE_RANK}.txt"
echo "launched pgid=$PGID  log: $RUN_DIR/node${NODE_RANK}.log  (stop: kill -TERM -$PGID)"
wait "$PGID" && echo "DONE node $NODE_RANK" || echo "node $NODE_RANK FAILED (see log)"
