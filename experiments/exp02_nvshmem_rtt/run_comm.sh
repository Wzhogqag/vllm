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

# Pick a free GPU on THIS node, then use the RDMA NIC that is PIX-affinitized
# to it (same PCIe root). IBGDA needs the GPU to reach the NIC's resources; a
# GPU paired with a non-affinitized NIC (NODE/SYS in `nvidia-smi topo -m`) is a
# likely cause of cudaErrorIllegalAddress. On this box the affinity is 1:1 --
# GPU_k <-> the mlx5 whose netdev is ib(k+1) -- but we derive it from sysfs
# rather than hard-code, and let NVSHMEM_HCA_LIST override if set.
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 2000 {print $1; exit}')
if [[ -z "${FREE_GPU:-}" ]]; then
  echo "ERROR: no free GPU on this node" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="$FREE_GPU"

# Derive the affinitized HCA from the chosen GPU's PCIe path, unless the caller
# pinned NVSHMEM_HCA_LIST explicitly. Match = longest shared PCI-path prefix
# between the GPU and each ACTIVE mlx5 device (PIX = same PCIe switch).
if [[ -z "${NVSHMEM_HCA_LIST:-}" ]]; then
  gpu_bus=$(nvidia-smi -i "$FREE_GPU" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null \
    | tr 'A-F' 'a-f' | sed -E 's/^0+([0-9a-f]{4}:)/\1/')   # -> 0000:10:00.0
  gpu_path=$(readlink -f "/sys/bus/pci/devices/$gpu_bus" 2>/dev/null)
  best_hca=""; best_score=0
  for dev in /sys/class/infiniband/*; do
    hca=$(basename "$dev")
    [[ "$hca" == *bond* ]] && continue
    st=$(cat "$dev"/ports/1/state 2>/dev/null); [[ "$st" == *ACTIVE* ]] || continue
    dpath=$(readlink -f "$dev/device" 2>/dev/null)
    # longest shared PCI-path prefix, via awk (bash IFS splitting was unreliable)
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
# NVSHMEM_HCA_LIST is already set above (derived from the chosen GPU's PCIe
# affinity, or overridden by the caller).
# DeepEP's IBGDA env (buffer.py:108-126, production-proven). Missing these was
# a likely cause of our IBGDA instability. QP_DEPTH must exceed in-flight WRs.
export NVSHMEM_IBGDA_NUM_RC_PER_PE="${NVSHMEM_IBGDA_NUM_RC_PER_PE:-1}"
export NVSHMEM_QP_DEPTH="${NVSHMEM_QP_DEPTH:-1024}"
export NVSHMEM_MAX_TEAMS="${NVSHMEM_MAX_TEAMS:-7}"
export NVSHMEM_DISABLE_NVLS="${NVSHMEM_DISABLE_NVLS:-1}"
export NVSHMEM_CUMEM_GRANULARITY="${NVSHMEM_CUMEM_GRANULARITY:-536870912}"
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
[[ "$NODE_RANK" == "0" ]] && OUT_ARG="--out $RUN_DIR/indexer_comm.json"

setsid "$PY" -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=1 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_IB_IP" --master_port="$MASTER_PORT" \
  bench_indexer_comm.py $OUT_ARG \
  >"$RUN_DIR/node${NODE_RANK}.log" 2>&1 &
PGID=$!
echo "$PGID" > "$RUN_DIR/pid_node${NODE_RANK}.txt"
echo "launched pgid=$PGID  log: $RUN_DIR/node${NODE_RANK}.log  (stop: kill -TERM -$PGID)"
wait "$PGID" && echo "DONE node $NODE_RANK" || echo "node $NODE_RANK FAILED (see log)"
