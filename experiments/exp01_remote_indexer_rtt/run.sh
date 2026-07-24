#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Scheme A remote-indexer RTT bench runner.
#   - probes for free GPUs (used < 2000 MiB) unless two indices are given
#   - Beijing-timestamped run dir
#   - setsid into its own process group + PID file; stop with:
#       kill -TERM -$(cat <run_dir>/pids.txt)
set -euo pipefail
cd "$(dirname "$0")"

PY=../../.venv/bin/python
# torch is inherited from system site-packages, so there is no venv console
# script; invoke the launcher as a module through the venv interpreter.
TORCHRUN=(../../.venv/bin/python -m torch.distributed.run)

pick_gpus() {
  if [[ $# -ge 2 ]]; then echo "$1 $2"; return; fi
  mapfile -t free < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 2000 {print $1}')
  if [[ ${#free[@]} -lt 2 ]]; then
    echo "ERROR: need 2 free GPUs (used<2000MiB), found ${#free[@]}: ${free[*]:-none}" >&2
    exit 1
  fi
  echo "${free[0]} ${free[1]}"
}

read -r G0 G1 <<< "$(pick_gpus "$@")"
export CUDA_VISIBLE_DEVICES="$G0,$G1"

TS=$(TZ='Asia/Shanghai' date '+%Y%m%d-%H%M%S')
RUN_DIR="run_${TS}_CST"
mkdir -p "$RUN_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES | run dir: $RUN_DIR"

# Run the 2-rank job in its own process group so we never wildcard-kill.
run_one() {
  local name="$1"; shift
  echo ">>> $name"
  setsid "${TORCHRUN[@]}" --nproc_per_node=2 --nnodes=1 \
    --master_addr=127.0.0.1 --master_port=29591 \
    "$@" >"$RUN_DIR/${name}.log" 2>&1 &
  local pgid=$!
  echo "$pgid" >> "$RUN_DIR/pids.txt"
  wait "$pgid" || { echo "FAILED: $name (see $RUN_DIR/${name}.log)"; return 1; }
  echo "    log: $RUN_DIR/${name}.log"
}

run_one p2p_rtt bench_p2p_rtt.py --out "$RUN_DIR/p2p_rtt.json"

echo "DONE. Results in $RUN_DIR/"
