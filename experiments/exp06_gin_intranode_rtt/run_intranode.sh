#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 同机 2 卡 GIN RTT bench 启动器。
#
# 单机跑,自动挑同机 2 张空闲卡,torchrun 起 2 rank(--nnodes=1 --nproc_per_node=2)。
# 目的:验证 GIN 在 LSA(NVLink)域内的时延地板 -- 期望 <1μs 单层,远优于跨机 RoCE 3.88μs。
#
# 用法(单机一条命令):
#   ./run_intranode.sh
set -euo pipefail
cd "$(dirname "$0")"

MASTER_PORT="${MASTER_PORT:-29920}"
PY=../../.venv/bin/python

# 挑 2 张空闲卡(used<2000MiB),同机
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 2000 {print $1}' | head -2 | paste -sd,)
if [[ -z "${FREE_GPUS:-}" || "${FREE_GPUS}" != *,* ]]; then
  echo "ERROR: need at least 2 free GPUs on this node" >&2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$FREE_GPUS"
echo "using GPUs: $CUDA_VISIBLE_DEVICES"

# 强制加载 .venv 里的 NCCL 2.30.7
NVLIB=../../.venv/lib/python3.12/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH="$(readlink -f "$NVLIB"):${LD_LIBRARY_PATH:-}"

# 单机不走网络,但把 GLOO/NCCL socket 显式绑到 lo 避免误用
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
# NCCL_DEBUG=INFO 时可以在日志里确认 GIN 走 LSA path 而非 IB
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

TS=$(TZ='Asia/Shanghai' date '+%Y%m%d-%H%M%S')
RUN_DIR="run_${TS}_CST"
mkdir -p "$RUN_DIR"
echo "master=127.0.0.1:$MASTER_PORT  world=2 (single node)"

setsid "$PY" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  bench_rtt.py --out "$RUN_DIR/rtt.json" \
  >"$RUN_DIR/run.log" 2>&1 &
PGID=$!
echo "$PGID" > "$RUN_DIR/pid.txt"
echo "launched pgid=$PGID  log: $RUN_DIR/run.log"
echo "stop with:  kill -TERM -$PGID"
wait "$PGID" && echo "DONE" || echo "FAILED (see log)"
