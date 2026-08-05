#!/usr/bin/env bash
# C 机上启动 V3.2 + hook 的完整脚本
# 使用: ./start_c.sh
set -euo pipefail
cd "$(dirname "$0")"

# ---- 变量 ----
MODEL="${MODEL:-/models/DeepSeek-V3.2}"
TP="${TP:-8}"
PORT="${PORT:-8000}"
MAX_SEQS="${MAX_SEQS:-16}"
MAX_LEN="${MAX_LEN:-8192}"

# 挑 8 张空闲卡
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 2000 {print $1}' | head -n "$TP" | paste -sd,)
COUNT=$(echo "$FREE_GPUS" | tr ',' '\n' | wc -l)
if [[ "$COUNT" -lt "$TP" ]]; then
  echo "ERROR: 需要 $TP 张空卡,只找到 $COUNT 张 ($FREE_GPUS)" >&2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$FREE_GPUS"

# 时间戳目录
TS=$(TZ='Asia/Shanghai' date '+%Y%m%d-%H%M%S')
RUN_DIR="run_${TS}_CST"
mkdir -p "$RUN_DIR"
export VLLM_DUMP_DIR="$(pwd)/$RUN_DIR"
export VLLM_DUMP_FULL_CALLS="${VLLM_DUMP_FULL_CALLS:-5}"

echo "=========================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES  (want $TP)"
echo "MODEL: $MODEL"
echo "OUTPUT: $RUN_DIR"
echo "DUMP FULL FIRST N CALLS: $VLLM_DUMP_FULL_CALLS"
echo "=========================================="

# 找 python(优先 .venv)
if [[ -x ../../.venv/bin/python ]]; then
  PY=../../.venv/bin/python
elif [[ -x /export/home/weizhongqiang.3/vllm/.venv/bin/python ]]; then
  PY=/export/home/weizhongqiang.3/vllm/.venv/bin/python
else
  echo "ERROR: .venv/bin/python not found. Run 'uv venv --python 3.12' + install vllm first." >&2
  exit 1
fi

setsid "$PY" launch_with_hook.py \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --enforce-eager \
  --max-num-seqs "$MAX_SEQS" \
  --max-model-len "$MAX_LEN" \
  --port "$PORT" \
  > "$RUN_DIR/server.log" 2>&1 &

PGID=$!
echo "$PGID" > "$RUN_DIR/pid.txt"
echo "started pgid=$PGID  log: $RUN_DIR/server.log"
echo "server ready 大约需要 5-15 分钟(TP=8 加载 650GB fp8 权重)"
echo ""
echo "跟进日志:  tail -f $RUN_DIR/server.log"
echo "等 ready:  grep -qE 'Application startup complete|Uvicorn running' $RUN_DIR/server.log && echo READY"
echo "停止:     kill -TERM -$PGID"
