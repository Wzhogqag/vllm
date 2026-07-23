#!/bin/bash
set -e

# =============================================================================
# vLLM 1P1D disaggregated serving with Mooncake KV transfer.
#
#   [proxy]  --prefill--> [vllm serve  kv_producer]  (GPU A, port PREFILL_PORT)
#                          bootstrap HTTP on BOOTSTRAP_PORT (rank-0 side)
#            --decode--->  [vllm serve  kv_consumer]  (GPU B, port DECODE_PORT)
#                          pulls KV from prefill over Mooncake
#
# Override via env: MODEL_PATH, TP, PROXY_PORT, PREFILL_PORT, DECODE_PORT,
#                    BOOTSTRAP_PORT, MOONCAKE_PROTOCOL (tcp|rdma), LOG_DIR,
#                    GPU_FREE_MEM_MIB, FORCE_CVD, PREFILL_GPUS, DECODE_GPUS.
# =============================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
VLLM_BIN="$SCRIPT_DIR/.venv/bin/vllm"
PY_BIN="$SCRIPT_DIR/.venv/bin/python"
PROXY_PY="$SCRIPT_DIR/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py"

for p in "$VLLM_BIN" "$PY_BIN" "$PROXY_PY"; do
        [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }
done

# ---- Tunables --------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/models/Qwen3-8B}"
TP="${TP:-1}"                       # per-role tensor-parallel size
PROXY_PORT="${PROXY_PORT:-8000}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"   # tcp is safest without RDMA NIC
LOG_DIR="${LOG_DIR:-log}"
GPU_FREE_MEM_MIB="${GPU_FREE_MEM_MIB:-2000}"

SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL_PATH")}"

mkdir -p "$LOG_DIR"

# ---- GPU selection ---------------------------------------------------------
# Need 2*TP free GPUs total: first TP for prefill, next TP for decode.
# Manual override: set PREFILL_GPUS=... DECODE_GPUS=... .
need=$(( 2 * TP ))

if [[ -n "${PREFILL_GPUS:-}" && -n "${DECODE_GPUS:-}" ]]; then
        echo "Using manual GPU assignment: prefill=$PREFILL_GPUS decode=$DECODE_GPUS"
else
        if ! command -v nvidia-smi >/dev/null 2>&1; then
                echo "nvidia-smi not found; cannot autodetect GPUs" >&2
                exit 1
        fi
        mapfile -t FREE_GPUS < <(
                nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
                        awk -F', *' -v thr="$GPU_FREE_MEM_MIB" \
                                '$2+0 < thr { print $1 }'
        )
        if (( ${#FREE_GPUS[@]} < need )); then
                echo "Need $need free GPU(s) but found only ${#FREE_GPUS[@]}: [${FREE_GPUS[*]}]" >&2
                nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader >&2
                exit 1
        fi
        PREFILL_GPUS=$(IFS=,; echo "${FREE_GPUS[*]:0:$TP}")
        DECODE_GPUS=$(IFS=,;  echo "${FREE_GPUS[*]:$TP:$TP}")
        echo "Auto-selected: prefill=$PREFILL_GPUS decode=$DECODE_GPUS"
fi

# ---- Common env ------------------------------------------------------------
export TZ=Asia/Shanghai
# Same ephemeral-port workaround as start.sh — bind to the reserved 20000-32768
# range so EngineCore's _get_open_port doesn't fight the exhausted pool.
export VLLM_PORT="${VLLM_PORT:-30100}"
# Mooncake's C++ side still auto-probes RDMA HCAs (mlx5_*) even when the
# connector is configured with mooncake_protocol=tcp. On a host with many
# mlx5_* devices, same-host P/D endpoints then try to negotiate mismatched NIC
# paths and fail. MC_FORCE_TCP=1 tells libmooncake to skip RDMA entirely.
export MC_FORCE_TCP="${MC_FORCE_TCP:-1}"

KV_EXTRA=""
if [[ -n "$MOONCAKE_PROTOCOL" ]]; then
        KV_EXTRA=",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"$MOONCAKE_PROTOCOL\"}"
fi

COMMON_ARGS=(
        --tensor-parallel-size "$TP"
        --served-model-name "$SERVED_NAME"
        --trust-remote-code
        --max-num-batched-tokens 81920
        --no-enable-prefix-caching
        --gpu-memory-utilization 0.8
        --max-num-seqs 512
        --block-size 32
        # Disable async scheduling: with async on, VllmConfig.max_concurrent_batches=2
        # activates step_with_batch_queue in EngineCore, which races with Mooncake's
        # KV-transfer finish notifications on the decode side and trips the
        # `assert req_id in self.requests` in Scheduler._update_from_kv_xfer_finished.
        --no-async-scheduling
)

PIDS=()
cleanup() {
        echo "Stopping PD servers…" >&2
        trap - INT TERM EXIT
        for pid in "${PIDS[@]}"; do
                kill "$pid" 2>/dev/null || true
        done
        pkill -f "mooncake_connector_proxy.py" 2>/dev/null || true
}
trap cleanup INT TERM

# ---- Prefill (kv_producer) -------------------------------------------------
echo "Starting prefill: GPUs=$PREFILL_GPUS port=$PREFILL_PORT bootstrap=$BOOTSTRAP_PORT"
VLLM_MOONCAKE_BOOTSTRAP_PORT="$BOOTSTRAP_PORT" \
CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" \
"$VLLM_BIN" serve "$MODEL_PATH" --port "$PREFILL_PORT" \
        "${COMMON_ARGS[@]}" \
        --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\"$KV_EXTRA}" \
        > "$LOG_DIR/prefill.log" 2>&1 &
PIDS+=($!)

# ---- Decode (kv_consumer) --------------------------------------------------
# Give the decoder a different internal port base so both engines don't collide
# on VLLM_PORT while binding IPC sockets.
echo "Starting decode:  GPUs=$DECODE_GPUS port=$DECODE_PORT"
VLLM_PORT=30200 \
CUDA_VISIBLE_DEVICES="$DECODE_GPUS" \
"$VLLM_BIN" serve "$MODEL_PATH" --port "$DECODE_PORT" \
        "${COMMON_ARGS[@]}" \
        --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\"$KV_EXTRA}" \
        > "$LOG_DIR/decode.log" 2>&1 &
PIDS+=($!)

# ---- Wait for both HTTP servers to answer ----------------------------------
wait_for_http() {
        local port=$1 name=$2 max=1200 t=0
        echo -n "Waiting for $name on :$port "
        while (( t < max )); do
                if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
                        echo " ready"
                        return 0
                fi
                if ! kill -0 "${PIDS[0]}" 2>/dev/null || ! kill -0 "${PIDS[1]}" 2>/dev/null; then
                        echo " a vllm process died; see $LOG_DIR/{prefill,decode}.log" >&2
                        return 1
                fi
                sleep 5; t=$((t+5)); echo -n "."
        done
        echo " timeout" >&2; return 1
}

wait_for_http "$PREFILL_PORT" prefill || { cleanup; exit 1; }
wait_for_http "$DECODE_PORT"  decode  || { cleanup; exit 1; }

# ---- Proxy -----------------------------------------------------------------
echo "Starting proxy on :$PROXY_PORT"
"$PY_BIN" "$PROXY_PY" \
        --host 0.0.0.0 --port "$PROXY_PORT" \
        --prefill "http://127.0.0.1:${PREFILL_PORT}" "$BOOTSTRAP_PORT" \
        --decode  "http://127.0.0.1:${DECODE_PORT}" \
        > "$LOG_DIR/proxy.log" 2>&1 &
PIDS+=($!)

echo ""
echo "PD stack up:"
echo "  prefill:   http://127.0.0.1:${PREFILL_PORT}   (log $LOG_DIR/prefill.log)"
echo "  decode:    http://127.0.0.1:${DECODE_PORT}    (log $LOG_DIR/decode.log)"
echo "  bootstrap: http://127.0.0.1:${BOOTSTRAP_PORT}"
echo "  proxy:     http://127.0.0.1:${PROXY_PORT}    (log $LOG_DIR/proxy.log)"
echo ""
echo "Client hits the proxy, e.g."
echo "  curl -s http://127.0.0.1:${PROXY_PORT}/v1/completions \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"model\":\"$SERVED_NAME\",\"prompt\":\"Hello\",\"max_tokens\":32}'"
echo ""
echo "Ctrl-C here to stop everything. PIDs: ${PIDS[*]}"
wait
