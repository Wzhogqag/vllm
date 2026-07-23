#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
VLLM_BIN=${VLLM_BIN:-"$ROOT_DIR/.venv/bin/vllm"}
PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-""}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-8000}
LABEL=${LABEL:-"vllm-perf-suite"}
DATASET_NAME=${DATASET_NAME:-"random"}
DATASET_PATH=${DATASET_PATH:-""}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-200}
REQUEST_RATES=${REQUEST_RATES:-"1,2,4,8,16"}
MAX_CONCURRENCIES=${MAX_CONCURRENCIES:-"1,2,4,8"}
GOODPUT_SLOS=${GOODPUT_SLOS:-"ttft:1000 tpot:50 e2el:3000"}
RESULT_DIR=${RESULT_DIR:-"$ROOT_DIR/log/vllm_bench_results"}
LOG_DIR=${LOG_DIR:-"$ROOT_DIR/log/vllm_bench_logs"}
START_SERVER=${START_SERVER:-1}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-180}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-""}
EXTRA_BENCH_ARGS=${EXTRA_BENCH_ARGS:-""}

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "vllm executable not found: $VLLM_BIN" >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR" "$LOG_DIR"

server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$START_SERVER" == "1" ]]; then
    read -r -a extra_serve_args <<< "$EXTRA_SERVE_ARGS"
    serve_cmd=(
        "$VLLM_BIN"
        serve
        "$MODEL"
        --host "$HOST"
        --port "$PORT"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --max-model-len "$MAX_MODEL_LEN"
    )

    if [[ -n "$SERVED_MODEL_NAME" ]]; then
        serve_cmd+=(--served-model-name "$SERVED_MODEL_NAME")
    fi

    if [[ ${#extra_serve_args[@]} -gt 0 ]]; then
        serve_cmd+=("${extra_serve_args[@]}")
    fi

    echo "Starting vLLM server on ${HOST}:${PORT}"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        "${serve_cmd[@]}" >"$LOG_DIR/server.log" 2>&1 &
    server_pid=$!
fi

read -r -a goodput_args <<< "$GOODPUT_SLOS"
read -r -a extra_bench_args <<< "$EXTRA_BENCH_ARGS"

run_cmd=(
    "$PYTHON_BIN"
    "$SCRIPT_DIR/vllm_perf_suite.py"
    run
    --vllm-bin "$VLLM_BIN"
    --model "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --label "$LABEL"
    --dataset-name "$DATASET_NAME"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS"
    --request-rates "$REQUEST_RATES"
    --max-concurrencies "$MAX_CONCURRENCIES"
    --result-dir "$RESULT_DIR"
    --log-dir "$LOG_DIR"
    --ready-timeout-seconds "$READY_TIMEOUT_SECONDS"
    --disable-tqdm
)

if [[ -n "$SERVED_MODEL_NAME" ]]; then
    run_cmd+=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [[ -n "$DATASET_PATH" ]]; then
    run_cmd+=(--dataset-path "$DATASET_PATH")
fi

if [[ ${#goodput_args[@]} -gt 0 ]]; then
    run_cmd+=(--goodput "${goodput_args[@]}")
fi

for arg in "${extra_bench_args[@]}"; do
    run_cmd+=(--bench-arg "$arg")
done

"${run_cmd[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/vllm_perf_suite.py" summarize \
    --result-dir "$RESULT_DIR" \
    --summary-csv "$RESULT_DIR/summary.csv"