#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
VLLM_BIN=${VLLM_BIN:-"$ROOT_DIR/.venv/bin/vllm"}
PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}

HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-8000}
MODEL=${MODEL:-""}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-""}
BACKEND=${BACKEND:-"vllm"}
ENDPOINT=${ENDPOINT:-"/v1/completions"}
LABEL=${LABEL:-"served-gsm8k-eval"}
REQUEST_RATE=${REQUEST_RATE:-4}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-4}
PERF_NUM_PROMPTS=${PERF_NUM_PROMPTS:-200}
ACCURACY_NUM_QUESTIONS=${ACCURACY_NUM_QUESTIONS:-200}
ACCURACY_NUM_SHOTS=${ACCURACY_NUM_SHOTS:-5}
ACCURACY_MAX_TOKENS=${ACCURACY_MAX_TOKENS:-256}
ACCURACY_CONCURRENCY=${ACCURACY_CONCURRENCY:-8}
TEMPERATURE=${TEMPERATURE:-0}
GOODPUT_SLOS=${GOODPUT_SLOS:-"ttft:1000 tpot:200 e2el:3000"}
RESULT_DIR=${RESULT_DIR:-"$ROOT_DIR/log/vllm_bench_results"}
LOG_DIR=${LOG_DIR:-"$ROOT_DIR/log/vllm_bench_logs"}
DATASET_DIR=${DATASET_DIR:-"$ROOT_DIR/log/datasets/gsm8k"}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-180}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-600}
EXTRA_BENCH_ARGS=${EXTRA_BENCH_ARGS:-""}

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "vllm executable not found: $VLLM_BIN" >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$DATASET_DIR"

if [[ ! -f "$DATASET_DIR/train.jsonl" || ! -f "$DATASET_DIR/test.jsonl" ]]; then
    echo "GSM8K local cache missing under $DATASET_DIR" >&2
    echo "Run: bash benchmarks/download_gsm8k.sh" >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import pandas' >/dev/null 2>&1; then
    echo "Missing benchmark dependency: pandas" >&2
    echo "Install it with: uv pip install pandas" >&2
    exit 1
fi

read -r -a goodput_args <<< "$GOODPUT_SLOS"
read -r -a extra_bench_args <<< "$EXTRA_BENCH_ARGS"

cmd=(
    "$PYTHON_BIN"
    "$SCRIPT_DIR/vllm_perf_suite.py"
    evaluate-gsm8k
    --vllm-bin "$VLLM_BIN"
    --host "$HOST"
    --port "$PORT"
    --backend "$BACKEND"
    --endpoint "$ENDPOINT"
    --label "$LABEL"
    --request-rate "$REQUEST_RATE"
    --max-concurrency "$MAX_CONCURRENCY"
    --perf-num-prompts "$PERF_NUM_PROMPTS"
    --accuracy-num-questions "$ACCURACY_NUM_QUESTIONS"
    --accuracy-num-shots "$ACCURACY_NUM_SHOTS"
    --accuracy-max-tokens "$ACCURACY_MAX_TOKENS"
    --accuracy-concurrency "$ACCURACY_CONCURRENCY"
    --temperature "$TEMPERATURE"
    --result-dir "$RESULT_DIR"
    --log-dir "$LOG_DIR"
    --dataset-dir "$DATASET_DIR"
    --ready-timeout-seconds "$READY_TIMEOUT_SECONDS"
    --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS"
    --disable-tqdm
)

if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
fi

if [[ -n "$SERVED_MODEL_NAME" ]]; then
    cmd+=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [[ ${#goodput_args[@]} -gt 0 ]]; then
    cmd+=(--goodput "${goodput_args[@]}")
fi

for arg in "${extra_bench_args[@]}"; do
    cmd+=(--bench-arg "$arg")
done

"${cmd[@]}"