#!/bin/bash
# Client-side A/B sweep against an already-running vLLM server.
#
# Why this exists:
#   start.sh starts a long-lived server with one set of *server* flags. For L1
#   you mostly want to vary *client* pressure (request_rate, max_concurrency)
#   against that same server, and label the run with what was changed.
#   run_vllm_perf_suite.sh tries to start its own server and re-cold-boot it
#   for each config, which is the wrong shape here.
#
# What this does:
#   1. Verifies the server at HOST:PORT is up.
#   2. For each (rate, concurrency) point in the sweep:
#        a. Snapshot /metrics into a JSON BEFORE the run.
#        b. Start a background metrics scraper (scrape_metrics.sh).
#        c. Run `vllm bench serve` with the point's parameters.
#        d. Stop the scraper.
#        e. Snapshot /metrics AFTER the run.
#   3. All artifacts land under RESULT_DIR/<config_label>/ for easy diffing.
#
# Usage:
#   CONFIG_LABEL=baseline HOST=127.0.0.1 PORT=30000 SERVED_MODEL=Qwen3-8B \
#     RATES="1,4,8,16" CONCURRENCIES="none" \
#     INPUT_LEN=1024 OUTPUT_LEN=128 NUM_PROMPTS=200 \
#     bash benchmarks/ab_sweep.sh
#
# Important env vars:
#   CONFIG_LABEL    string written into every result file; the *only* way to
#                   distinguish A from B in summaries. Required.
#   SERVED_MODEL    must match --served-model-name from start.sh (Qwen3-8B
#                   for the current start.sh).
#   RATES           comma-separated request rates (req/s). Use "inf" for
#                   unthrottled bursts.
#   CONCURRENCIES   comma-separated max_concurrency values. "none" means
#                   unlimited (rate-controlled only).

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
VLLM_BIN=${VLLM_BIN:-"$ROOT_DIR/.venv/bin/vllm"}

CONFIG_LABEL=${CONFIG_LABEL:?CONFIG_LABEL is required, e.g. baseline / no-eager / mnbt-2048}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-8B}
BACKEND=${BACKEND:-vllm}
ENDPOINT=${ENDPOINT:-/v1/completions}
DATASET_NAME=${DATASET_NAME:-random}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-200}
RATES=${RATES:-"1,4,8,16"}
CONCURRENCIES=${CONCURRENCIES:-"none"}
GOODPUT_SLOS=${GOODPUT_SLOS:-"ttft:1000 tpot:200 e2el:3000"}
SCRAPE_INTERVAL=${SCRAPE_INTERVAL:-1}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-3}

BASE_URL="http://${HOST}:${PORT}"
RUN_DIR=${RUN_DIR:-"$ROOT_DIR/log/ab_runs/${CONFIG_LABEL}"}
RESULT_DIR="$RUN_DIR/results"
SNAP_DIR="$RUN_DIR/snapshots"
TRACE_DIR="$RUN_DIR/traces"
BENCH_LOG_DIR="$RUN_DIR/logs"

mkdir -p "$RESULT_DIR" "$SNAP_DIR" "$TRACE_DIR" "$BENCH_LOG_DIR"

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "vllm binary not found: $VLLM_BIN" >&2
    exit 1
fi

# 1) Server liveness check. We do this once up front so a failed start.sh
#    fails the whole sweep loudly, not silently mid-run.
if ! curl --silent --fail --max-time 5 "${BASE_URL}/health" >/dev/null; then
    echo "server is not reachable at ${BASE_URL}/health" >&2
    echo "start.sh first, or set HOST/PORT correctly" >&2
    exit 1
fi

# `vllm bench serve` needs a tokenizer locally to encode prompts and to
# count completion tokens. When the server is launched with --served-model-name
# (e.g. "Qwen3-8B") that name is NOT a HuggingFace repo, so the default
# AutoTokenizer.from_pretrained("Qwen3-8B") raises OSError. The server
# already knows the real path via the "root" field on /v1/models, so query
# it and pass --tokenizer explicitly.
TOKENIZER=${TOKENIZER:-}
if [[ -z "$TOKENIZER" ]]; then
    TOKENIZER=$(curl --silent --fail --max-time 5 "${BASE_URL}/v1/models" \
        | "$ROOT_DIR/.venv/bin/python" -c '
import json, sys
try:
    data = json.load(sys.stdin).get("data") or []
    if data:
        print(data[0].get("root") or "", end="")
except Exception:
    pass
' 2>/dev/null)
fi
if [[ -z "$TOKENIZER" || ! -e "$TOKENIZER" ]]; then
    echo "could not auto-detect tokenizer path from ${BASE_URL}/v1/models" >&2
    echo "pass TOKENIZER=/path/to/model explicitly" >&2
    exit 1
fi
echo "using tokenizer: $TOKENIZER"

snapshot_metrics() {
    # Save the raw /metrics body. Doing this in JSON-ish style would lose
    # histogram buckets; raw text is easier to diff and easier for the
    # summarize step to parse.
    local out="$1"
    curl --silent --fail --max-time 5 "${BASE_URL}/metrics" >"$out"
}

format_rate_token() {
    local rate="$1"
    if [[ "$rate" == "inf" ]]; then
        echo "inf"
    else
        # 4 -> 4, 0.5 -> 0p5
        echo "${rate//./p}"
    fi
}

IFS=',' read -r -a rate_list <<<"$RATES"
IFS=',' read -r -a conc_list <<<"$CONCURRENCIES"

run_index=0
for rate in "${rate_list[@]}"; do
    for conc in "${conc_list[@]}"; do
        run_index=$((run_index + 1))
        rate_tok=$(format_rate_token "$rate")
        point="rate-${rate_tok}-conc-${conc}"
        echo "==== [$run_index] ${CONFIG_LABEL} :: rate=${rate} concurrency=${conc} ===="

        # 2a) BEFORE snapshot. This is the cumulative counter baseline.
        before="$SNAP_DIR/${point}.before.txt"
        snapshot_metrics "$before"

        # 2b) Start scraper. We start it AFTER the before-snapshot so the
        #     trace tsv only contains the active period of this point.
        trace_file="$TRACE_DIR/${point}.tsv"
        : >"$trace_file" # truncate, so the scraper's header is fresh
        HOST="$HOST" PORT="$PORT" INTERVAL="$SCRAPE_INTERVAL" \
            OUT="$trace_file" "$SCRIPT_DIR/scrape_metrics.sh" &
        scraper_pid=$!
        # Ensure scraper dies even if bench serve crashes.
        trap 'kill "$scraper_pid" 2>/dev/null || true' EXIT INT TERM

        # 2c) The actual benchmark.
        result_file="${CONFIG_LABEL}-${point}.json"
        bench_log="$BENCH_LOG_DIR/${point}.log"
        bench_cmd=(
            "$VLLM_BIN" bench serve
            --backend "$BACKEND"
            --host "$HOST"
            --port "$PORT"
            --endpoint "$ENDPOINT"
            --model "$SERVED_MODEL"
            --served-model-name "$SERVED_MODEL"
            --tokenizer "$TOKENIZER"
            --dataset-name "$DATASET_NAME"
            --num-prompts "$NUM_PROMPTS"
            --request-rate "$rate"
            --percentile-metrics "ttft,tpot,itl,e2el"
            --metric-percentiles "50,90,95,99"
            --save-result
            --result-dir "$RESULT_DIR"
            --result-filename "$result_file"
            --metadata "config_label=${CONFIG_LABEL}"
            --metadata "point=${point}"
            --disable-tqdm
        )
        if [[ "$conc" != "none" ]]; then
            bench_cmd+=(--max-concurrency "$conc")
        fi
        if [[ "$DATASET_NAME" == "random" ]]; then
            bench_cmd+=(--random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --ignore-eos)
        fi
        if [[ -n "$GOODPUT_SLOS" ]]; then
            read -r -a goodput_args <<<"$GOODPUT_SLOS"
            bench_cmd+=(--goodput "${goodput_args[@]}")
        fi

        if ! "${bench_cmd[@]}" >"$bench_log" 2>&1; then
            echo "  bench serve FAILED for ${point}, see ${bench_log}" >&2
            kill "$scraper_pid" 2>/dev/null || true
            trap - EXIT INT TERM
            continue
        fi

        # 2d) Stop the scraper before snapshotting after, so the after-snapshot
        #     reflects steady state (no inflight bench request).
        kill "$scraper_pid" 2>/dev/null || true
        wait "$scraper_pid" 2>/dev/null || true
        trap - EXIT INT TERM

        # 2e) AFTER snapshot. preemptions_after - preemptions_before is the
        #     count actually attributable to this benchmark point.
        after="$SNAP_DIR/${point}.after.txt"
        snapshot_metrics "$after"

        echo "  saved: $RESULT_DIR/$result_file"
        echo "  trace: $trace_file"

        # Brief cooldown so the next point starts from a clean state
        # (running -> 0, waiting -> 0). Without this, back-to-back high-rate
        # runs bleed into each other.
        sleep "$COOLDOWN_SECONDS"
    done
done

# 3) Summarize at the end. Failures here are non-fatal — raw artifacts are
#    still on disk and can be re-summarized.
if [[ -x "$SCRIPT_DIR/ab_summarize.py" || -f "$SCRIPT_DIR/ab_summarize.py" ]]; then
    PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}
    "$PYTHON_BIN" "$SCRIPT_DIR/ab_summarize.py" \
        --run-dir "$RUN_DIR" \
        --out-csv "$RUN_DIR/summary.csv" || \
        echo "summary step failed; rerun with: ab_summarize.py --run-dir $RUN_DIR"
fi

echo
echo "all artifacts under: $RUN_DIR"
echo "  - results/*.json     vllm bench serve output (client-side TTFT/TPOT/...)"
echo "  - snapshots/*.before /metrics dump before each point"
echo "  - snapshots/*.after  /metrics dump after each point"
echo "  - traces/*.tsv       time series of running/waiting/kv_pct/preempt during the run"
echo "  - summary.csv        merged view"
