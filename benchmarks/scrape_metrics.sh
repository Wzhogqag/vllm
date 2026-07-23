#!/bin/bash
# Poll the vLLM /metrics endpoint and append a single-row-per-sample TSV.
#
# Why this exists:
#   The Prometheus /metrics endpoint exposes counters and gauges that name the
#   bottleneck (KV usage, preemptions, running/waiting reqs, prefix-cache hits).
#   We want a time series we can correlate with a `vllm bench serve` run, not a
#   one-shot dump.
#
# Output format: tab-separated, one row per sample, with a header. Columns:
#   ts                    wall-clock seconds since epoch (server-side)
#   running               vllm:num_requests_running (gauge)
#   waiting               vllm:num_requests_waiting (gauge)
#   kv_pct                vllm:kv_cache_usage_perc * 100 (gauge, %)
#   preemptions_total     vllm:num_preemptions_total (cumulative counter)
#   prefix_hits_total     vllm:prefix_cache_hits_total
#   prefix_queries_total  vllm:prefix_cache_queries_total
#   iter_tokens_avg       vllm:iteration_tokens_total_sum / _count (avg tokens/step)
#
# Usage:
#   benchmarks/scrape_metrics.sh                              # default localhost:30000, 1s interval
#   HOST=127.0.0.1 PORT=30000 INTERVAL=2 OUT=log/metrics.tsv benchmarks/scrape_metrics.sh
#
# Run in the background while benchmarking:
#   benchmarks/scrape_metrics.sh &
#   SCRAPER_PID=$!
#   bash benchmarks/ab_sweep.sh ...
#   kill $SCRAPER_PID

set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
INTERVAL=${INTERVAL:-1}
OUT=${OUT:-log/metrics_trace.tsv}

mkdir -p "$(dirname "$OUT")"

URL="http://${HOST}:${PORT}/metrics"

# Header is only written when the file is new or empty, so multiple runs append.
if [[ ! -s "$OUT" ]]; then
    printf "ts\trunning\twaiting\tkv_pct\tpreemptions_total\tprefix_hits_total\tprefix_queries_total\titer_tokens_avg\n" >"$OUT"
fi

# Single-sample helper. We pull all matching gauges in one curl to keep the
# scrape itself cheap (a Prometheus /metrics call is O(N) in metric count, so
# don't do 8 separate curls).
sample_once() {
    local body
    body=$(curl --silent --max-time 2 --fail "$URL" 2>/dev/null || true)
    if [[ -z "$body" ]]; then
        # Server down or unreachable; skip the tick rather than poisoning the row
        return 0
    fi

    awk -v ts="$(date +%s)" '
        # Helper: take the last numeric field on a line, ignoring labels.
        # Lines look like:  vllm:num_requests_running{engine="0",...} 3
        function val(line,    a, n) {
            n = split(line, a, /[ \t]+/);
            return a[n];
        }
        # Skip comments and empty lines
        /^[# ]/ { next }
        $0 == "" { next }

        # Strip everything after the closing "}" so we can match on metric name
        {
            line = $0
            name = line
            sub(/[{ \t].*$/, "", name)
        }

        name == "vllm:num_requests_running"        { running        += val(line) + 0 }
        name == "vllm:num_requests_waiting"        { waiting        += val(line) + 0 }
        name == "vllm:kv_cache_usage_perc"         { kv             += val(line) + 0; kv_n += 1 }
        name == "vllm:num_preemptions_total"       { preempt        += val(line) + 0 }
        name == "vllm:prefix_cache_hits_total"     { phits          += val(line) + 0 }
        name == "vllm:prefix_cache_queries_total"  { pqueries       += val(line) + 0 }
        name == "vllm:iteration_tokens_total_sum"  { iter_sum       += val(line) + 0 }
        name == "vllm:iteration_tokens_total_count"{ iter_count     += val(line) + 0 }

        END {
            # gauge: mean across engines if multi-engine; else single value
            kv_pct = (kv_n > 0) ? (kv * 100.0 / kv_n) : 0
            iter_avg = (iter_count > 0) ? (iter_sum / iter_count) : 0
            printf "%d\t%d\t%d\t%.2f\t%d\t%d\t%d\t%.2f\n",
                   ts, running, waiting, kv_pct, preempt, phits, pqueries, iter_avg
        }
    ' <<<"$body" >>"$OUT"
}

echo "scraping ${URL} every ${INTERVAL}s -> ${OUT} (Ctrl-C to stop)" >&2
while true; do
    sample_once
    sleep "$INTERVAL"
done
