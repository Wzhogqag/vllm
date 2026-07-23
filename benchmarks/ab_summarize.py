#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Merge ab_sweep.sh artifacts into a single bottleneck-focused CSV.

For each sweep point we combine:
  - vllm bench serve result JSON (client-side latency/throughput)
  - /metrics snapshot before and after (cumulative counters -> deltas)
  - /metrics scrape trace (per-second peaks during the run)

The output CSV is what L1 of the learning plan asks for: one row per A/B
point, with both client-side numbers and the server-side metrics that tell
you *where* the bottleneck was.

Usage:
    .venv/bin/python benchmarks/ab_summarize.py \
        --run-dir log/ab_runs/baseline \
        --out-csv log/ab_runs/baseline/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import regex as re

# Columns chosen so each row tells a story:
#   client-side: how the user experienced this point
#   server-side delta: what the engine *did* during this point
#   server-side peak: what the engine *saw* at the worst moment
FIELDS = [
    "config_label",
    "point",
    "request_rate",
    "max_concurrency",
    "num_prompts",
    "completed",
    "failed",
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
    "preemptions_delta",
    "prompt_tokens_delta",
    "generation_tokens_delta",
    "prefix_cache_hit_rate_delta",
    "kv_pct_peak",
    "running_peak",
    "waiting_peak",
    "iter_tokens_avg",
    "result_file",
]


# /metrics is the Prometheus text exposition format. Lines we care about:
#   <metric_name>{labels...} <value>
# Histograms expose _sum / _count / _bucket as separate metric_name lines.
_METRIC_LINE = re.compile(
    r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def parse_metrics(path: Path) -> dict[str, float]:
    """Return a dict of {metric_name: summed_value_across_labels}.

    We collapse labels because at L1 the user has a single-engine, single-model
    server; cross-label sums match what the LoggingStatLogger prints.
    """
    totals: dict[str, float] = {}
    if not path.exists():
        return totals
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = _METRIC_LINE.match(line)
            if not m:
                continue
            name, _labels, value = m.groups()
            try:
                totals[name] = totals.get(name, 0.0) + float(value)
            except ValueError:
                continue
    return totals


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def diff_metric(after: dict[str, float], before: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def parse_trace(path: Path) -> dict[str, float]:
    """Pull peaks and a representative iter_tokens_avg from the TSV trace.

    The scraper file format (header + tab-separated rows) is defined in
    benchmarks/scrape_metrics.sh.
    """
    out = {
        "kv_pct_peak": 0.0,
        "running_peak": 0,
        "waiting_peak": 0,
        "iter_tokens_avg": 0.0,
    }
    if not path.exists():
        return out
    iter_values: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                kv = float(row.get("kv_pct", 0) or 0)
                running = int(row.get("running", 0) or 0)
                waiting = int(row.get("waiting", 0) or 0)
                iter_tokens = float(row.get("iter_tokens_avg", 0) or 0)
            except (TypeError, ValueError):
                continue
            out["kv_pct_peak"] = max(out["kv_pct_peak"], kv)
            out["running_peak"] = max(out["running_peak"], running)
            out["waiting_peak"] = max(out["waiting_peak"], waiting)
            if iter_tokens > 0:
                iter_values.append(iter_tokens)
    if iter_values:
        out["iter_tokens_avg"] = sum(iter_values) / len(iter_values)
    return out


def load_bench_result(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def point_from_filename(path: Path) -> str:
    # ab_sweep.sh writes:  {CONFIG_LABEL}-rate-{rate}-conc-{conc}.json
    name = path.stem
    # Strip the leading config_label component; the canonical "point" string
    # used elsewhere is the "rate-...-conc-..." tail.
    return re.sub(r"^.+?-rate-", "rate-", name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="The CONFIG_LABEL-scoped run dir created by ab_sweep.sh",
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    results_dir = args.run_dir / "results"
    snap_dir = args.run_dir / "snapshots"
    trace_dir = args.run_dir / "traces"
    if not results_dir.is_dir():
        parser.error(f"missing results dir: {results_dir}")

    config_label = args.run_dir.name

    rows = []
    for result_path in sorted(results_dir.glob("*.json")):
        bench = load_bench_result(result_path)
        point = point_from_filename(result_path)

        # bench serve writes metadata as a dict {"suite_label":..., "point":...}
        # but older versions write a flat list of "k=v" strings. Handle both.
        md = bench.get("metadata") or {}
        if isinstance(md, list):
            md_dict = {}
            for kv in md:
                if isinstance(kv, str) and "=" in kv:
                    k, v = kv.split("=", 1)
                    md_dict[k] = v
            md = md_dict
        cfg = md.get("config_label") or config_label
        point = md.get("point") or point

        before = parse_metrics(snap_dir / f"{point}.before.txt")
        after = parse_metrics(snap_dir / f"{point}.after.txt")
        trace = parse_trace(trace_dir / f"{point}.tsv")

        preempt_d = diff_metric(after, before, "vllm:num_preemptions_total")
        prompt_d = diff_metric(after, before, "vllm:prompt_tokens_total")
        gen_d = diff_metric(after, before, "vllm:generation_tokens_total")
        hits_d = diff_metric(after, before, "vllm:prefix_cache_hits_total")
        q_d = diff_metric(after, before, "vllm:prefix_cache_queries_total")

        rows.append(
            {
                "config_label": cfg,
                "point": point,
                "request_rate": bench.get("request_rate"),
                "max_concurrency": bench.get("max_concurrency"),
                "num_prompts": bench.get("num_prompts"),
                "completed": bench.get("completed"),
                "failed": bench.get("failed"),
                "request_throughput": bench.get("request_throughput"),
                "request_goodput": bench.get("request_goodput"),
                "output_throughput": bench.get("output_throughput"),
                "median_ttft_ms": bench.get("median_ttft_ms"),
                "p99_ttft_ms": bench.get("p99_ttft_ms"),
                "median_tpot_ms": bench.get("median_tpot_ms"),
                "p99_tpot_ms": bench.get("p99_tpot_ms"),
                "median_e2el_ms": bench.get("median_e2el_ms"),
                "p99_e2el_ms": bench.get("p99_e2el_ms"),
                "preemptions_delta": int(preempt_d),
                "prompt_tokens_delta": int(prompt_d),
                "generation_tokens_delta": int(gen_d),
                "prefix_cache_hit_rate_delta": round(safe_div(hits_d, q_d), 4),
                "kv_pct_peak": round(trace["kv_pct_peak"], 2),
                "running_peak": trace["running_peak"],
                "waiting_peak": trace["waiting_peak"],
                "iter_tokens_avg": round(trace["iter_tokens_avg"], 2),
                "result_file": result_path.name,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
