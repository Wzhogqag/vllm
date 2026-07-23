# Benchmarks

This directory used to contain vLLM's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Usage

For detailed usage instructions, examples, and dataset information, see the [Benchmark CLI documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/#benchmark-cli).

For full CLI reference see:

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>

## Local Perf Suite

For repeatable TTFT, TPOT, and goodput sweeps against a local vLLM server,
use [benchmarks/run_vllm_perf_suite.sh](/workspace/weizhongqiang.3/myvllm/vllm/benchmarks/run_vllm_perf_suite.sh)
or [benchmarks/vllm_perf_suite.py](/workspace/weizhongqiang.3/myvllm/vllm/benchmarks/vllm_perf_suite.py).

Example:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct \
PORT=8000 \
REQUEST_RATES=1,2,4,8,16 \
MAX_CONCURRENCIES=1,2,4,8 \
GOODPUT_SLOS='ttft:1000 tpot:200 e2el:3000' \
bash benchmarks/run_vllm_perf_suite.sh
```

The suite saves each `vllm bench serve` result JSON to `log/vllm_bench_results/`
and writes a compact `summary.csv` with TTFT, TPOT, E2EL, throughput, and
goodput columns.

If you already started the server yourself and only want client-side accuracy
plus performance evaluation, use
[benchmarks/evaluate_served_vllm.sh](/workspace/weizhongqiang.3/myvllm/vllm/benchmarks/evaluate_served_vllm.sh).
It reads the pre-downloaded official GSM8K files from the local cache,
evaluates accuracy over the served endpoint, then runs `vllm bench serve`
against a local custom dataset materialized from GSM8K and records TTFT,
TPOT, and goodput.

Before the first run, download the official dataset once:

```bash
bash benchmarks/download_gsm8k.sh
```

That stores the original `train.jsonl` and `test.jsonl` under
`log/datasets/gsm8k/`. The evaluation flow then reads the local files directly.

Example:

```bash
PORT=8000 \
BACKEND=vllm \
REQUEST_RATE=4 \
MAX_CONCURRENCY=4 \
bash benchmarks/evaluate_served_vllm.sh
```
