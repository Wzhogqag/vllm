---
name: ai-infra-benchmark-design
description: Design or review AI infra benchmarks for serving systems, kernels, schedulers, caching, batching, quantization, or hardware comparisons with realistic workloads and decision-safe interpretation.
---

# AI Infra Benchmark Design

Use this skill when creating, reviewing, or interpreting benchmarks for AI
systems.

## Objective

Produce benchmarks that are decision-safe. A benchmark is only useful if the
workload, metrics, and environment match the decision it is supposed to inform.

## Workflow

1. Name the decision the benchmark will support.
2. Define the realistic workload slices.
3. Define primary and guardrail metrics.
4. Lock down environment and warmup policy.
5. Identify likely confounders.
6. Explain how to interpret the results without over-claiming.

## Output Format

### Decision Target

- what choice this benchmark informs

### Workload Matrix

- model families
- prompt and output length buckets
- concurrency levels
- batch patterns
- cache and reuse expectations
- structured output or multimodal variations if relevant

### Metrics

- primary: throughput, p50, p95, p99, TTFT, TPOT, startup, memory
- guardrails: error rate, OOM rate, accuracy or schema success, variance

### Environment Controls

- exact hardware and count
- software stack
- placement or affinity
- warmup and repetition policy
- synthetic versus replayed traffic source

### Interpretation Rules

- what result is actionable
- what result is inconclusive
- what result is invalid because of setup defects

## Benchmark Rules

- Always tie the benchmark to a concrete decision.
- Never report only averages when tail latency matters.
- Never compare systems with different cache states unless the comparison is
  explicitly about cache state.
- Call out when a microbenchmark does not justify a product claim.
- Prefer a small number of representative workload buckets over a fake single
  "average request".

## Additional Guidance

Read [metrics.md](metrics.md) for metric definitions and common benchmark traps.

## vLLM Repository Anchors

Default benchmark entrypoints in this repository:

- benchmark docs: `benchmarks/README.md`
- benchmark scripts: `benchmarks/benchmark_latency.py`,
  `benchmarks/benchmark_serving.py`, `benchmarks/benchmark_throughput.py`,
  `benchmarks/benchmark_prefix_caching.py`,
  `benchmarks/benchmark_serving_structured_output.py`
- package benchmark CLI: `vllm/benchmarks/`
- metrics design: `docs/design/metrics.md`
- observability example: `examples/observability/prometheus_grafana/README.md`

Preferred repo commands:

- `.venv/bin/vllm bench latency --help`
- `.venv/bin/vllm bench serve --help`
- `.venv/bin/vllm bench throughput --help`
- `.venv/bin/python benchmarks/benchmark_prefix_caching.py --help`

vLLM metrics that should usually appear in benchmark interpretation:

- TTFT: `vllm:time_to_first_token_seconds`
- TPOT: `vllm:inter_token_latency_seconds`
- end-to-end latency: `vllm:e2e_request_latency_seconds`
- queue pressure: `vllm:request_queue_time_seconds`
- prefill vs decode split: `vllm:request_prefill_time_seconds`,
  `vllm:request_decode_time_seconds`
- concurrency pressure: `vllm:num_requests_running`,
  `vllm:num_requests_waiting`
- cache behavior: `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_hits`

## Examples

- compare two schedulers for long-context serving
- measure the impact of chunked prefill on TTFT and TPOT
- evaluate quantization tradeoffs across latency, memory, and output quality
- compare H100 and MI300 under the same workload contract
