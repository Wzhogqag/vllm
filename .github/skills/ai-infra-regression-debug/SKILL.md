---
name: ai-infra-regression-debug
description: Investigate AI infra regressions in latency, throughput, memory, startup, stability, or quality by finding boundaries, eliminating confounders, and driving toward a minimal reproducible case.
---

# AI Infra Regression Debug

Use this skill when behavior became worse after a code change, config change,
dependency update, environment drift, or workload shift.

## Core Principle

Do not jump from "metric got worse" to "this commit caused it". First prove the
regression boundary, then isolate the confounder, then demand a mechanism.

## Workflow

1. Define the regression precisely.
2. Define the baseline and candidate boundary.
3. List all confounders before analyzing root cause.
4. Design the smallest reproducer that preserves the failure.
5. Decide what evidence would count as confirmation.
6. Propose one next experiment at a time.

## Required Outputs

### Regression Definition

- metric
- magnitude
- workload
- environment
- first observed time

### Suspected Boundary

- commit, image, config, dependency, model, or hardware change
- confidence level and why

### Confounders To Eliminate

- traffic mix
- prompt shape
- model revision
- cache warmness
- driver or firmware
- autoscaling and placement
- unrelated code paths in the same rollout

### Minimal Repro Plan

- smallest request set
- smallest environment difference
- exact success and failure thresholds

### Root-Cause Standard

- state what evidence would prove the issue and what would falsify it

## Debug Rules

- Prefer A/B or bisection evidence over intuition.
- If the baseline is unstable, say the experiment is invalid.
- If the suspected change is large, split it into feature flags or narrower
  toggles before explaining behavior.
- If the issue appears only in production, still force a reduced synthetic or
  replayable test case.
- Distinguish between trigger, amplifier, and symptom.

## Additional Guidance

Read [evidence.md](evidence.md) when you need concrete evidence patterns for
performance, memory, startup, or correctness regressions.

## vLLM Repository Anchors

For regressions in this repository, trace the boundary through these files and
paths before broad searching:

- offline entrypoint: `vllm/entrypoints/llm.py`
- serving entrypoint: `vllm/entrypoints/openai/api_server.py`
- V1 engine outer loop: `vllm/v1/engine/async_llm.py`
- scheduler and request lifecycle: `vllm/v1/engine/core.py`
- worker execution path: `vllm/v1/worker/`
- model execution internals: `vllm/model_executor/`
- regression-facing tests: `tests/engine/`, `tests/entrypoints/`, `tests/v1/`,
  `tests/distributed/`, `tests/test_regression.py`
- benchmark surfaces: `benchmarks/` and `vllm/benchmarks/`

Concrete repo commands:

- run a focused regression file:
  `.venv/bin/python -m pytest tests/test_regression.py -v`
- run serving-path tests:
  `.venv/bin/python -m pytest tests/entrypoints/openai tests/entrypoints/serve -v`
- run engine and V1 slices:
  `.venv/bin/python -m pytest tests/engine tests/v1 -v`
- inspect the benchmark CLI surface: `.venv/bin/vllm bench serve --help`

When the regression is performance-related, prefer pairing code inspection with
these metrics:

- `vllm:time_to_first_token_seconds`
- `vllm:inter_token_latency_seconds`
- `vllm:e2e_request_latency_seconds`
- `vllm:request_queue_time_seconds`
- `vllm:kv_cache_usage_perc`

## Examples

- throughput dropped 18 percent after a batching heuristic change
- model startup time doubled after upgrading torch and CUDA
- quantized serving path now shows intermittent correctness drift
- prefix cache hit rate fell after prompt template refactor
