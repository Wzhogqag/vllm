---
name: ai-infra-incident-triage
description: Triage AI infra incidents such as latency spikes, OOMs, crashes, error bursts, rollout regressions, or correctness drift, with containment-first investigation and explicit fact vs hypothesis separation.
---

# AI Infra Incident Triage

Use this skill for active or recent incidents in model serving, distributed
training support systems, batch inference, or deployment infrastructure.

## Goals

- restore or protect user-facing service quickly
- separate observed facts from theories
- reduce time wasted on low-signal speculation
- produce a clear next-action list with owners or data needs

## Required Behavior

When this skill is active, follow this sequence:

1. State the incident in one sentence with impact, scope, and current status.
2. Separate facts, hypotheses, and unknowns into distinct sections.
3. Prioritize containment before root-cause depth if the incident is ongoing.
4. Identify the smallest set of metrics or logs that can collapse uncertainty.
5. Build a ranked hypothesis list, but keep only hypotheses that predict a
   specific observable.
6. Prefer recent deltas: deploys, config flips, traffic mix changes, model
   changes, driver updates, hardware pool changes, dependency bumps, allocator
   behavior, scheduler changes, tokenizer or prompt shape changes.
7. End with a short action plan for the next 30 minutes, not a vague essay.

## Output Format

Use this structure:

### Incident Summary

- user or system impact
- affected paths, tenants, models, clusters, or hardware
- start time and current status

### Confirmed Facts

- only direct observations

### Leading Hypotheses

- one line per hypothesis
- each line must include why it fits and what would falsify it

### Highest-Value Next Checks

- smallest checks first
- prefer checks that distinguish between hypotheses

### Containment Options

- rollback
- traffic shed
- disable risky feature
- reduce concurrency or batch size
- fail over to known-good path

### Communication Draft

- short status update suitable for stakeholders

## Triage Rules

- Never treat correlation as cause without a boundary or mechanism.
- Never recommend broad restarts or rollouts without stating the operational
  risk.
- If observability is missing, say exactly which metric, tag, or log field is
  missing and what decision it blocks.
- If there is not enough data, ask targeted follow-up questions instead of
  producing invented certainty.
- Prefer reversible containment actions.

## Common Incident Axes

Read [checklists.md](checklists.md) when you need detailed prompts for a latency,
memory, crash, distributed coordination, or correctness incident.

## vLLM Repository Anchors

For incidents in this repository, start from the narrowest relevant surface:

- online serving path: `vllm/entrypoints/openai/api_server.py`
- offline generation path: `vllm/entrypoints/llm.py`
- V1 async engine handoff: `vllm/v1/engine/async_llm.py`
- scheduling and engine core: `vllm/v1/engine/core.py`
- GPU execution and memory behavior: `vllm/v1/worker/gpu_worker.py` and
  `vllm/v1/worker/gpu_model_runner.py`
- observability definitions: `docs/design/metrics.md`
- deployment docs: `docs/serving/`

Priority metrics for vLLM incidents:

- `vllm:e2e_request_latency_seconds`
- `vllm:time_to_first_token_seconds`
- `vllm:inter_token_latency_seconds`
- `vllm:request_queue_time_seconds`
- `vllm:request_prefill_time_seconds`
- `vllm:request_decode_time_seconds`
- `vllm:num_requests_running`, `vllm:num_requests_waiting`
- `vllm:kv_cache_usage_perc`
- `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits`

Useful vLLM commands:

- start a local server: `.venv/bin/vllm serve <model> --port 8000`
- inspect exported metrics:
  `curl -s http://127.0.0.1:8000/metrics | rg '^(vllm:|http_)'`
- validate OpenAI serving surface:
  `.venv/bin/python -m pytest tests/entrypoints/openai -v`
- validate V1-focused behavior:
  `.venv/bin/python -m pytest tests/v1 -v`

## Examples

- p99 latency doubled after a scheduler flag change
- H100 pool shows sudden OOMs after enabling a new quantized path
- only one model family returns malformed outputs after a tokenizer update
- throughput collapses during traffic bursts despite stable request count
