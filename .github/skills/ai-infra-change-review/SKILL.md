---
name: ai-infra-change-review
description: Review AI infra code or design changes for reliability, performance, correctness, observability, rollout safety, and rollback risk, with severity-ordered findings and validation expectations.
---

# AI Infra Change Review

Use this skill for PR reviews, design reviews, rollout plans, and architecture
changes that affect serving or training-support infrastructure.

## Review Stance

Prioritize bugs, regressions, blind spots, and missing validation. Findings come
before summary. Avoid style-only commentary unless it hides operational risk.

## Review Workflow

1. Identify the change surface.
2. Ask what could regress: correctness, latency, memory, startup, fault
   tolerance, observability, rollout safety.
3. Check whether the change narrows or widens blast radius.
4. Check whether the validation matches the risk.
5. Write findings in severity order.

## Required Output

### Findings

- severity first
- each finding must include mechanism, impact, and why current validation is
  insufficient

### Open Questions

- only questions that materially affect confidence

### Missing Validation

- narrowest test, benchmark, or rollout check that would reduce risk

### Residual Risk

- short statement of what may still fail after merge

## Review Rules

- Flag unbounded memory growth, hidden queue amplification, and non-obvious
  rollback hazards aggressively.
- Treat observability regressions as product risk, not polish.
- If a performance optimization reduces determinism or debuggability, call out
  the tradeoff explicitly.
- If a change couples multiple risk surfaces, recommend splitting it.
- Distinguish hard blockers from concerns.

## Additional Guidance

Read [risk-catalog.md](risk-catalog.md) for a domain-specific checklist of AI
infra review risks.

## vLLM Repository Anchors

Map the changed files to the matching review and validation surface:

- serving API and request parsing:
  `vllm/entrypoints/openai/`, `vllm/entrypoints/serve/`, `tests/entrypoints/`
- engine, scheduling, and request lifecycle:
  `vllm/v1/engine/`, `vllm/engine/`, `tests/engine/`, `tests/v1/`
- worker and GPU execution:
  `vllm/v1/worker/`, `vllm/model_executor/`, `tests/cuda/`, `tests/distributed/`
- model integration and feature support:
  `vllm/model_executor/models/`, `tests/models/`, `tests/multimodal/`,
  `tests/quantization/`
- observability and metrics:
  `docs/design/metrics.md`, `examples/observability/`
- low-level kernels and bindings: `csrc/`, `tests/cuda/`,
  `benchmarks/attention_benchmarks/`

Review-time validation commands for this repo:

- Python changes: `.venv/bin/python -m pytest tests/path/to/affected_test.py -v`
- serving changes:
  `.venv/bin/python -m pytest tests/entrypoints/openai tests/entrypoints/serve -v`
- engine changes: `.venv/bin/python -m pytest tests/engine tests/v1 -v`
- model or quantization changes:
  `.venv/bin/python -m pytest tests/models tests/quantization tests/multimodal -v`
- static checks: `pre-commit run ruff-check --all-files`

## Examples

- scheduler rewrite that also changes queue accounting
- new attention backend with different fallback semantics
- quantization path that adds a silent precision downgrade
- rollout plan that lacks canary metrics or rollback thresholds
