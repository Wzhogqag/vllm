# AI Infra Skill Suite

This folder contains a practical skill set for AI infrastructure work. The
suite follows Anthropic's skill guidance: keep each skill narrow, use the
description as the trigger surface, put the default workflow in SKILL.md, and
push specialized details into sidecar files so the agent loads only what it
needs.

## Included Skills

### ai-infra-incident-triage

Use when production or staging serving has an active issue: latency spike,
error burst, OOM, crash loop, throughput collapse, or correctness drift.

Primary output:

- incident summary
- containment actions
- prioritized next checks
- explicit fact vs hypothesis split

### ai-infra-regression-debug

Use when a change caused a measurable regression in latency, throughput,
stability, memory, startup time, or model quality.

Primary output:

- regression boundary
- confounder analysis
- minimal repro plan
- proof standard for root cause

### ai-infra-benchmark-design

Use when planning or reviewing benchmarks for model serving, kernels,
schedulers, caching, batching, or deployment configuration choices.

Primary output:

- benchmark matrix
- workload realism check
- metric definitions
- interpretation guardrails

### ai-infra-change-review

Use when reviewing infra PRs or design changes for reliability, performance,
rollback safety, correctness, or observability risk.

Primary output:

- severity-ordered findings
- missing validation
- rollout and rollback concerns
- residual risk statement

### ai-infra-model-integration

Use when onboarding a new model, backend, quantization path, scheduler mode,
attention implementation, or hardware target.

Primary output:

- compatibility checklist
- correctness validation plan
- performance validation plan
- rollout readiness summary

### ai-infra-skill-authoring

Use when turning a repeated AI infra workflow into a new skill or when deciding
whether to adapt an existing skill instead of writing a new one.

Primary output:

- proposed skill boundary
- trigger description
- file layout
- evaluation plan

### vllm-repo-navigation

Use when a model needs to read, modify, or review code in this repository but
needs help choosing the right starting file, tracing the relevant call path, and
finding the matching tests and design docs quickly.

Primary output:

- best entry file for the task
- likely downstream call path
- matching tests, benchmarks, and docs
- narrow search commands instead of broad repo wandering

## How To Use This Suite Effectively

For an AI infra developer, the highest leverage pattern is not "one giant expert
skill". It is a set of narrow skills that each own one decision surface:

- incident handling
- regression isolation
- benchmark design
- change review
- model onboarding
- future skill creation
- large-repo navigation for vLLM itself

That structure improves both efficiency and accuracy:

- efficiency improves because Claude can trigger a small relevant skill instead
  of loading a large generic playbook
- accuracy improves because each skill encodes a tighter evidence standard and a
  clearer output format
- maintainability improves because you can update one workflow without changing
  all others

## vLLM-Specific Operating Notes

This repository has a few constraints that should be encoded into your daily
skill usage:

- all Python environment work should go through `uv`
- use `.venv/bin/python` for pytest invocations in non-interactive commands
- serving and engine changes usually need both code-path validation and metrics
  awareness
- this repo is large enough that guided navigation is often more important than
  raw reasoning power

High-value repo anchors:

- code: `vllm/entrypoints/`, `vllm/engine/`, `vllm/v1/`, `vllm/model_executor/`,
  `vllm/distributed/`, `vllm/config/`
- tests: `tests/entrypoints/`, `tests/engine/`, `tests/distributed/`,
  `tests/models/`, `tests/quantization/`, `tests/multimodal/`, `tests/v1/`
- docs: `docs/design/`, `docs/serving/`, `docs/configuration/`
- benchmarks: `benchmarks/` and `vllm/benchmarks/`

Default validation commands for this repo:

- `uv pip install -r requirements/lint.txt`
- `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
- `.venv/bin/python -m pytest tests/path/to/test_file.py -v`
- `pre-commit run ruff-check --all-files`

## When To Reuse Existing Anthropic Skills

Prefer existing Anthropic skills when the problem is already about a document or
artifact format rather than an AI infra reasoning workflow.

Good pairings:

- use document or word-style skills to turn an incident report into a polished
  postmortem
- use spreadsheet skills to analyze benchmark tables or compare release gates
- use presentation skills to generate design reviews or launch readiness decks
- use PDF skills to extract data from vendor benchmark reports or hardware
  qualification documents

Do not create a custom AI infra skill for these artifact-specific tasks unless
your team has strict templates, policy language, or review gates that the
built-in skill does not capture.

## When To Create A New Skill

Create a new skill only if all of the following are true:

- the task is repeated enough to justify maintenance
- the workflow has a stable evidence standard or checklist
- the model often misses the same domain context without help
- you can test whether the skill improved the output

If the task is one-off, volatile, or mostly requires live repo context, prefer a
prompt or instruction file instead.

## Packaging Notes

Each skill folder is self-contained and can be zipped individually for Claude,
or kept in this repository as shared team assets.

Recommended process:

- validate the folder name matches the skill name in frontmatter
- test with prompts that should and should not trigger the skill
- adjust the description first if triggering is too weak or too broad
- split sidecar files further only after SKILL.md becomes crowded
