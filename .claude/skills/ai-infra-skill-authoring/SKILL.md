---
name: ai-infra-skill-authoring
description: Create or refine AI infra skills by choosing the right workflow boundary, writing strong trigger descriptions, structuring progressive disclosure files, and defining an evaluation plan.
---

# AI Infra Skill Authoring

Use this skill when you want to codify a repeated AI infra workflow into a skill,
or when deciding whether to adapt an existing skill instead.

## Decision Rule

Create a new skill only if the workflow is repeated, stable enough to encode,
and measurably improved by better domain context. Otherwise, use a prompt,
instruction file, or existing skill.

## Workflow

1. Name the repeated workflow.
2. Define the decision boundary the skill owns.
3. Write a trigger description that says what it does and when to use it.
4. Keep SKILL.md short and procedural.
5. Move rarely needed details into sidecar files.
6. Define prompts that should trigger and should not trigger the skill.
7. Iterate based on actual failures, not imagined ones.

## Required Output

### Workflow Boundary

- what the skill owns
- what it explicitly does not own

### Trigger Description Draft

- under the platform limit
- contains task type, scope, and trigger phrases

### File Layout

- SKILL.md for default workflow
- sidecar files for special cases, evidence catalogs, checklists, or templates
- scripts only when deterministic execution clearly beats pure prompting

### Evaluation Plan

- success prompts
- near-miss prompts
- false-trigger prompts
- quality rubric

## Authoring Rules

- Optimize description quality before adding more body text.
- Prefer multiple narrow skills over one sprawling skill.
- If a task mostly needs repo-wide standing policy, use instructions instead of a
  skill.
- If a task is single-shot and parameterized, use a prompt instead of a skill.
- Every sidecar file should exist to save context, not to hide essential steps.

## Additional Guidance

Read [patterns.md](patterns.md) for reusable design patterns and anti-patterns.

## vLLM Repository Anchors

When authoring skills for this repository, prefer encoding repo reality rather
than generic infra advice.

High-value sources to bundle into a vLLM-specific skill:

- architecture and process model: `docs/design/arch_overview.md`
- metrics and observability: `docs/design/metrics.md`
- serving and deployment docs: `docs/serving/`
- benchmark entrypoints: `benchmarks/README.md`, `benchmarks/`
- main code surfaces: `vllm/entrypoints/`, `vllm/engine/`, `vllm/v1/`,
  `vllm/model_executor/`, `vllm/distributed/`
- validation surfaces: `tests/entrypoints/`, `tests/engine/`, `tests/v1/`,
  `tests/models/`, `tests/quantization/`, `tests/multimodal/`

Default authoring commands for this repo:

- list candidate files: `rg --files vllm tests docs benchmarks .github/skills`
- find ownership anchors:
  `rg -n 'class LLM|class AsyncLLM|class EngineCore|class GPUWorker' vllm`
- validate skill file layout:
  `find .github/skills -mindepth 1 -maxdepth 1 -type d | sort`

If a future skill is only about navigating this codebase, keep it separate from
incident, benchmark, or review skills. Navigation is its own workflow boundary.

## Examples

- turn repeated GPU incident response into a team skill
- package a benchmark review workflow for launch decisions
- split a broad serving skill into triage, benchmarking, and review skills
