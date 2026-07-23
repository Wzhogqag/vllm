---
name: vllm-repo-navigation
description: Navigate the large vLLM repository efficiently by choosing the right entry file, tracing likely call flows, using narrow search commands, and pairing code reads with the matching tests, docs, and benchmarks.
---

# vLLM Repository Navigation

Use this skill when working inside the vLLM repository and the main problem is
finding the right place to read or edit, not yet deciding the final code change.

## Goal

Reduce low-quality edits caused by broad searching or starting from the wrong
layer. This skill should help the model jump to the smallest controlling code
surface and the most relevant validation path.

## Workflow

1. Classify the task by surface before reading files.
2. Jump to the best entrypoint or owning abstraction.
3. Trace one likely call path, not every possible path.
4. Read the nearest tests and design docs for the same surface.
5. Stop searching once you have one falsifiable local hypothesis.

## Task Classification

Map the request to one primary surface:

- offline generation API
- online serving or OpenAI-compatible API
- V1 engine or scheduler
- worker, model runner, or GPU execution
- model integration or tokenizer behavior
- distributed or parallelism behavior
- metrics or observability
- benchmarks or performance evaluation

## Primary Entry Files

Start from these files unless the user already named a better anchor:

- offline inference API: `vllm/entrypoints/llm.py`
- online serving API: `vllm/entrypoints/openai/api_server.py`
- CLI-to-serving boot path: `vllm/entrypoints/cli/`
- V1 async engine: `vllm/v1/engine/async_llm.py`
- scheduler and request lifecycle: `vllm/v1/engine/core.py`
- worker and GPU execution: `vllm/v1/worker/gpu_worker.py`
- model execution helpers: `vllm/model_executor/`
- distributed logic: `vllm/distributed/`
- configuration wiring: `vllm/config/`
- metrics and observability: `docs/design/metrics.md`
- architecture overview: `docs/design/arch_overview.md`

## Search Playbook

Use targeted search commands before semantic wandering:

- list files in likely surfaces:
  `rg --files vllm tests docs benchmarks | rg 'entrypoints|engine|v1|worker|model_executor|distributed|metrics'`
- find major control classes:
  `rg -n 'class LLM|class AsyncLLM|class EngineCore|class GPUWorker' vllm`
- find request entry methods:
  `rg -n 'def generate\(|build_async_engine_client|register_.*router' vllm`
- find tests for a code surface:
  `rg --files tests | rg 'entrypoints|engine|v1|models|quantization|multimodal|distributed'`

## Reading Rules

- Prefer the file that decides behavior over files that only forward arguments.
- If the current file is mostly wiring, hop once to the file that computes,
  schedules, mutates state, or validates input.
- Pair each production file with one nearby test directory and one design doc
  when available.
- Avoid reading both legacy and V1 stacks unless the behavior crosses them.

## Validation Routing

After identifying the target surface, route validation narrowly:

- entrypoint changes: `tests/entrypoints/`
- engine or scheduler changes: `tests/engine/` and `tests/v1/`
- model or tokenizer changes: `tests/models/`, `tests/multimodal/`,
  `tests/quantization/`
- distributed changes: `tests/distributed/`
- performance claims: `benchmarks/` plus vLLM metrics

## Additional Guidance

Read [call-flows.md](call-flows.md) for common call paths and [path-map.md](path-map.md)
for directory ownership.

## Expected Output

When this skill is active, answer with:

- best starting file
- likely next hop
- matching tests to read or run
- any benchmark or metrics surface that should constrain the change
