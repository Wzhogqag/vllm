---
name: ai-infra-model-integration
description: Onboard a new model, backend, quantization path, scheduler mode, attention implementation, or hardware target with compatibility checks, validation planning, and rollout readiness criteria.
---

# AI Infra Model Integration

Use this skill when adding support for a new model or execution path.

## Goal

Turn model onboarding into an explicit compatibility and validation process,
instead of relying on ad hoc smoke tests.

## Workflow

1. Define the new integration surface.
2. Enumerate compatibility assumptions.
3. Plan correctness validation.
4. Plan performance and stability validation.
5. Define rollout gates and fallback path.

## Required Output

### Integration Scope

- model or backend name
- target hardware
- precision or quantization mode
- scheduler or caching assumptions

### Compatibility Checklist

- tokenizer and special-token handling
- context length and rope or position encoding assumptions
- supported attention and KV-cache behavior
- distributed or tensor-parallel assumptions
- structured output, speculative decode, prefix cache, or multimodal support

### Validation Plan

- load and startup tests
- deterministic correctness tests
- long-context and memory tests
- concurrency and tail-latency tests
- fallback and disable-path tests

### Rollout Gates

- must-pass metrics
- canary slice
- rollback trigger

## Integration Rules

- Never call a path "supported" based on one successful generation.
- Check feature interactions, not just basic forward pass.
- If one feature is unsupported, say so clearly instead of implying parity.
- Require both correctness and operational readiness.

## vLLM Repository Anchors

In this repository, model integration work usually spans these paths:

- offline and serving entrypoints: `vllm/entrypoints/llm.py`,
 `vllm/entrypoints/openai/`
- model execution stack: `vllm/model_executor/`,
 `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu_worker.py`
- model-family-specific code: `vllm/model_executor/models/` and `vllm/models/`
- config and capability wiring: `vllm/config/`, `vllm/platforms/`,
 `vllm/transformers_utils/`
- design references: `docs/design/huggingface_integration.md`,
 `docs/design/mm_processing.md`, `docs/design/attention_backends.md`
- validation surfaces: `tests/models/`, `tests/multimodal/`,
 `tests/quantization/`, `tests/entrypoints/`, `tests/weight_loading/`

Preferred repo commands:

- targeted model tests:
 `.venv/bin/python -m pytest tests/models -v`
- multimodal and quantization coverage:
 `.venv/bin/python -m pytest tests/multimodal tests/quantization -v`
- serving compatibility checks:
 `.venv/bin/python -m pytest tests/entrypoints/openai tests/entrypoints/llm -v`

Important vLLM validation dimensions:

- weight loading and revision handling
- tokenizer and chat-template compatibility
- context length and KV-cache behavior
- structured output and tool-calling support if claimed
- multimodal preprocessing if relevant
- TTFT, TPOT, queue time, and KV cache usage under load

## Examples

- add support for a new multimodal model family
- qualify a new quantized kernel path on H100
- onboard a ROCm backend for an existing model family
- enable speculative decode for a model that previously lacked it
