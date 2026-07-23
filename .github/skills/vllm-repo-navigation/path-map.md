# vLLM Path Map

Use this map to pick the correct starting directory quickly.

## Core Runtime

- `vllm/entrypoints/`: user-facing APIs, CLI, serving routers
- `vllm/engine/`: compatibility layer and engine-facing protocol surfaces
- `vllm/v1/engine/`: current V1 request lifecycle, async engine, scheduler core
- `vllm/v1/worker/`: GPU and device worker implementations
- `vllm/model_executor/`: model loading, execution helpers, layers, kernels,
  warmup, offload

## Supporting Systems

- `vllm/config/`: configuration schema and option wiring
- `vllm/distributed/`: distributed inference logic
- `vllm/platforms/`: hardware and platform specialization
- `vllm/tokenizers/` and `vllm/transformers_utils/`: tokenizer and HF integration
- `vllm/tool_parsers/` and `vllm/reasoning/`: tool-calling and reasoning output

## Validation Surfaces

- `tests/entrypoints/`: API and serving behavior
- `tests/engine/`: engine semantics and lifecycle
- `tests/v1/`: V1-specific behavior
- `tests/models/`: model support and feature coverage
- `tests/quantization/`: quantization paths
- `tests/multimodal/`: multimodal support
- `tests/distributed/`: TP, DP, EP, and distributed behavior
- `tests/cuda/`: lower-level accelerator behavior

## Performance And Docs

- `benchmarks/`: benchmark scripts and specialized performance studies
- `vllm/benchmarks/`: benchmark package and serving dataset utilities
- `docs/design/`: architecture and subsystem design docs
- `docs/serving/`: deployment and serving behavior
