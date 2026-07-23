# Common Call Flows

Use these as default hypotheses, not guaranteed truth for every task.

## Offline Generation

Start here:

- `vllm/entrypoints/llm.py`

Typical flow:

- `LLM.generate(...)`
- V1 engine creation through `vllm.v1.engine.llm_engine`
- request handling and scheduling in `vllm/v1/engine/core.py`
- GPU execution in `vllm/v1/worker/gpu_worker.py`
- model runner work in `vllm/v1/worker/gpu_model_runner.py`

Read with:

- `tests/entrypoints/llm/`
- `tests/engine/`
- `tests/v1/`

## Online Serving

Start here:

- `vllm/entrypoints/openai/api_server.py`

Typical flow:

- FastAPI app creation and router registration
- `build_async_engine_client_from_engine_args(...)`
- `vllm/v1/engine/async_llm.py`
- engine core loop in `vllm/v1/engine/core.py`
- worker execution in `vllm/v1/worker/`

Read with:

- `tests/entrypoints/openai/`
- `tests/entrypoints/serve/`
- `docs/serving/online_serving/`

## Metrics And Observability

Start here:

- `docs/design/metrics.md`

Typical flow:

- metric semantics in docs
- API server exposition via `/metrics`
- engine-core event generation and timing in V1 engine paths

Read with:

- `examples/observability/prometheus_grafana/README.md`
- `tests/entrypoints/serve/` when metrics exposure changes

## Model Integration

Start here:

- `vllm/model_executor/`
- `vllm/model_executor/models/`

Typical flow:

- config or model capability wiring
- model runner preparation
- worker execution path
- entrypoint compatibility for offline and online usage

Read with:

- `tests/models/`
- `tests/multimodal/`
- `tests/quantization/`
- `tests/weight_loading/`
