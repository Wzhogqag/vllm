---
name: vllm-retrieval
description: Retrieval routing for this vLLM repo — for a given question type (serving, engine, scheduler, KV cache, worker, attention, distributed/TP, config/args, model integration, sampling, quantization, benchmarks, or latency tracing), which files to read first, which grep to run, and which tests to run. Load this at the start of any "where is X / how do I change Y / why is Z slow" task in this repo before searching.
---

# vLLM Retrieval Routing

This repo is large. Route to the smallest controlling surface first; don't broad-grep. Use `vllm-architecture` for the layer map; use this for "what do I open and what do I run."

## Step 1 — classify the question, then open these files

| Question is about… | Open first | Confirm with grep |
|---|---|---|
| OpenAI HTTP API / routes / streaming | `vllm/entrypoints/openai/api_server.py`, `.../completion/serving.py`, `.../chat_completion/serving.py` | `rg -n 'create_completion\|register_.*router\|app.state' vllm/entrypoints` |
| Offline `LLM(...)` API | `vllm/entrypoints/llm.py` | `rg -n 'def generate\|def chat\|_run_engine' vllm/entrypoints/llm.py` |
| Async engine / streaming outputs | `vllm/v1/engine/async_llm.py` | `rg -n 'class AsyncLLM\|def generate\|output_handler' vllm/v1/engine/async_llm.py` |
| API↔engine IPC (ZMQ, hangs, serialization) | `vllm/v1/engine/core_client.py` | `rg -n 'class .*MPClient\|_send_input\|process_outputs_socket\|get_output_async' vllm/v1/engine/core_client.py` |
| Engine loop / step / shutdown | `vllm/v1/engine/core.py` | `rg -n 'class EngineCore\|def step\|run_busy_loop\|_handle_client_request' vllm/v1/engine/core.py` |
| Scheduling / batching / preemption / token budget | `vllm/v1/core/sched/scheduler.py` | `rg -n 'def schedule\|update_from_output\|allocate_slots\|preempt' vllm/v1/core/sched/scheduler.py` |
| Prefix caching / KV blocks / eviction | `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/block_pool.py` | `rg -n 'get_computed_blocks\|cache_full_blocks\|evict\|allocate_slots' vllm/v1/core` |
| TP / multiproc execution / worker RPC | `vllm/v1/executor/multiproc_executor.py` | `rg -n 'collective_rpc\|rpc_broadcast_mq\|worker_busy_loop\|execute_model' vllm/v1/executor/multiproc_executor.py` |
| GPU worker init / memory / device | `vllm/v1/worker/gpu_worker.py` | `rg -n 'class Worker\|init_device\|determine_available_memory\|execute_model' vllm/v1/worker/gpu_worker.py` |
| Model runner / input prep / cudagraph / sampling call | `vllm/v1/worker/gpu/model_runner.py` | `rg -n 'def execute_model\|prepare_inputs\|prepare_attn\|set_forward_context\|def sample' vllm/v1/worker/gpu/model_runner.py` |
| Attention backend behavior/perf | `vllm/v1/attention/backends/registry.py` then the backend file (`flashinfer.py`, `flash_attn.py`, `mla/*`) | `rg -n 'class .*Backend\|class .*Impl\|class .*MetadataBuilder\|def forward' vllm/v1/attention/backends` |
| CUDA graph capture/replay | `vllm/v1/cudagraph_dispatcher.py`, `vllm/v1/worker/gpu/cudagraph_utils.py` | `rg -n 'CUDAGraphMode\|dispatch\|capture' vllm/v1/cudagraph_dispatcher.py` |
| Distributed groups / all-reduce / TP wiring | `vllm/distributed/parallel_state.py` | `rg -n 'get_tp_group\|initialize_model_parallel\|class GroupCoordinator' vllm/distributed/parallel_state.py` |
| CLI flags / engine args | `vllm/engine/arg_utils.py`, `vllm/entrypoints/openai/cli_args.py` | `rg -n '<flag-name>' vllm/engine/arg_utils.py` |
| Config dataclasses / defaults | `vllm/config/` (`model.py`, `cache.py`, `parallel.py`, `scheduler.py`, `vllm.py`) | `rg -n 'class .*Config' vllm/config` |
| Adding / debugging a model | `vllm/model_executor/models/registry.py` + the model file | `rg -n '<Arch>ForCausalLM\|"<arch>"' vllm/model_executor/models/registry.py` |
| Quantization (fp8, awq, gptq, w8a8) | `vllm/model_executor/layers/quantization/` | `rg -n 'class .*Config\|def apply\|def create_weights' vllm/model_executor/layers/quantization/<method>.py` |
| Sampling / logits processors / penalties | `vllm/v1/sample/sampler.py`, `vllm/v1/sample/logits_processor/` | `rg -n 'class Sampler\|class .*LogitsProcessor\|def forward' vllm/v1/sample` |
| Spec decode (eagle, ngram, mtp) | `vllm/v1/spec_decode/` + `vllm/v1/worker/gpu/spec_decode/` | `rg -n 'class .*Proposer\|class Speculator\|rejection' vllm/v1/spec_decode vllm/v1/worker/gpu/spec_decode` |
| Metrics / observability | `vllm/v1/metrics/`, `docs/design/metrics.md` | `rg -n 'class .*Stat\|Counter\|Histogram\|Gauge' vllm/v1/metrics` |
| Benchmarks / perf suite | `benchmarks/`, `vllm/benchmarks/` | `ls benchmarks; rg -n 'def main' benchmarks/vllm_perf_suite.py` |

## Step 2 — run the matching tests (use `.venv/bin/python`, never bare python/pytest)

| Change surface | Tests |
|---|---|
| entrypoints / serving | `tests/entrypoints/` |
| engine / scheduler / core client | `tests/v1/` (and `tests/engine/` for legacy) |
| KV cache / prefix cache | `tests/v1/core/` |
| worker / model runner / attention | `tests/v1/worker/`, `tests/kernels/` |
| distributed / TP | `tests/distributed/` |
| models / tokenizer | `tests/models/`, `tests/multimodal/` |
| quantization | `tests/quantization/` |
| sampling | `tests/v1/sample/` (and `tests/samplers/`) |

Command: `.venv/bin/python -m pytest tests/<path>/test_x.py -v`

## Step 3 — latency / tracing questions (L2-trace branch)

This branch prints `perf_counter` timestamps at lifecycle chokepoints. To answer "where is time going":

1. Read the trace-point inventory in the `vllm-architecture` skill (§ L2-trace branch).
2. Reproduce with `start.sh`; logs land in `log/vllm.log`.
3. Separate processes by pid: API-server pid emits `core_client.py`/`async_llm.py` lines; EngineCore pid emits `core.py`/`scheduler.py`/`multiproc_executor.py` lines.
4. Latency deltas: end-to-end ≈ consecutive `async_llm output_handler`; schedule+exec+update ≈ `engine step`→`put_output`; worker RPC round-trip ≈ `execute_model`→next `update_from_output`.
5. To add a trace point, match the existing format exactly: `print(f"<file>.py [{time.perf_counter():.6f}] [pid={os.getpid()}] <label>", flush=True)`.

## Repo conventions (do not violate)

- All Python env work through `uv`; run pytest as `.venv/bin/python -m pytest`.
- Python line length 88; Google-style docstrings; `pre-commit run ruff-check --all-files`.
- This is the **V1** stack. Prefer `vllm/v1/**` over legacy `vllm/engine/**` and `vllm/worker/**` unless the task explicitly crosses stacks.
- Before proposing an upstream PR, run the duplicate-work checks in `AGENTS.md`.

## When still lost

Use `Agent` with `subagent_type: Explore` for multi-location searches ("very thorough"). Reserve direct `rg`/`Read` for a single known target. Don't read both legacy and V1 for the same behavior.
