---
name: vllm-architecture
description: Layer-by-layer map of this vLLM V1 stack — request flow from OpenAI HTTP handler through AsyncLLM, the ZMQ core client, EngineCore busy loop, scheduler, multiproc executor, GPU worker/model runner, attention backends, and KV cache. Load this when reading, modifying, reviewing, or debugging engine/serving/worker code, or when tracing end-to-end request latency (the L2-trace branch).
---

# vLLM V1 Architecture Map

Use this to jump straight to the controlling file for a task instead of grep-wandering a huge repo. All paths are relative to the repo root. Line numbers drift — grep the named class/function to confirm.

## The one request-lifecycle you must know

An OpenAI request crosses **two processes** (API server ⇄ EngineCore) over **ZMQ**, then fans out to **N worker processes** (one per TP rank) over **shared-memory message queues**.

```
HTTP → OpenAIServingCompletion.create_completion   vllm/entrypoints/openai/completion/serving.py
     → engine_client.generate() [AsyncLLM]         vllm/v1/engine/async_llm.py
     → AsyncMPClient._send_input (ZMQ ROUTER send) vllm/v1/engine/core_client.py
─ process boundary ─
     → EngineCoreProc.run_busy_loop                vllm/v1/engine/core.py
        _handle_client_request (ZMQ recv, ADD)
        _process_engine_step → step()
          scheduler.schedule()                     vllm/v1/core/sched/scheduler.py
          model_executor.execute_model()           vllm/v1/executor/multiproc_executor.py
─ SHM broadcast to TP ranks ─
             WorkerProc.worker_busy_loop → Worker.execute_model   vllm/v1/worker/gpu_worker.py
               GPUModelRunner.execute_model                       vllm/v1/worker/gpu/model_runner.py
                 attention backend forward                        vllm/v1/attention/backends/*.py
          scheduler.update_from_output() → EngineCoreOutputs
     → put_output → ZMQ PUSH
─ process boundary ─
     → AsyncMPClient.process_outputs_socket (ZMQ recv)
     → AsyncLLM.output_handler → per-request AsyncStream         vllm/v1/engine/async_llm.py
     → StreamingResponse back to HTTP client
```

## Layer → owning file

**Entrypoints / serving** (`vllm/entrypoints/`)

- CLI boot: `cli/main.py` → `cli/serve.py` (`ServeSubcommand.cmd`) → `openai/api_server.py::run_server`.
- App build: `openai/api_server.py::build_app` / `init_app_state`; routers registered per task in `generate/api_router.py::register_generate_api_routers`.
- Completion (modified on L2-trace): `openai/completion/api_router.py` (`POST /v1/completions`), `openai/completion/serving.py` (`OpenAIServingCompletion`, hands off at `self.engine_client.generate(...)`).
- Chat: `openai/chat_completion/serving.py`. Offline: `entrypoints/llm.py` (`LLM.generate` → `LLMEngine.step`).
- Engine-client abstraction: `vllm/engine/protocol.py::EngineClient`. Concrete = `AsyncLLM`.

**Engine (frontend process)** (`vllm/v1/engine/`)

- `async_llm.py`: `AsyncLLM(EngineClient)`, `generate`, `output_handler` (drains outputs → streams).
- `core_client.py`: `MPClient`/`AsyncMPClient`/`SyncMPClient` — the ZMQ bridge. `add_request_async`, `_send_input` (ROUTER), `process_outputs_socket` + `get_output_async` (PULL). Msgpack-encoded frames.

**Engine core (backend process)** (`vllm/v1/engine/core.py`)

- `EngineCore.step()`: schedule → execute (non-blocking future) → grammar bitmask → update_from_output.
- `EngineCoreProc.run_busy_loop`: `_process_input_queue` + `_process_engine_step`; `_handle_client_request` (ADD/ABORT/UTILITY), `_handle_shutdown`. Output-forwarder thread does ZMQ PUSH to clients.

**Scheduler** (`vllm/v1/core/sched/scheduler.py`)

- `Scheduler.schedule()`: token-budget allocation, prefix-cache lookup, `kv_cache_manager.allocate_slots`, preemption (PRIORITY or FCFS), encoder budget → `SchedulerOutput`.
- `update_from_output()`: append sampled tokens, detect stops, free KV, build `EngineCoreOutputs`. Async variant: `async_scheduler.py`.

**KV cache** (`vllm/v1/core/`)

- `kv_cache_manager.py` (`get_computed_blocks`, `allocate_slots`, `free`), `block_pool.py` (hash→block, prefix cache insert/evict), `kv_cache_coordinator.py`, `single_type_kv_cache_manager.py`, `encoder_cache_manager.py`.
- Specs: `vllm/v1/kv_cache_interface.py` (`FullAttentionSpec`, `MLAAttentionSpec`, `SlidingWindowSpec`, `MambaSpec`, `KVCacheConfig`), `vllm/v1/kv_cache_spec_registry.py`.

**Executor** (`vllm/v1/executor/multiproc_executor.py`)

- `MultiprocExecutor`: `rpc_broadcast_mq` (SHM) broadcasts a method call to all TP/PP workers; `collective_rpc` enqueues + awaits `response_mqs`; only `output_rank` returns tensors. `non_block=True` → `FutureWrapper` so grammar mask overlaps compute.
- `WorkerProc.worker_busy_loop` dequeues and dispatches (`execute_model`, `sample_tokens`, ...).

**Worker / runner / attention** (`vllm/v1/worker/`)

- `gpu_worker.py`: `Worker(WorkerBase)` — `init_device`, `load_model`, `determine_available_memory`, `execute_model` (PP recv/send around runner). Note `use_v2_model_runner` flag selects `gpu/model_runner.py` (V2) vs `gpu_model_runner.py`.
- `gpu/model_runner.py`: `GPUModelRunner.execute_model` — `prepare_inputs`/`prepare_attn`, `set_forward_context`, three exec paths (FULL cudagraph / PIECEWISE / eager), `sample`. Helpers in `gpu/` (`input_batch.py`, `block_table.py`, `attn_utils.py`, `sample/`, `spec_decode/`).
- `vllm/v1/attention/backends/`: `registry.py::AttentionBackendEnum`; each backend (`flash_attn.py`, `flashinfer.py` [modified], `triton_attn.py`, `mla/*`) exposes Backend/Metadata/Builder/Impl. `cudagraph_dispatcher.py` picks FULL vs PIECEWISE.

**Config / models / distributed / sampling**

- Config: `vllm/config/` (`model.py::ModelConfig`, `cache.py::CacheConfig`, `parallel.py::ParallelConfig` [`tensor_parallel_size`], `scheduler.py::SchedulerConfig`, `vllm.py::VllmConfig`). Args: `vllm/engine/arg_utils.py::EngineArgs`.
- Models: `vllm/model_executor/models/registry.py` (arch→module maps, `inspect_model_cls`, Transformers fallback); interfaces in `interfaces_base.py` / `interfaces.py`. Layers in `vllm/model_executor/layers/` (`linear.py`, `fused_moe/`, `quantization/`, `rotary_embedding/`, `attention/`).
- Distributed: `vllm/distributed/parallel_state.py` (`GroupCoordinator`, `get_tp_group`, `initialize_model_parallel`). TP wired from `ParallelConfig.tensor_parallel_size` → worker calls `ensure_model_parallel_initialized`.
- Sampling: `vllm/v1/sample/` (`sampler.py`, `metadata.py`, `logits_processor/`, `ops/`). Spec decode: `vllm/v1/spec_decode/` (proposers) + `vllm/v1/worker/gpu/spec_decode/` (runner-side).

## The L2-trace branch (this branch)

Adds `print(f"... [{time.perf_counter():.6f}] [pid={os.getpid()}] <label>", flush=True)` at lifecycle chokepoints to measure per-stage latency. Trace points, in request order:

1. `core_client.py` — `add_request`, `sending`/`sent` (client→engine ZMQ egress, API-server pid).
2. `core.py` — `_handle_client_request` (engine ingress), `run_busy_loop`, `_process_engine_step`, `engine step`, `put_output`, `process_output_sockets` (engine egress).
3. `scheduler.py` — `schedule`, `update_from_output`.
4. `multiproc_executor.py` — `execute_model` (SHM RPC fire), `worker_main` (per-worker startup).
5. `async_llm.py` — `output_handler` (frontend drains output).

Latency reading: Δ between consecutive `async_llm output_handler` ≈ end-to-end iteration latency; Δ `engine step`→`put_output` ≈ schedule+execute+update; Δ `execute_model`→next `update_from_output` ≈ worker RPC round-trip. pids separate the API-server process from the EngineCore process; grep a single pid to isolate one process's timeline.

## Reading rules

- Prefer the file that *decides* behavior over files that only forward args. If a file is mostly wiring, hop once.
- Don't read both legacy (`vllm/engine/`, `vllm/worker/`) and V1 (`vllm/v1/`) stacks unless the behavior crosses them. This deployment runs **V1**.
- Pair each production file with its test dir — see the `vllm-retrieval` skill for the routing table.
