# Incident Checklists

Use only the section relevant to the current incident.

## Latency Spike

Check in this order:

- did request shape change: prompt length, output length, multimodal payload
- did batch formation change: queue depth, max batch size, scheduler policy,
  prefill vs decode balance
- did cache hit rate or prefix reuse drop
- did a rollout change kernel choice, attention backend, quantization path, or
  graph capture behavior
- is the issue cluster-wide, model-specific, or hardware-pool-specific
- did retry rate or downstream timeout behavior amplify the tail

## OOM Or Memory Growth

Check in this order:

- peak activation or KV footprint shifted because sequence lengths changed
- allocator fragmentation rose after workload or block-size changes
- memory leak correlates with one feature gate or code path
- host memory growth comes from preprocessing, tokenizer, or logging
- eviction, offload, or cache release policy stopped reclaiming memory

## Crash Loop Or Error Burst

Check in this order:

- exact first failing build, image, config, or driver boundary
- whether failure is deterministic per request type or per host
- whether readiness checks pass while serving path fails
- whether one dependency started timing out or returning malformed data
- whether the crash happens on startup, graph warmup, first request, or only at
  scale

## Correctness Drift

Check in this order:

- tokenizer or chat template changes
- model weight or revision mismatch
- dtype, quantization, or backend path divergence
- stop-token, logit bias, or sampling config changes
- prompt rendering or structured-output schema drift

## Distributed Coordination Issues

Check in this order:

- rank membership, rendezvous, or world-size mismatch
- one slow rank causing straggler amplification
- NCCL or transport version boundary
- topology or affinity drift after scheduling changes
- timeout values mismatched to actual startup or all-reduce behavior
