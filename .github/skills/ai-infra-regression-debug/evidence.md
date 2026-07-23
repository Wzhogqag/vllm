# Evidence Patterns

## Performance Regression

Strong evidence:

- same workload, same model, same hardware, same software except one variable
- repeated runs with variance reported
- a profiler view that matches the claimed bottleneck shift
- a rollback or feature disable that restores prior behavior

Weak evidence:

- one run on a noisy cluster
- changed prompt mix
- comparing warm cache to cold cache
- using average latency while ignoring tail blow-up

## Memory Regression

Strong evidence:

- memory timeline before and after change
- stable workload with explicit sequence lengths and concurrency
- proof of leak versus expected cache growth
- fragmentation metrics or allocator behavior aligned with the claim

## Startup Regression

Strong evidence:

- stage-level timing split: weight load, init, graph compile, warmup, cache prep
- proof whether slowdown is CPU-bound, IO-bound, or GPU-bound
- one-variable change between good and bad runs

## Correctness Regression

Strong evidence:

- deterministic reproduction with saved inputs and exact configs
- token-level or schema-level diff, not just a subjective output judgment
- proof whether issue originates in preprocessing, forward pass, decoding, or
  postprocessing
