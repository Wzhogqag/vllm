# Metrics And Traps

## Metric Definitions

- TTFT: time to first token; sensitive to queueing, prefill, and scheduling
- TPOT: time per output token; useful for decode-path efficiency
- throughput: specify units clearly, such as requests per second or tokens per
  second
- utilization: never use as a success metric by itself
- memory: report both peak and steady-state if relevant

## Common Traps

- measuring only one prompt length bucket
- mixing streaming and non-streaming requests without separating results
- hiding instability by reporting a single best run
- claiming product wins from synthetic microbenchmarks only
- failing to state whether prefix caching or reuse was enabled
- ignoring startup or warm-cache costs when rollout decisions depend on them

## Review Questions

- does this workload resemble real production distributions
- is the benchmark sensitive to one hidden bottleneck that dominates everything
- would a rollout owner feel safe making a launch decision from these numbers
