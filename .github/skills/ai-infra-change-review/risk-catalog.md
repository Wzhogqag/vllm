# AI Infra Risk Catalog

## Correctness Risks

- tokenizer, template, or schema drift
- fallback path returns different semantics than fast path
- quantization or dtype change alters output quality silently
- distributed state mismatch across ranks or workers

## Performance Risks

- queueing behavior changed without tail-latency tests
- hot-path allocation added in decode or scheduler loops
- cache invalidation or reuse behavior changed implicitly
- startup or warmup costs moved to first request path

## Reliability Risks

- retry loops amplify downstream failures
- partial failure handling leaves corrupted local state
- feature flag cannot cleanly disable the new path
- rollback depends on data or caches already mutated by the new path

## Observability Risks

- metric names or labels changed without migration plan
- new code path lacks stage timing or error classification
- success metric hides degraded quality or partial failure

## Rollout Risks

- too many independent variables changed at once
- no canary thresholds or stop conditions
- no cluster or model slice strategy
- no explicit owner for aborting rollout
