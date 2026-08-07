# Pipeline observability

Online pipelines use the canonical stages `parse`, `expansion`, `encode`,
`search`, `fusion`, `video_aggregation`, `rerank`, `localization`, `answer`,
and `materialization`. A pipeline records only the stages that apply; skipped
optional stages are recorded explicitly.

Each structured stage log contains request ID, task type, duration, status,
input/output counts, backend, fallback status, and a safe error category.
Prompts, raw image payloads, answer text, credentials, and complete provider
responses are excluded. Code that must inspect user content during local
debugging must call `safe_content(..., debug=True)` explicitly; production
defaults to `[REDACTED]`.

Health exposes process-local stage counters, failure counters, fixed latency
histograms, task readiness, shared-retrieval readiness, and discovered remote
capabilities. These counters reset at process restart and are diagnostic, not
a durable benchmark store.

For benchmark trace capture, serialize `SearchResponse.trace`, both
time-to-first fields, the final response latency, warnings, configuration and
git commit into the run record under `runs/`. Do not include query content in
shared logs; keep benchmark inputs in the versioned evaluation fixture.
