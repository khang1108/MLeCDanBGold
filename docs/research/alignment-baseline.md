# Ordered event-to-frame alignment baseline

**Status:** PROPOSED baseline; not an HCMAI accuracy claim.

The active temporal implementation aligns ordered semantic events to a
strictly increasing sequence of BTC keyframes within one video. It is shared
by KIS and TRAKE, which differ only in output projection.

## Score

For event scores `s(e_i, f_j)`, the decoder selects positions
`p_1 < ... < p_n` that maximize:

```text
sum_i transform(s(e_i, f[p_i]))
  - lambda_gap * sum_{i>1} max(0, timestamp[p_i] - timestamp[p_{i-1}])
```

`event_power` controls `transform`; `lambda_gap`, `top_k`, video shortlist
size, and decoder limits are explicit `search.alignment` configuration. The
implementation keeps the highest-scoring paths per video and then ranks them
globally. `AlignedPath` carries only canonical `frame_ids`; score metadata is
validated against `DataService`, and full frame records are resolved by the
task output adapters.

## What this baseline does not do

- It does not prove an event happened; it ranks retrieval evidence.
- It does not learn query decomposition, fusion weights, or temporal
  transitions.
- It does not permit a keyframe to satisfy two ordered events.
- It does not use mutable progressive search state, scene clustering, or
  default VLM reranking.
- It does not establish accuracy without a frozen HCMAI evaluation set and
  competition-compatible scorer.

## Experiment convention

Record every comparison as:

```text
ALIGN-B0-visual-dp-<queryset>-<artifact-version>-<config-id>
```

For each run, record the query-set and artifact/index versions, code revision,
model checkpoint, full alignment configuration, hardware/provider, official
metric or clearly labelled proxy, retrieval/path recall, P50/P95 stage
latency, and representative failures. Compare one factor at a time, for
example `event_power`, `lambda_gap`, a retrieval modality, or planner input.

The migration gate remains documented in
[`2026-08-30-temporal-migration-gate.md`](2026-08-30-temporal-migration-gate.md).
No result should be called an improvement until that record contains a
reproducible measured comparison.
