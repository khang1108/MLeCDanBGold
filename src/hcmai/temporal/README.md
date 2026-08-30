# Shared temporal alignment

`hcmai.temporal` owns the stateless ordered event-to-frame baseline used by
both public task heads. It is deliberately small: planning, score-matrix
decoding, and canonical path materialization. Task-specific formatting stays
in the KIS and TRAKE workflows.

## 1. Problem definition: ordered event-to-frame alignment

Given ordered events, return a same-video sequence of BTC keyframes whose
positions strictly increase. Every result preserves `video_id`, `frame_id`,
`frame_idx`, and `timestamp_ms`; `frame_idx` remains the competition-facing
coordinate and is never inferred from array order.

## 2. Query planning

`planner.py` maps explicit events, or a deterministic decomposition of a
query, into `AlignmentPlan` and ordered `AlignmentEvent` values. Planning does
not retrieve data, call models, or decide task-specific response formatting.

## 3. Candidate video scoring

`TemporalAlignmentService` calls `RetrievalService.score_event_videos()` once
per alignment request. Requested video and time filters are applied during
score acquisition; unsupported `min_score` filters are rejected rather than
silently ignored.

## 4. Event x frame score matrix

For each candidate video, retrieval yields a `VideoEventScores` matrix with
one row per event and one ordered column per canonical frame. The temporal
package validates returned metadata against `DataService` before exposing it
as an `AlignmentPath`.

## 5. Monotonic DP

`dp.py` is pure dynamic programming: it has no HTTP, model, or datastore
dependency. It decodes strictly increasing frame positions and ranks the best
paths across candidate videos.

```text
sum_i transform(score(event_i, frame[p_i]))
  - lambda_gap * sum_{i>1} max(0, timestamp[p_i] - timestamp[p_(i-1)])
```

`event_power`, `lambda_gap`, score depth, shortlist size, and decoder limits
are explicit `search.alignment` configuration, not hidden constants.

## 6. KIS projection

KIS aligns the planned events and deterministically projects each path to a
single representative frame while retaining the path's identity for evidence
inspection. It returns the existing `SearchResponse` shape and does not create
server-side state from the compatibility `search_id` field.

## 7. TRAKE projection

TRAKE aligns caller-provided ordered events and exposes the complete canonical
path in `TRAKEResponse`. It must keep the path chronological and preserve the
parallel `frame_ids`, `frame_idxs`, and `timestamps_ms` sequences.

## 8. Known limitations

This is a deterministic visual-event baseline. It does not model entity
identity continuity, object state transitions, multimodal dense alignment,
multi-frame VLM verification, or incremental DP. It does not establish an
HCMAI accuracy improvement without a frozen evaluation set and compatible
competition scorer.

## 9. Research extension points

Future experiments may change one measurable component at a time: enrich the
event/frame score matrix with modality-specific evidence, add a transition
term, verify top paths with multiple frames, or cache DP layers after profiling
shows recomputation is a bottleneck. Each must compare with this baseline.

The baseline equation, experiment identifier convention, and required run
record are in
[`docs/research/alignment-baseline.md`](../../../docs/research/alignment-baseline.md).

## Module ownership and testing

| Module | Owns | Does not own |
| --- | --- | --- |
| `planner.py` | deterministic event plans | retrieval or response formatting |
| `dp.py` | pure monotonic path decoding | data access or HTTP concerns |
| `service.py` | scoring, identity validation, canonical paths | KIS/TRAKE output schemas |

Run small hand-checkable fixtures with:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/temporal -q
```
