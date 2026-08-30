# Unified Temporal Alignment Design

**Date:** 2026-08-30

**Status:** PROPOSED semantic migration. This is not a claim of an HCMAI
accuracy improvement. The implementation plan requires a frozen baseline,
task-contract review, and an explicit cut-over decision before legacy KIS
behavior is removed.

## Evidence and decision boundary

- **SOURCE:** The active runtime has two temporal implementations:
  `KISPipeline -> TemporalEvidenceCore.localize()` uses process-local
  progressive state and scene candidates, while
  `TRAKEPipeline -> TemporalEvidenceCore.align_ordered()` uses dense visual
  event/frame scores and monotonic DP. The current DP is strict by keyframe
  position and preserves canonical identity through `DataService`.
- **PAPER:** Ordered temporal grounding and constrained alignment are established
  problem settings. [CrossTask](https://arxiv.org/abs/1903.08225) uses ordered
  step constraints for instructional-video alignment; [Drop-DTW](https://arxiv.org/abs/2108.11996)
  studies monotonic sequence alignment with a richer drop model; the
  [temporal-grounding survey](https://arxiv.org/abs/2109.08043) situates
  natural-language-to-video temporal localization more broadly. None proves
  that this exact visual-only baseline improves HCMAI KIS.
- **PROPOSED:** Reusing the TRAKE-style alignment core for KIS makes the
  runtime smaller and the baseline easier to ablate. It changes KIS retrieval
  semantics, so it requires the migration gate in the companion plan rather
  than being represented as a neutral cleanup.

The 2026 organizer contract and scorer remain higher-priority sources. If they
require progressive interaction or non-strict frame reuse, this design must be
revised before implementation.

## Problem

The current codebase has two different temporal semantics:

- KIS uses progressive hint state, `UNKNOWN/MATCHED/EVALUATED_NO_MATCH`, video-level scoring, scene clustering, soft temporal relations, representative-frame selection, then single-frame reranking.
- TRAKE uses a dense event-by-frame matrix and monotonic dynamic programming.

The research target is one problem: given an ordered natural-language event sequence, find one chronologically coherent path of frames in one video. KIS, TRAKE, and a future VQA head should differ only in how they consume that aligned path.

## Goals

1. Make monotonic event-to-frame alignment the single temporal core.
2. Keep existing visual embedding/index/data infrastructure.
3. Make the baseline stateless and deterministic.
4. Preserve the public KIS and TRAKE response shapes during migration.
5. Make research hypotheses easy to ablate by keeping query planning, evidence scoring, alignment, and task heads separate.
6. Remove progressive scene/state abstractions that do not participate in the new semantics.

## Non-goals

1. Do not add entity tracking, state-transition verification, or a VLM sequence verifier in this refactor.
2. Do not add a VQA task contract in this refactor; the current repository has no VQA pipeline.
3. Do not redesign offline enrichment, FAISS artifacts, caption/OCR/ASR generation, or the frontend.
4. Do not optimize incremental DP yet. Recompute the current full query on every request; incremental execution can be added after profiling proves it is needed.
5. Do not make multimodal dense alignment part of the baseline. The first clean baseline uses the existing visual `score_visual_videos` mechanism behind a task-agnostic service.

## Core semantics

For ordered events `E_0 ... E_n` and a video with frames `f_0 ... f_m`, the baseline solves:

`argmax sum_i semantic(E_i, f_ti) - lambda_gap * sum_i (timestamp(t_i) - timestamp(t_{i-1}))`

subject to:

`t_0 < t_1 < ... < t_n`

The output is an aligned path containing one canonical frame per event.

The strict inequality matches the current monotonic-DP implementation: the
same keyframe cannot satisfy two events. It is a baseline assumption, not a
competition rule; permitting repeated frames is a separately tested semantic
change.

The baseline does **not** interpret nearest-neighbor presence as a binary event truth value. There is no `MATCHED/NO_MATCH/UNKNOWN` matrix in the new core.

## Architecture

```text
Public query
   |
   v
QueryPlanner
   |  ordered events
   v
TemporalAlignmentService
   |
   +--> RetrievalService.score_event_videos(...)
   |       shortlist videos + dense visual event/frame scores
   |
   +--> monotonic DP
   |
   v
AlignmentPath[]
   |
   +--> KIS head: choose one representative frame, retain path frame_ids
   |
   +--> TRAKE head: expose every aligned event frame
   |
   +--> future VQA head: send aligned frames/clip to answerer
```

## Query planning

The core consumes explicit ordered events. Query planning is outside the DP.

For research reproducibility, the first planner is deterministic:

1. If the caller supplies `events`, preserve them in order after whitespace normalization.
2. Otherwise split a multi-line query on non-empty lines.
3. Otherwise split on sentence boundaries (`.`, `!`, `?`) when this produces at least two non-empty events.
4. Otherwise treat the entire query as one event.

KIS gains an optional `events` field. TRAKE keeps its existing required `events` field. A future LLM event/coreference parser can replace this planner without touching alignment.

## Stateless baseline

`search_id` remains in KIS request/response for frontend compatibility but does not index a server-side temporal state. If absent, KIS generates a fresh opaque search id for the response. Repeated cumulative-query requests are simply recomputed from the current query text.

This intentionally deletes:

- progressive snapshot diffing,
- progressive state TTL/store/versioning,
- rescued-video backfill,
- `UNKNOWN/MATCHED/EVALUATED_NO_MATCH`,
- candidate-video coverage scoring,
- temporal scene clustering,
- soft relation parsing/scoring.

If incremental DP later becomes necessary, it must be implemented as an optimization over the same alignment semantics, not as a second temporal model.

## Configuration

Move ordered alignment settings into `SearchConfig.alignment` as one `AlignmentConfig`:

- `top_k=500`
- `max_videos=200`
- `rrf_k=60`
- `lambda_gap=1e-5`
- `event_power=1.0`
- `chunk_size=65536`
- `cluster_delta=0.0`

Remove dead/default-path settings tied only to the deleted progressive pipeline:

- `candidate_count`
- `temporal_window_ms`
- `ProgressiveSearchConfig`
- scene/backfill/progressive weights and budgets

Detach the single-frame reranker from the default KIS path. Keep the `retrieval/reranking` package available as an experiment, but do not let it overwrite path ranking in the baseline.

## Filters

The unified visual score provider preserves KIS video/time filtering. `score_event_videos` uses the visual index's existing `search_filtered()` and restricts rescored frame windows to `filtered_positions()`. `SearchFilters.min_score` is explicitly rejected in the aligned baseline because a single threshold has ambiguous semantics across multiple ordered events; it must never be silently ignored.

TRAKE calls the same service with `filters=None`.

## KIS representative frame

KIS returns one frame per aligned path while preserving all aligned frame ids in `SearchResult.frame_ids`.

Baseline policy:

- one event: the only aligned frame;
- multiple events: the middle aligned event frame, index `len(frames) // 2`.

This policy is deliberately simple and deterministic. Representative-frame research is a separate hypothesis.

The KIS candidate `final_score` is the path score. No single-frame model overwrites it.

## TRAKE

TRAKE continues to expose one frame per ordered event. Its workflow becomes a thin adapter over the same `TemporalAlignmentService` used by KIS.

## Research extension points after cleanup

The clean core should make these future experiments isolated:

1. Replace visual-only score provider with multimodal event/frame scores.
2. Add pairwise transition terms for same-entity continuity.
3. Add state-transition scores for changing object appearance.
4. Add top-B path generation and multi-frame VLM path verification.
5. Add incremental DP state for progressive UI latency.
6. Add a VQA head consuming `AlignmentPath` without changing the core.

Each experiment must be independently switchable and benchmarkable against the visual-only monotonic baseline.

## Acceptance criteria

1. KIS and TRAKE both call the same stateless alignment service.
2. A multi-event KIS query produces a chronological `frame_ids` path and one representative frame.
3. TRAKE output remains one chronological frame per event.
4. KIS video/time filters still constrain alignment, and non-null `min_score` is rejected explicitly.
5. No runtime imports remain for progressive state, scene clustering, backfill, or relation scoring.
6. Default KIS ranking is the DP path score, not a single-frame reranker score.
7. Python tests cover DP ordering, gap penalty, query planning, filters, KIS path projection, and TRAKE materialization.
8. Dense-score metadata is validated against `DataService` before an
   `AlignmentPath` is materialized: every `frame_id`, `video_id`, `frame_idx`,
   and `timestamp_ms` must agree with canonical data.
9. The legacy progressive/scene files and their config fields are deleted only
   after migration tests pass and the recorded cut-over decision accepts the
   measured KIS/TRAKE trade-off.
