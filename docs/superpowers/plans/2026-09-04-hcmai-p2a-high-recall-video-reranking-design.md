# HCMAI P2a High-Recall Video-Level Reranking Design

**Date:** 2026-09-04  
**Status:** User-approved design; ready for implementation planning after final review  
**Primary metric:** Recall@20  
**Secondary metrics:** Recall@100, Recall@200 candidate union, median GT rank, MRR  
**Baseline inspected:** `src_hcmai_v16.zip`

## 1. Problem

The current KIS and TRAKE workflows share `TemporalSearchService`. KIS splits a narrative query into events, receives ranked temporal paths, and projects each path to one representative frame. TRAKE consumes explicit events and exposes the full temporal path.

Empirical behavior shows that TRAKE can find the correct narrative/video while KIS Recall@20 is poor. A representative failure is a query describing five people, a yellow animal, a pumpkin-like object being hidden, and a man waking the animal. A biology lecture can rank above the correct scene because fragmented textual/generic evidence such as `person`, `animal`, OCR, and ASR can accumulate into a strong temporal score.

The immediate P2a problem is therefore:

> Temporal/event evidence is useful, but it is too risky to be the sole authority for KIS video ranking.

P2a changes the ranking architecture, not the temporal recurrence.

## 2. Goal

Build a high-recall video-level reranking layer that:

1. keeps full-query dense retrieval as the semantic recall anchor;
2. keeps event-level retrieval as a local/discriminative rescue source;
3. keeps temporal alignment as a narrative/coherence source;
4. unions these views into at most 200 unique videos;
5. selects two complementary representative frames per video:
   - `F_global`: best full-query frame;
   - `F_temporal`: representative from the temporal path;
6. reuses the existing VLM `RerankingService`;
7. computes one video score using `max(frame reranker scores)`;
8. ranks unique videos and returns Top-20;
9. lets KIS return the frame that actually wins reranking;
10. leaves TRAKE path coordinates unchanged.

Primary rollout objective: **Recall@20**.

## 3. Non-goals

P2a does not implement:

- optional/skip events;
- confidence-adaptive event weights;
- entity/coreference consistency;
- learned LTR;
- new embeddings or indexes;
- artifact regeneration;
- DP semantic changes;
- beam search;
- hard reranker thresholds;
- latency optimization as a rollout gate.

Those remain P2b/P3 topics.

## 4. Existing code to reuse

The v16 source already contains:

```text
src/hcmai/retrieval/reranking/
├── pipeline.py
├── config.py
├── adapters/
└── models/contracts.py
```

The frame-level provider boundary is:

```python
RerankingService.rerank(
    query: str,
    candidates: Sequence[RetrievalCandidate],
) -> list[RetrievalCandidate]
```

It resolves canonical frame images, batches model calls, writes `reranker_score`, sets `final_score`, and returns candidates ordered by that score.

`RetrievalService` already exposes:

```python
search(query, top_k)
search_batch(queries, top_k)
score_event_videos(events, ...)
```

`SearchService` currently builds one `TemporalSearchService` shared by `KISPipeline` and `TRAKEPipeline`.

`setup.py` intentionally leaves reranking detached from online KIS/TRAKE. P2a wires it in at the composition root rather than constructing it inside requests.

## 5. Target architecture

```text
                     QUERY
                       |
          +------------+------------+
          |            |            |
          v            v            v
   full-query dense  event search  temporal search
          |            |            |
          +------------+------------+
                       |
                       v
              unique-video union
                 <= 200 videos
                       |
              +--------+--------+
              |                 |
              v                 v
           F_global         F_temporal
              |                 |
              +--------+--------+
                       |
                       v
               existing VLM
                  reranker
                       |
                       v
               R_video = max
                       |
                       v
                 Top-20 videos
                  /          \
                 v            v
               KIS          TRAKE
          winner frame   existing path
```

Core rule:

> Dense protects recall. Temporal supplies coherence. The VLM reranker decides final video ordering.

## 6. Candidate source A: global full-query dense

Search the complete user query as one semantic unit. This preserves joint specificity that event decomposition can lose.

Use the existing retrieval service. The current competition-scale raw candidate budget (~500 frame hits) is sufficient as an initial default and does not need to be increased merely for P2a.

For every video retain the strongest global hit:

```text
global_frame_id
global_score
global_rank
```

This frame becomes `F_global`.

## 7. Candidate source B: event-wise retrieval

For dynamic events `E1..EN`, retrieve each event independently, preferably through `search_batch`.

Merge event candidates round-robin:

```text
E1 rank 1
E2 rank 1
...
EN rank 1
E1 rank 2
...
```

This prevents one generic event from monopolizing candidate capacity.

Retain per-event provenance:

```text
event_index
frame_id
score
rank
```

No assumption that `N = 4`.

## 8. Candidate source C: temporal search

Retain video candidates from the existing temporal search.

P2a does not change:

- temporal DP recurrence;
- frame coordinates;
- timestamps;
- path semantics.

Temporal search contributes `temporal_path`, `temporal_rank`, and a deterministic representative frame `F_temporal`.

## 9. Video-level union

Target:

```text
candidate_max_videos = 200
```

One video consumes exactly one candidate slot regardless of how many frames/events retrieve it.

Use deterministic quota-preserving union rather than raw-score concatenation.

Initial evaluation targets:

```text
global     ~120 slots
events      ~50 slots
temporal    ~30 slots
```

These are initial reservoir allocations, not model weights.

Overlap consumes one slot. Freed capacity is filled from remaining ranked lists until 200 unique videos are collected or all lists are exhausted.

Global retrieval remains the largest source because it is the semantic recall anchor.

Do not directly compare raw dense/event/DP scores to decide union membership.

## 10. Internal video candidate contract

Introduce an internal video-level object conceptually equivalent to:

```python
@dataclass(frozen=True, slots=True)
class EventCandidateHit:
    event_index: int
    frame_id: str
    score: float | None
    rank: int


@dataclass(frozen=True, slots=True)
class VideoCandidate:
    video_id: str

    global_frame_id: str | None
    global_score: float | None
    global_rank: int | None

    event_hits: tuple[EventCandidateHit, ...]

    temporal_path: AlignedPath | None
    temporal_rank: int | None

    rerank_global_score: float | None = None
    rerank_temporal_score: float | None = None
    final_rerank_score: float | None = None

    winner_frame_id: str | None = None
```

This is internal state only. Existing HTTP contracts remain compatible.

## 11. Representative frame 1: `F_global`

`F_global` is the best frame for the complete query from global retrieval.

Purpose:

- preserve whole-query visual/semantic identity;
- protect against noisy temporal localization;
- keep simple KIS queries strong.

## 12. Representative frame 2: `F_temporal`

`F_temporal` comes from the best temporal path for that video.

Preferred selection rule:

1. if event-emission provenance for path frames is already available, choose the path frame with strongest event evidence;
2. otherwise use the existing deterministic path representative rule.

The implementation must use real interfaces available in the current code; it must not fabricate per-frame emission metadata just to implement this selection.

If `F_global == F_temporal`, score the image once.

No similarity/distance deduplication is added in P2a.

## 13. Reuse the existing reranker

For each candidate video:

```text
query + F_global   -> R_global
query + F_temporal -> R_temporal
```

Use the existing `RerankingService` and its adapter/provider boundary.

Do not create a second VLM reranker implementation.

The video-level layer is responsible only for:

- selecting frame candidates;
- invoking the existing frame reranker;
- aggregating scores;
- returning ordered videos.

## 14. Video score

If both representative scores exist:

```text
R_video = max(R_global, R_temporal)
```

If only one succeeds, use the available score.

Why `max`: Recall@20 is the target. One strong representative image should be enough to keep a correct video alive. A weak/misaligned second frame must not suppress it.

Do not average the two scores in P2a.

Do not hard-filter candidates with a reranker threshold.

## 15. Final ordering

Sort candidate videos by:

```text
1. final reranker score descending
2. global/full-query dense evidence
3. candidate-union insertion order
4. video_id deterministic tie break
```

Temporal DP score is not the semantic tie-break anchor.

## 16. KIS semantics

After video-level reranking, KIS returns the frame that caused the video to score best.

```text
if R_global >= R_temporal:
    KIS frame = F_global
else:
    KIS frame = F_temporal
```

This replaces the current behavior where KIS blindly projects a temporal path to a fixed middle representative after ranking.

The full temporal path may still be retained wherever the current result contract already exposes it.

## 17. TRAKE semantics

TRAKE may consume the new video ordering but must keep the selected video's existing temporal path unchanged:

```text
frame_ids
frame_idxs
timestamps_ms
path score
```

The reranker changes video ordering, not temporal coordinates.

KIS and TRAKE therefore share the candidate/reranking core but have different output heads:

```text
KIS   -> reranker winner frame
TRAKE -> temporal path
```

## 18. Shared service boundary

Add a focused shared service, preferably:

```text
src/hcmai/orchestration/workflows/video_reranking.py
```

Its responsibilities are:

1. global retrieval;
2. event retrieval;
3. temporal candidate collection;
4. video deduplication/union;
5. representative-frame selection;
6. existing reranker invocation;
7. video-score aggregation;
8. deterministic fallback;
9. diagnostics.

Conceptual interface:

```python
class VideoCandidateRerankingService:
    def search_and_rerank(
        self,
        query: str,
        events: Sequence[str],
        *,
        retrieval_events: Sequence[str],
        top_videos: int,
        ...
    ) -> VideoRerankingResult:
        ...
```

KIS and TRAKE remain thin projections.

## 19. Composition root

At startup:

```text
load retrieval/corpus
-> create reranking adapter/client
-> construct RerankingService once
-> construct VideoCandidateRerankingService once
-> inject into SearchService/KIS/TRAKE
```

Do not instantiate rerankers inside request handlers.

Keep the existing provider error categories from `RerankingService`.

## 20. Configuration

Add a dedicated config section, conceptually:

```yaml
video_reranking:
  enabled: false
  candidate_max_videos: 200
  representative_frames: 2

  candidate_union:
    global_target: 120
    event_target: 50
    temporal_target: 30

  aggregation: max
```

Safety contract:

```text
representative_frames = 2
aggregation = max
no hard reranker threshold
```

`enabled` remains `false` until Recall@20 evaluation passes.

## 21. Fallback semantics

Recall protection is mandatory.

### One representative frame fails

Use the surviving frame score.

### Both representative frames for one video fail

Do not drop the video. Fall back to global evidence/candidate-union ordering.

### Entire reranker request/service fails

Return the candidate-union ordering.

### Disabled flag

Return the existing baseline behavior.

P2a competition mode is fail-open for Recall.

## 22. Evaluation matrix

Use the same query preparation and artifacts for every run.

| Run | Candidate pool | Temporal | Reranker |
|---|---|---:|---:|
| R0 | current dense/KIS baseline | baseline/no | no |
| R1 | current temporal KIS | yes | no |
| R2 | global + event + temporal union | yes | no |
| R3 | R2 + one representative frame | yes | yes |
| R4 | R2 + two representative frames | yes | yes |

Interpretation:

```text
R2 - R1 = candidate-union gain
R3 - R2 = reranker gain
R4 - R3 = second-frame gain
```

No per-query tuning.

## 23. Required diagnostics

For each labeled query:

```text
query_id
gt_video_id

gt_in_global_pool
gt_global_rank

gt_in_event_pool
gt_best_event_rank

gt_in_temporal_pool
gt_temporal_rank

gt_in_union200
gt_union_rank

global_frame_id
temporal_frame_id

gt_rerank_global_score
gt_rerank_temporal_score
gt_final_rerank_score

gt_final_rank
winner_frame_source
winner_frame_id
```

Record equivalent source/rank information for the highest-ranked false positive.

## 24. Candidate gate

Before judging reranker quality:

```text
Recall@200_union >= max(Recall@200_global, Recall@200_event, Recall@200_temporal)
```

subject to the explicit 200-video cap.

If an already retrieved GT is lost, diagnostics must identify which source entry was displaced.

## 25. Reranker gate

P2a is useful only if:

```text
Recall@20_R4 > Recall@20_R2
```

on the labeled benchmark.

The gain must not come from only one isolated query.

## 26. Regression gate

Simple visual/single-scene queries that already work through global dense retrieval must not systematically leave Top-20 after P2a.

This is a high-recall reranker, not a mandatory-temporal filter.

## 27. False-positive diagnostic

P2a should reduce visually implausible fragmented-evidence false positives such as:

```text
lecture slide
+ generic person
+ OCR/ASR words about animals
```

when the full query actually describes a group scene, yellow animal, pumpkin-like object, and narrative actions.

This is a qualitative diagnostic, not a blacklist or handcrafted semantic rule.

## 28. Rollback

P2a is feature-flagged:

```yaml
video_reranking:
  enabled: false
```

Rollback requires no index rebuild and no artifact migration.

## 29. Expected source boundaries

Modify:

```text
src/hcmai/common/config.py
src/hcmai/orchestration/setup.py
src/hcmai/orchestration/pipeline.py
src/hcmai/orchestration/workflows/kis.py
src/hcmai/orchestration/workflows/trake.py
```

Potentially modify `temporal_search.py` only to expose a read-only candidate/path interface needed by the shared reranking layer.

Create a focused shared module:

```text
src/hcmai/orchestration/workflows/video_reranking.py
```

Reuse unchanged where possible:

```text
src/hcmai/retrieval/reranking/pipeline.py
src/hcmai/retrieval/reranking/adapters/*
src/hcmai/retrieval/reranking/models/contracts.py
```

## 30. Required tests

### Candidate union

Verify:

- one video occupies one slot;
- global/event/temporal overlap is deduplicated;
- event lists are merged round-robin;
- freed quotas refill;
- 200-video cap is deterministic.

### Representative frames

Verify:

- `F_global` is the full-query best frame;
- `F_temporal` is deterministic;
- exact duplicate representative is reranked once;
- missing one representative is supported.

### Aggregation

Verify `max(0.91, 0.32) = 0.91` and winner frame follows the maximum score.

### Fallback

Verify:

- one frame failure does not drop video;
- two frame failures do not drop video;
- request-level reranker failure returns union ranking;
- disabled mode preserves baseline behavior.

### KIS

Verify:

- final Top-20 is video-diverse;
- KIS returns reranker winner frame.

### TRAKE

Verify:

- video ordering may change;
- temporal path coordinates remain unchanged.

### Integration

Verify existing HTTP contracts remain compatible.

## 31. Why P2a comes before skip/confidence alignment

Current evidence suggests the more fundamental KIS failure is ranking architecture:

- correct video can already exist in candidate space;
- temporal retrieval can be useful;
- fragmented evidence can still outrank global semantic relevance.

Making DP even more flexible before fixing video ranking could worsen this.

P2a therefore targets:

```text
candidate diversity
-> global semantic anchor
-> VLM visual reranking
-> unique video Top-20
```

Only after P2a evaluation should P2b consider:

```text
optional events
confidence weighting
entity consistency
```

## 32. Final contract

```text
FULL QUERY
    |
    +---------------------------+
    |                           |
    v                           v
GLOBAL DENSE              PREPARED EVENTS
    |                     /          \
    |                    v            v
    |               EVENT SEARCH    TEMPORAL
    |                    |            |
    +--------------------+------------+
                         |
                         v
                UNIQUE VIDEO UNION
                   max 200
                         |
              +----------+----------+
              |                     |
              v                     v
           F_global              F_temporal
              |                     |
              +----------+----------+
                         |
                         v
                 EXISTING VLM
                    RERANKER
                         |
                         v
                   R_video=max
                         |
                         v
                  TOP-20 VIDEOS
                   /          \
                  v            v
                KIS          TRAKE
           winner frame    existing path
```

The P2a rollout decision is driven primarily by Recall@20.
