# Search orchestration

`hcmai.orchestration` owns the online application flow. FastAPI routers call
the explicit KIS or TRAKE method on `SearchService`; routers do not wire
retrieval, alignment, or materialization internals themselves.

```text
orchestration/
├── pipeline.py              # Explicit SearchService public facade
├── temporal_search.py       # Shared scoring and aligned-path materialization
├── workflows/               # Thin KIS and TRAKE projections
├── setup.py                 # Single application composition root
└── materializer.py          # Representative evidence and asset URLs
```

## Runtime path

```text
FastAPI (:8000)
    → SearchService
    → explicit KISPipeline or TRAKEPipeline
    → RemoteTemporalSearchService + RemoteImageSearchService (gRPC to :8002)
    → Corpus-backed canonical materialization
```

### Progressive multimodal KIS retrieval flow

```text
Query Composer
  -> InitialResolve | PatchEvents | GlobalRewrite | SearchOnly
  -> KISIntent
  -> KISRetrievalPlan
  -> text/image temporal evidence
  -> existing fusion
  -> DP
  -> ranked results
```

Note: EventTrail is the next design cycle, not part of S0/S1. KIS semantic interactions are stateless per-request operations against explicit base intents.

Both workflows consume one stateless ordered event-to-frame alignment service.
KIS handles progressive multimodal event graphs (`KISIntent`) with immutable
revisions, while TRAKE accepts ordered events directly and returns every ranked
path independently, including multiple paths from the same video.

`setup.py` loads configuration and artifacts once, constructs the selected
services, and injects them into `SearchService`. Request handling does not
create or modify offline artifacts.

Canonical identity is immutable. `TemporalSearchService` validates every
retrieval-owned `frame_id`, `video_id`, `frame_idx`, and timestamp against
`Corpus` before exposing an `AlignedPath`. Workflow projection preserves
those parallel arrays, and the backend supplies all frame and thumbnail URLs.

KIS and TRAKE return the same public latency stages: `query_ms`,
`retrieval_ms`, `alignment_ms`, `materialization_ms`, and `total_ms`. Scores
are raw DP scores. Context/ASR retrieval, reciprocal-rank fusion, and VLM
reranking remain detached research capabilities and are not called by these
Phase A workflows.

## Verification

```bash
PYTHONPATH=src python -m pytest tests/orchestration tests/api -q
```
