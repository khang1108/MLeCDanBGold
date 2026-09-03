# Hybrid Dense + BM25 Temporal Retrieval and Query Preparation Design

**Status:** Approved architecture draft for implementation planning  
**Date:** 2026-09-02  
**Baseline:** `src_hcmai_v6.zip` + `frontend_v4.zip`  
**Supersedes:** the unfinished metadata Filter-page direction. The Filter workflow is removed rather than extended.

## 1. Goal

Extend the cleaned KIS/TRAKE temporal-search baseline with two independently selectable evidence families:

1. **Dense Retrieval** — Visual SigLIP2 + Context BGE + ASR BGE.
2. **BM25 lexical retrieval** — fielded lexical evidence over canonical-frame `title`, `caption`, `ocr`, and `asr` text.

Both families produce full-corpus event-to-frame scores, are normalized and fused before the existing monotonic DP decoder, and are controlled by two UI toggles. Local Qwen3-4B query preparation generates optional English retrieval candidates; BM25 searches the Vietnamese corpus directly.

The temporal decoder itself is not redesigned in this feature.

## 2. Current Baseline

The backend currently routes KIS and TRAKE through the shared `TemporalSearchService` in `src/hcmai/orchestration/temporal_search.py`. `RetrievalService.score_event_videos()` in `src/hcmai/retrieval/retriever/pipeline.py` deliberately uses only the visual retriever and scores the full visual index before `rank_paths()`.

The existing generic retrieval stack already knows how to compose Visual, Context, and segment-ASR retrievers behind RRF, but that RRF stack is detached from temporal DP. This design keeps it detached. RRF remains useful for ordinary ranked retrieval experiments, but it is not the score representation consumed by DP.

The frontend currently owns one Query workspace in `src/features/search/components/SearchWorkspace.jsx`, a Top-K sidebar in `src/features/search-controls/components/ToolBox.jsx`, and a separate Filter workspace. The Filter workspace and `/api/v1/filter` placeholder are removed by this feature.

## 3. Non-goals

This feature does **not**:

- change strict monotonic DP ordering, full-event alignment, gap penalty, event power, path diversity, or representative-frame semantics;
- add candidate-video pruning or sparse Top-K gating before DP;
- add query sessions, candidate IDs, or server-side search state;
- add a learned cross-modal calibration model;
- use RRF scores as DP matrix values;
- add object labels as a BM25 field in the initial baseline;
- replace the existing generic RRF retrieval capability or reranker package;
- silently fall back to a weaker retrieval mode when a user-selected required capability is unavailable.

## 4. User-visible Retrieval Modes

The Query workspace exposes exactly two independent toggles:

```text
[✓] Dense Retrieval
[✓] BM25
```

The semantics are:

| Dense | BM25 | Result |
|---|---|---|
| ON | OFF | Dense-only temporal evidence |
| OFF | ON | BM25-only temporal evidence |
| ON | ON | Hybrid Dense + BM25 temporal evidence |
| OFF | OFF | Retrieve action disabled and backend request rejected |

These controls apply identically to KIS and TRAKE.

`top_k` keeps its current meaning: the maximum number of ranked DP paths returned. KIS projects each returned path to its representative frame; TRAKE exposes the ordered path.

## 5. Query Representations

The user input is Vietnamese. Every request preserves two conceptual representations:

- **original events (`q_vi`)** — the user's Vietnamese events after the existing deterministic event planning/normalization;
- **retrieval events (`q_r`)** — the exact event strings selected by the user for Dense retrieval.

When the user retrieves the original query:

```text
q_r = q_vi
```

When the user selects one generated English candidate bundle:

```text
q_r = selected_candidate_en
```

The original Vietnamese events are never discarded because BM25 title/OCR/ASR must continue to use Vietnamese lexical terms.

## 6. Local Query Preparation

### 6.1 Model ownership

A local **Qwen3-4B** deployment performs query preparation. It runs on the user's Thundercompute A6000/L40 inference environment, not inside each FastAPI web worker. The web/runtime code depends on a narrow query-preparation adapter; it does not import Transformers or allocate Qwen weights directly.

This prevents duplicated model memory across API workers and keeps model replacement independent of retrieval/DP code.

### 6.2 Operations

`QueryPreparationService` exposes two distinct operations:

```python
translate_literal(events_vi: Sequence[str]) -> tuple[str, ...]
generate_candidates(events_vi: Sequence[str]) -> QueryCandidateSet
```

`translate_literal()` remains available for query-preparation compatibility, but online BM25 search does not call it. Translation and rewriting are useful only when producing optional Dense retrieval candidates.

`generate_candidates()` is an explicit user action. One model request produces:

- one literal English event bundle;
- exactly five English retrieval candidate bundles.

The five candidates are controlled paraphrases, not unconstrained enrichment. The prompt must preserve entities, colors, numbers, quantities, actions, unknown placeholders such as `X`, and event order. It must not infer facts absent from the Vietnamese input.

### 6.3 Event-boundary invariant

For `N` original events:

```text
len(literal_en) == N
len(candidates) == 5
len(candidate_k.events) == N for every k
```

and position `i` always means the same event:

```text
original_events[i]
      ↕
literal_en[i]
      ↕
candidate_k.events[i]
```

Qwen may rephrase an event but may not merge, split, drop, duplicate, or reorder events.

Malformed model output is rejected. Candidate generation may retry the structured-generation request once; after that it returns an explicit error rather than silently repairing event semantics.

### 6.4 Caching

Query-preparation results use a bounded process-local cache keyed by:

```text
(operation, normalized_original_events, model_name, model_revision, prompt_version)
```

The cache avoids repeated Qwen calls when the same query is searched again with different Top-K or Dense/BM25 toggles. Candidate text itself is still sent in the later search request, so search execution remains stateless and reproducible from the concrete request payload.

Editing the original query invalidates the candidate set in the frontend.

## 7. Stateless Public API

### 7.1 Query candidate generation

Add:

```text
POST /api/v1/query-candidates
```

The request accepts exactly one input form:

```python
class QueryCandidateRequest(BaseModel):
    query: str | None = None
    events: list[str] | None = None
```

Exactly one of `query` or `events` must be provided.

- KIS sends `query`; the backend uses the same deterministic planner as search before calling Qwen.
- TRAKE sends explicit `events`; their boundaries are preserved exactly.

Response:

```python
class QueryCandidate(BaseModel):
    index: int                 # 1..5, display-only
    events: list[str]

class QueryCandidateResponse(BaseModel):
    original_events: list[str]
    literal_en: list[str]
    candidates: list[QueryCandidate]  # exactly five
```

`index` is a presentation ordinal only. It is never a server-side lookup key and is not accepted by `/search` or `/trake`.

### 7.2 KIS request

Extend `SearchRequest` to:

```python
class SearchRequest(BaseModel):
    query: str
    retrieval_events: list[str] | None = None
    use_dense: bool = True
    use_bm25: bool = True
    top_k: int = 20
```

Semantics:

- `query` is always the original Vietnamese KIS input.
- backend planner produces `original_events` from `query`;
- absent `retrieval_events` means "retrieve the original";
- supplied `retrieval_events` are the selected English candidate bundle and must have the same event count as `original_events`;
- at least one of `use_dense` or `use_bm25` must be true.

### 7.3 TRAKE request

Extend `TRAKERequest` analogously:

```python
class TRAKERequest(BaseModel):
    events: list[str]                   # original Vietnamese events
    retrieval_events: list[str] | None = None
    use_dense: bool = True
    use_bm25: bool = True
    top_k: int = 20
```

If `retrieval_events` is supplied, its length must equal `events`.

### 7.4 Response observability

KIS and TRAKE responses retain their current result/path contracts and expose the concrete query representation used by each evidence family:

```python
dense_events: list[str] | None
bm25_caption_events: list[str] | None
use_dense: bool
use_bm25: bool
```

`events` continues to mean the original canonical event sequence. `dense_events` is the selected retrieval bundle when Dense is enabled. `bm25_caption_events` reports the original Vietnamese events used for caption BM25. Candidate selection never changes this BM25 representation.

No search/candidate session ID is introduced.

## 8. Query Routing Rules

For each event position `i`:

### 8.1 Dense ON

Dense uses `retrieval_events[i]`.

- Original selected: the Vietnamese event goes directly to Dense.
- English candidate selected: that English event goes directly to Dense.

Dense query preparation never performs an implicit translation.

### 8.2 BM25 ON

BM25 sends the original Vietnamese event to every lexical field:

```text
title_vi   ← original_events[i]
caption_vi ← original_events[i]
ocr_vi     ← original_events[i]
asr_vi     ← original_events[i]
```

Therefore:

- Dense-only + Original does not invoke Qwen.
- BM25 never invokes Qwen and can run without query preparation.
- Candidate retrieval uses the candidate only for Dense and keeps original Vietnamese for BM25.

## 9. BM25 Corpus and Artifact Contract

### 9.1 Document unit

The canonical frame is the only BM25 document unit.

Each lexical document contains:

```text
frame_id
video_id
frame_idx
timestamp_ms
title
caption
ocr
asr
```

Field sources are:

- `title`: organizer/video metadata duplicated onto every canonical frame in that video;
- `caption`: existing English frame caption artifact;
- `ocr`: existing Vietnamese frame OCR artifact;
- `asr`: existing frame-aligned ASR enrichment for the canonical frame;
- missing evidence becomes an empty field, not a fabricated value.

No runtime translation of corpus fields occurs.

### 9.2 Offline build

The BM25 artifact is built offline under the repository's offline index-building boundary. It consumes existing canonical/enrichment artifacts and does not change their formats or paths.

The published BM25 bundle must contain enough identity metadata to validate every document against `Corpus`:

```text
mapping: document_position ↔ frame_id/video_id/frame_idx/timestamp_ms
field indexes/statistics: title/caption/ocr/asr
metadata: dataset version, tokenizer version, schema version, build metadata
```

At runtime, the BM25 mapping is reordered once into the visual/canonical frame order used by temporal scoring. Identity mismatch is a startup error, not a query-time best effort.

### 9.3 Tokenization baseline

The first lexical baseline is deterministic and lightweight:

- Unicode normalization;
- lowercase;
- punctuation boundary cleanup while preserving alphanumeric tokens;
- whitespace/word tokenization;
- no stemming;
- no corpus translation;
- no Vietnamese word-segmentation model.

More advanced Vietnamese tokenization is a later benchmarkable experiment.

### 9.4 Fielded BM25 semantics

The baseline is a **fielded BM25 scorer** implemented as independent field BM25 scores with configurable weights:

```text
B_raw(i, j) =
    w_title   * BM25_title(original_vi_i, frame_j.title)
  + w_caption * BM25_caption(original_vi_i, frame_j.caption)
  + w_ocr     * BM25_ocr(original_vi_i, frame_j.ocr)
  + w_asr     * BM25_asr(original_vi_i, frame_j.asr)
```

This design intentionally calls the component "fielded BM25" rather than claiming a strict native BM25F implementation. Native BM25F can be benchmarked later without changing the temporal interface.

Default field weights are equal and configurable.

The scorer exposes full-corpus values, not only nearest/top-k hits:

```python
score_events(...) -> np.ndarray  # [event_count, canonical_frame_count]
```

## 10. Dense Temporal Evidence

### 10.1 Dense sources

Dense temporal evidence consists of:

1. Visual SigLIP2 frame index.
2. Context BGE frame index.
3. Segment-native ASR BGE scores projected onto canonical frames.

Temporal Dense scoring uses complete-ASR segment scores and projects them onto canonical frame rows before alignment; it does not rely on Top-K segment retrieval because DP requires a complete frame score row.

`IndexConfig.asr_segment_path` is the only ASR Dense artifact. `asr_projection_max_gap_ms` bounds whether a segment score may contribute to a frame. No frame-native ASR Dense artifact is created or consumed.

### 10.2 Shared encoding

For one request:

- selected retrieval events are encoded once by SigLIP2 for Visual scores;
- selected retrieval events are encoded once by BGE and the same text query batch scores both Context and segment-ASR indexes.

The implementation must not load duplicate BGE/SigLIP model instances for the hybrid scorer.

### 10.3 Dense source fusion

For each event, every source scores the full canonical frame corpus. Each source row is min-max normalized independently:

```text
normalize(x) = (x - min(x)) / (max(x) - min(x))
```

If `max(x) == min(x)`, the normalized row is all zeros.

The Dense block is then:

```text
D(i,j) =
    wd_visual  * V_norm(i,j)
  + wd_context * C_norm(i,j)
  + wd_asr     * A_norm(i,j)
```

Default Dense source weights are equal (`1/3` each) and configurable. They must sum to `1.0` after validation.

For the baseline defined by this spec, selecting Dense requires all three dense source indexes. If one is unavailable, Dense capability is reported unavailable rather than silently changing the meaning of "Dense".

## 11. Dense + BM25 Hybrid Fusion

BM25 field fusion first produces `B_raw`; each event row is then min-max normalized to `B_norm`.

The final event-to-frame evidence is:

```text
if Dense only:
    M = D

if BM25 only:
    M = B_norm

if Dense + BM25:
    M = w_dense * D + w_bm25 * B_norm
```

Default hybrid weights:

```text
w_dense = 0.5
w_bm25  = 0.5
```

They are configurable and must sum to `1.0` when both modes are active.

No RRF, rank conversion, or Top-K candidate sparsification occurs in this temporal scoring path.

## 12. Temporal Boundary

Introduce one task-agnostic temporal evidence boundary between orchestration and retrieval, conceptually:

```python
class TemporalEvidenceScorer:
    def score_events(
        self,
        original_events: Sequence[str],
        retrieval_events: Sequence[str],
        *,
        use_dense: bool,
        use_bm25: bool,
    ) -> list[VideoEventScores]: ...
```

`TemporalSearchService` calls this scorer and then continues to call the existing `rank_paths()` decoder.

The following DP behavior remains frozen:

- strict monotonic ordering;
- one frame per event;
- no skipped events;
- existing `lambda_gap`;
- existing `event_power`;
- existing `cluster_delta` / level-wise path ranking semantics;
- canonical keyframes only;
- multiple paths from one video remain legal.

This boundary is the main ablation seam: changes to Dense/BM25 evidence do not require changes to `hcmai/temporal/dp.py`.

## 13. Backend Package Ownership

Target ownership:

```text
src/hcmai/
├── api/
│   ├── contracts/
│   │   ├── query_candidates.py
│   │   ├── search.py
│   │   └── trake.py
│   └── routers/
│       ├── query_candidates.py
│       ├── search.py
│       └── trake.py
│
├── query_preparation/
│   ├── models.py
│   ├── service.py
│   ├── cache.py
│   └── adapters/
│       └── qwen.py
│
├── retrieval/
│   └── evidence/
│       ├── dense.py
│       ├── bm25.py
│       ├── hybrid.py
│       └── normalization.py
│
├── orchestration/
│   └── temporal_search.py
│
└── temporal/
    └── dp.py        # decoder behavior unchanged
```

The existing `retrieval/retriever/fusion/rrf.py`, generic retrievers, and reranking package remain detached capabilities.

The BM25 offline builder belongs under the existing offline index-building package and publishes only artifacts consumed by runtime `retrieval/evidence/bm25.py`.

## 14. Configuration

Add explicit configuration blocks rather than reusing RRF weights:

```yaml
query_preparation:
  model_name: Qwen/Qwen3-4B
  model_revision: <pinned revision>
  prompt_version: query-prep-v1
  candidate_count: 5
  cache_enabled: true
  cache_ttl_seconds: 3600
  cache_max_entries: 2048

hybrid_temporal:
  dense:
    visual_weight: 0.3333333333
    context_weight: 0.3333333333
    asr_weight: 0.3333333334
  bm25:
    title_weight: 0.25
    caption_weight: 0.25
    ocr_weight: 0.25
    asr_weight: 0.25
  dense_weight: 0.5
  bm25_weight: 0.5
```

The Qwen model revision must be pinned in deployment configuration before benchmark/freeze. Candidate count is fixed to five by the public product design even though it is represented in config for validation/observability.

Existing RRF `FusionConfig` remains owned by generic retrieval and does not configure temporal hybrid scoring.

## 15. Health and Failure Semantics

Health reporting exposes independent readiness:

```text
query_preparation
bm25
visual_dense
context_dense
asr_dense
dense_temporal
hybrid_temporal
```

Rules:

- Dense toggle is disabled in UI if `dense_temporal` is unavailable.
- BM25 toggle is disabled if the BM25 artifact is unavailable.
- Generate Candidates is disabled if query preparation is unavailable.
- Original Dense-only search can still run when Qwen is unavailable.
- Original and candidate-backed BM25 search can run when query preparation is unavailable.
- A frontend-supplied English candidate changes Dense retrieval only.
- No selected evidence source is silently omitted.

## 16. Frontend Design

### 16.1 Remove Filter workflow

Delete the separate Filter navigation/page and its client API. The Query workspace becomes the only retrieval workspace.

This removes the current `FilterWorkspace` integration from `src/App.jsx`, the filter feature components, filter API wrapper/tests, and filter-specific styles that have no remaining consumer.

### 16.2 Existing query form

The existing primary Search action becomes **Retrieve Original**. Keyboard Enter for normal KIS input keeps the same fast path.

Add an explicit secondary action:

```text
[Generate Candidates]
```

Generating candidates does not run search.

### 16.3 Candidate panel

After generation, render:

```text
Original (Vietnamese)
E1: ...
E2: ...
[Retrieve Original]

Literal English
E1: ...
E2: ...

Candidate 1
E1: ...
E2: ...
[Retrieve]

...

Candidate 5
E1: ...
E2: ...
[Retrieve]
```

`literal_en` is displayed for debugging but is not a sixth generated candidate. The product has the original retrieval hypothesis plus five generated candidate hypotheses.

For a single-event KIS query, each bundle simply renders one line.

### 16.4 Retrieval toggles

Extend `ToolBox` with Dense and BM25 switches above Top-K. Both default ON. Turning both OFF disables all Retrieve buttons.

Reset Parameters restores:

```text
Dense = ON
BM25 = ON
Top-K = 20
```

### 16.5 Candidate selection request

Candidate cards never send `candidate_id`.

- Original button sends no `retrieval_events`.
- Candidate button sends that card's concrete English `events` array.

KIS/TRAKE mode detection remains based on the original input. Selecting an English candidate does not change whether the request is KIS or TRAKE.

### 16.6 State invalidation

Changing the original query immediately clears:

- literal translation;
- five candidates;
- candidate-generation errors.

Current search results may remain until the user retrieves again, but the UI must not allow a stale candidate generated from the previous original query to be submitted.

## 17. Latency and Observability

Keep existing top-level latency fields for compatibility, with `retrieval_ms` covering query preparation required by the actual retrieval request plus Dense/BM25 scoring/fusion.

Candidate generation is a separate endpoint and reports its own simple latency value:

```python
query_preparation_ms: float
```

Internal tracing should separately record:

```text
qwen_translation_ms
qwen_candidate_generation_ms
visual_scoring_ms
context_scoring_ms
asr_scoring_ms
bm25_scoring_ms
normalization_fusion_ms
alignment_ms
```

These diagnostic stages do not need to become the public KIS/TRAKE response schema.

## 18. Testing Strategy

### 18.1 Query preparation

Tests must prove:

- literal translation preserves event count/order;
- candidate generation returns exactly five bundles;
- every bundle preserves the original event count;
- malformed structured model output is rejected;
- placeholder `X`, numbers, colors, and named tokens survive the adapter contract;
- cache keys separate model/prompt revisions and translation vs candidate generation.

### 18.2 BM25

Use a tiny canonical corpus containing Vietnamese evidence to prove:

- Vietnamese query terms match title/caption/OCR/ASR fields;
- generated English candidates do not replace the original BM25 query;
- missing fields contribute zero;
- field weights are applied deterministically;
- loaded BM25 identity maps exactly to canonical frames;
- score output is full-corpus and ordered consistently with temporal frame identity.

### 18.3 Dense evidence

Tests must prove:

- selected retrieval events are encoded once per encoder family;
- Visual, Context, and frame-ASR all score the full frame corpus;
- source normalization is per event;
- constant rows normalize to zeros;
- Dense source weights sum to one and produce expected synthetic scores;
- missing required Dense source makes Dense capability unavailable rather than silently dropping the source.

### 18.4 Hybrid fusion

Synthetic matrices must verify all three modes:

```text
Dense only -> D
BM25 only -> B_norm
Both -> 0.5*D + 0.5*B_norm by default
```

and backend validation rejects both toggles OFF.

### 18.5 Temporal regression

Feed controlled hybrid matrices into the existing DP tests and verify that decoder behavior remains unchanged: monotonic ordering, gap penalty, full alignment, multiple same-video paths, and level-wise ranking.

### 18.6 API

Contract tests cover:

- query-candidate raw KIS input;
- query-candidate explicit TRAKE events;
- original KIS search;
- candidate KIS search;
- original TRAKE search;
- candidate TRAKE search;
- event-count mismatch rejection;
- both toggles OFF rejection;
- query-preparation unavailable failure only when the selected path actually needs it.

### 18.7 Frontend

Tests cover:

- Filter navigation no longer exists;
- Dense/BM25 toggles default ON and cannot produce an enabled Retrieve action when both are OFF;
- Generate Candidates performs only candidate generation;
- exactly five candidate cards are rendered;
- candidate events preserve display order;
- editing original query invalidates candidates;
- Original retrieval omits `retrieval_events`;
- Candidate #k sends concrete `retrieval_events`, never an ID;
- same behavior works for KIS and TRAKE;
- existing KIS alignment, TRAKE path, and submission tests remain green.

## 19. Benchmark Matrix After Implementation

The feature should make the following ablations available without code changes:

```text
A. Dense original VI
B. BM25 original VI over Vietnamese fields
C. Hybrid original
D. Dense candidate #k
E. BM25 candidate #k
F. Hybrid candidate #k
```

Dense source and BM25 field weights are configuration variables, not UI controls in the first release. This keeps the user-facing experiment surface to the two intended toggles while preserving research flexibility in configuration.

## 20. Acceptance Criteria

The feature is ready to freeze when:

1. Filter UI/API is removed.
2. Qwen3-4B query preparation runs through the local Thundercompute inference boundary.
3. Generate Candidates returns one literal translation plus exactly five event-aligned candidates.
4. KIS and TRAKE requests are stateless and accept concrete selected retrieval events.
5. Dense mode uses Visual + Context + frame-ASR full-corpus scoring.
6. BM25 uses canonical-frame title/caption/OCR/ASR fielded scoring with the original Vietnamese events.
7. BM25 search never requires or invokes query preparation.
8. Candidate retrieval uses the selected candidate only for Dense and retains original Vietnamese for all BM25 fields.
9. Per-event normalization and configurable Dense/BM25 fusion feed the unchanged DP decoder.
10. UI supports Original + five generated retrieval hypotheses and independent Dense/BM25 toggles.
11. Automated backend/frontend tests cover mode routing, event invariants, fusion math, API contracts, and unchanged DP semantics.
12. No candidate/session state is stored server-side.
