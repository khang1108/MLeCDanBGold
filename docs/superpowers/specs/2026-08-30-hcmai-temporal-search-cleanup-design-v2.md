# HCMAI Temporal Search Cleanup Design v2

**Status:** Frozen after architecture QA on 2026-08-30.

**Source snapshots reviewed:** `src_hcmai_v4.zip` backend and `frontend_v2.zip` frontend.

## 1. Purpose

The cleanup establishes one research baseline for ordered temporal retrieval and removes legacy workflow complexity before new paper-driven improvements are added. KIS and TRAKE share one visual event-to-frame DP core. They differ only in input shaping and output projection.

The work is deliberately split into two independently testable phases:

1. **Phase A — Temporal/KIS/TRAKE/API/frontend cleanup.**
2. **Phase B — Data/schema/Corpus architecture cleanup.**

Phase A must be stable before Phase B starts.

## 2. Global constraints

- Phase A is a breaking cleanup; do not add compatibility shims for removed request/response fields.
- Phase A does not change the visual similarity model or the current DP recurrence.
- KIS and TRAKE use canonical keyframes only; no dense frame refinement is introduced.
- KIS and TRAKE do not use Context, ASR, RRF, or Qwen reranking in the baseline.
- Existing multimodal retrieval and Qwen/VLM reranking code stays available as detached research capability.
- Caption, OCR, ASR/transcript, object, embedding, index, and keyframe-generation logic must preserve the artifact paths, formats, naming, manifests, and behavior already used by the team.
- Phase B reorganizes source ownership but does not migrate the current `artifacts/` layout.
- Runtime artifact loading must be read-only and fail fast; runtime code never auto-generates missing artifacts.

## 3. Phase A architecture

```text
KIS raw query
    |
    v
Deterministic event splitter
    |
    +-------------------------------+
                                    |
TRAKE explicit events[] ------------+
                                    v
                         visual event encoding
                                    |
                                    v
                         full-corpus frame scoring
                                    |
                                    v
                         strict monotonic DP
                                    |
                                    v
                         level-wise ranked paths
                              /             \
                             v               v
                      KIS projection     TRAKE projection
                      middle frame       full ordered path
```

### 3.1 Query semantics

KIS accepts only a raw `query` string. The backend is the source of truth for event splitting:

1. Normalize whitespace.
2. If the query contains two or more non-empty lines, each line is one event.
3. Otherwise split on sentence boundaries (`.`, `!`, `?`) when this yields at least two non-empty events.
4. Otherwise the entire query is one event.

There is no LLM parsing, query expansion, coreference resolution, entity extraction, or state tracking in Phase A.

TRAKE accepts `events: list[str]` directly and does not run the KIS splitter.

### 3.2 DP semantics

For ordered events `E_0 ... E_n` and canonical video frames `f_0 ... f_m`, the baseline chooses one strictly later frame for every successive event.

```text
position(E_0) < position(E_1) < ... < position(E_n)
```

The current scoring recurrence and gap penalty remain unchanged. Full alignment is required: every event receives exactly one canonical frame. If a video has fewer frames than events or no finite path, that video contributes no path.

The cleanup keeps:

- visual similarity only;
- strict chronological order;
- current `lambda_gap` behavior;
- current optional `event_power` and `cluster_delta` numerical behavior until a later research experiment explicitly changes them;
- multiple paths from the same video;
- the current level-wise diversity behavior in `rank_paths()`.

The cleanup removes candidate-video shortlisting from the KIS/TRAKE path. Every configured canonical visual-index frame is scored for every event, then results are split into per-video matrices before DP.

### 3.3 Internal path contract

Internal runtime code uses frozen dataclasses rather than public Pydantic models. `temporal/dp.py` owns both the private numerical decoder row and the canonical aligned path:

```python
@dataclass(frozen=True, slots=True)
class AlignedPath:
    video_id: str
    score: float
    frame_ids: tuple[str, ...]
    frame_idxs: tuple[int, ...]
    timestamps_ms: tuple[int, ...]
```

`orchestration/temporal_search.py` owns only the timed orchestration result:

```python
@dataclass(frozen=True, slots=True)
class TemporalSearchResult:
    paths: tuple[AlignedPath, ...]
    retrieval_ms: float
    alignment_ms: float
```

`DPPath` remains the private numerical decoder output in `temporal/dp.py`. `TemporalSearchService` converts ranked `DPPath` rows plus the corresponding `VideoEventScores` identity/timestamp arrays into canonical `AlignedPath` values and returns them inside `TemporalSearchResult`.

Invariant:

```text
events[i]
   <-> frame_ids[i]
   <-> frame_idxs[i]
   <-> timestamps_ms[i]
```

### 3.4 `top_k`

`top_k` has one meaning for both APIs: the maximum number of ranked DP paths returned by the shared temporal search.

- KIS projects each returned path to one representative result.
- TRAKE returns each returned path directly.

Multiple returned paths may share the same `video_id`.

### 3.5 KIS projection

For each aligned path, KIS selects the deterministic upper-middle frame:

```python
representative_index = len(path.frame_ids) // 2
```

A one-event query therefore returns its only aligned frame.

The full path is retained in the result so the frontend can inspect alignment.

### 3.6 KIS HTTP contract

```python
class SearchRequest(BaseModel):
    query: str
    top_k: int = 20


class SearchResultMetadata(BaseModel):
    title: str | None = None
    caption: str | None = None
    ocr: str | None = None
    objects: list[str] = []
    asr: str | None = None


class SearchResult(BaseModel):
    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    score: float
    frame_ids: list[str]
    timestamps_ms: list[int]
    thumbnail_urls: list[str]
    frame_url: str
    thumbnail_url: str
    metadata: SearchResultMetadata


class SearchLatency(BaseModel):
    query_ms: float
    retrieval_ms: float
    alignment_ms: float
    materialization_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    query: str
    events: list[str]
    results: list[SearchResult]
    latency: SearchLatency
```

`frame_idx` is retained deliberately as the representative frame's canonical submission coordinate because the current submission workflow needs it. It is not the internal frame identity and must never be used to derive the aligned timestamp or asset URL.

KIS result invariant:

```text
response.events[i]
    <-> result.frame_ids[i]
    <-> result.timestamps_ms[i]
    <-> result.thumbnail_urls[i]
```

`metadata` is materialized only for the representative frame. `title` is video-level data but is duplicated inside every result metadata object for frontend simplicity.

Raw DP score is exposed as `score`; it is not normalized to a percentage.

### 3.7 TRAKE HTTP contract

Request:

```python
class TRAKERequest(BaseModel):
    events: list[str]
    top_k: int = 20
```

Response path:

```python
class TRAKEPath(BaseModel):
    video_id: str
    score: float
    frame_ids: list[str]
    frame_idxs: list[int]
    timestamps_ms: list[int]
    thumbnail_urls: list[str]
```

Response:

```python
class TRAKEResponse(BaseModel):
    events: list[str]
    paths: list[TRAKEPath]
    latency: SearchLatency
```

TRAKE invariant:

```text
events[i]
   <-> path.frame_ids[i]
   <-> path.frame_idxs[i]
   <-> path.timestamps_ms[i]
   <-> path.thumbnail_urls[i]
```

A path is an independent result. The frontend must never merge multiple paths because they share a `video_id`.

### 3.8 Empty-result semantics

A valid request with no alignable path returns HTTP 200 with `results=[]` or `paths=[]`. Invalid input remains a validation error.

### 3.9 Latency

KIS and TRAKE expose the same stages:

```text
query_ms
retrieval_ms
alignment_ms
materialization_ms
total_ms
```

Fusion, rerank, scene, progressive, backfill, and time-to-first legacy fields are removed from the public contract. Fine-grained debug tracing may remain internal.

### 3.10 Phase A deletions

Delete from the default search workflow:

- progressive search/session state;
- `UNKNOWN`, `MATCHED`, `EVALUATED_NO_MATCH` evidence-state machinery;
- backfill;
- scene clustering and soft relation scoring;
- KIS single-frame reranking wiring;
- `SearchFilters` and all filter plumbing;
- `search_id`;
- generic `TaskType`, `TaskRequest`, `TaskResponse`, and task registry dispatch;
- search quotas and shortlist controls such as candidate/global/local/backfill/scene budgets;
- frontend Fusion/Rerank score UI;
- frontend progressive `sessionStorage` state;
- Suggest Query;
- query-file parsing/upload workflow.

### 3.11 Detached capability retained

The following code is not called by Phase A KIS/TRAKE but remains available for later experiments:

- Context retrieval;
- ASR retrieval;
- RRF fusion;
- caption/OCR retrieval code;
- Qwen/VLM reranking.

Removing `TaskType` and filters from common runtime interfaces may require these detached components to use one task-agnostic source-weight configuration. Equal default weights must preserve the current default behavior.

## 4. Phase A frontend

KIS cards show the representative frame, raw alignment score, timestamp, and representative metadata. Each card has an `Alignment` accordion showing ordered event, timestamp, and thumbnail rows.

```text
E1  00:12.4  [thumbnail]
E2  00:18.1  [thumbnail]
E3  00:25.7  [thumbnail]
```

TRAKE renders each backend path independently:

```text
Path #1 — V01 — Alignment score: 2.73
E1  00:12.4  [thumbnail]
E2  00:18.1  [thumbnail]
E3  00:25.7  [thumbnail]
[Submit this path]
```

The frontend does not construct frame/thumbnail routes from `frame_id`; backend-provided URLs are authoritative.

No metadata filtering is added in Phase A.

## 5. Phase B architecture

After Phase A is stable, runtime and offline data construction are separated:

```text
src/hcmai/
  api/
  corpus/
  retrieval/
  temporal/
  orchestration/

offline/
  ingestion/
  preprocessing/
  enrichment/
  embeddings/
  indexes/
  keyframes/
```

### 5.1 Corpus facade

`DataService` is replaced in one breaking migration by a small read-only `Corpus` facade. Public runtime callers import only `Corpus`, not specialist stores.

Target read API:

```python
corpus.frame(frame_id)
corpus.frames(frame_ids)
corpus.caption(frame_id)
corpus.ocr(frame_id)
corpus.objects(frame_id)
corpus.transcript(video_id, start_ms, end_ms)
corpus.image_path(frame_id)
corpus.thumbnail_path(frame_id)
```

`Corpus.open(...)` opens existing artifacts and fails immediately when required inputs are missing. It never calls ingestion, enrichment, embedding, indexing, or keyframe generation.

Private implementation classes such as `_FrameStore`, `_EvidenceStore`, `_TranscriptStore`, and `_AssetResolver` may exist behind the facade.

### 5.2 Runtime models

Runtime canonical data types use `@dataclass(frozen=True, slots=True)`. Pydantic is reserved for API/external boundaries and offline artifact-validation boundaries.

Ownership rule:

- API contracts -> `src/hcmai/api/contracts/`
- runtime corpus types -> `src/hcmai/corpus/models.py`
- retrieval internal types -> retrieval-owned modules/dataclasses
- temporal `AlignedPath` -> temporal-owned module
- evaluation types -> evaluation-owned module
- observability types -> observability-owned module

`src/hcmai/common/schemas/` is deleted after all ownership migrations complete.

### 5.3 Offline separation

Any code that creates, writes, publishes, or mutates data artifacts belongs under `offline/`. Code that only reads artifacts for runtime search belongs under `src/hcmai/corpus/` or `src/hcmai/retrieval/`.

The existing standalone CLIs remain separate. Do not create a new unified CLI in this cleanup.

`offline/` is a Python package with `__init__.py` files and should avoid importing runtime `hcmai` domain dataclasses. Runtime/offline communication is primarily through the existing artifact formats.

### 5.4 Index ownership

Corpus owns canonical data/evidence reads. Retrieval owns encoder/index load/search. Runtime index loaders remain under `src/hcmai/retrieval/`; index-building and artifact-writing entry points move to `offline/indexes/` or `offline/embeddings/`.

### 5.5 Keyframe extractor

The complete C++ keyframe extractor tree lives at:

```text
offline/keyframes/keyframes_extraction/
```

without changing extraction logic, file naming, publication behavior, or tests except for path/build-system references.

### 5.6 Artifact compatibility

The cleanup preserves the current layout, including:

```text
artifacts/
  enrichment/
    captions/
    context/
    objects/
    ocr/
    transcripts/
  frame_store/
  indexes/
```

Do not introduce a new canonical corpus directory, immutable snapshot layout, renamed artifact, or changed manifest in Phase B.

## 6. Final dependency direction

```text
API
 |
 v
Orchestration
 |        \
 v         v
Temporal  Retrieval
    \      /
     v    v
      Corpus
```

Offline code is outside the runtime dependency graph:

```text
Offline pipeline -> existing artifacts -> Corpus / Retrieval
```

Forbidden directions include `Corpus -> API`, `Corpus -> offline`, `Temporal -> FastAPI contracts`, and `offline -> runtime orchestration`.

## 7. Deferred research

Only after both cleanup phases are stable should research branches test:

- multimodal event/frame evidence;
- entity/coreference continuity;
- state transitions;
- visual continuity features;
- richer transition scoring;
- candidate-video pruning as a measured speed/recall tradeoff;
- dense temporal refinement;
- VLM path reranking or verification.
