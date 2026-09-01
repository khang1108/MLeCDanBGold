# HCMAI Phase A Temporal Search Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the KIS/TRAKE runtime with one stateless full-corpus visual DP baseline and simplify its API/frontend while preserving detached multimodal and reranking research capabilities.

**Architecture:** KIS raw text is deterministically split into ordered events; TRAKE supplies ordered events directly. Both call one `TemporalSearchService`, which scores every canonical visual-index frame, decodes strict monotonic paths with the current DP recurrence, and returns task-agnostic `AlignedPath` dataclasses. KIS projects each path to its middle frame; TRAKE returns the path itself.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, NumPy, existing FAISS/SigLIP retrieval stack, pytest; React 19, react-scripts, Testing Library/Jest.

**Spec:** `docs/superpowers/specs/2026-08-30-hcmai-temporal-search-cleanup-design-v2.md`

## Global Constraints

- Use `src_hcmai_v4.zip` as the backend source snapshot and `frontend_v2.zip` as the frontend source snapshot.
- This phase is a breaking cleanup; do not add compatibility shims for `search_id`, filters, task dispatch, old scores, or old latency fields.
- Preserve the current visual encoder/index artifact format and canonical frame identity.
- Do not change the DP recurrence, strict chronological semantics, `lambda_gap`, `event_power`, `cluster_delta`, or level-wise path ranking behavior in this phase.
- KIS and TRAKE must score the full canonical visual corpus; no video shortlist or filter restriction may gate the DP input.
- Keep Context/ASR/RRF and Qwen/VLM reranking code present but detached from KIS/TRAKE.
- Do not change caption/OCR/ASR/object/keyframe/index artifact paths, formats, manifests, or generation logic.
- Return HTTP 200 with an empty result list for a valid query that produces no aligned path.
- Backend paths in Tasks 1-9 and 14 are relative to the backend repository root. Frontend paths in Tasks 10-13 are relative to the `frontend_v2` repository root. Do not mix the two `src/` trees.

---

## File Structure Locked by This Plan

Backend files created or made authoritative in Phase A:

```text
src/hcmai/api/contracts/
  __init__.py
  latency.py          # shared public latency contract
  search.py           # KIS request/response contracts
  trake.py            # TRAKE request/response contracts

src/hcmai/orchestration/
  temporal_search.py  # shared event scoring + validation + DP facade

src/hcmai/temporal/
  planner.py          # deterministic KIS query splitter only
  dp.py               # pure DP + internal AlignedPath
```

Frontend files created:

```text
src/features/alignment/components/AlignmentAccordion.jsx
src/features/alignment/components/AlignmentAccordion.test.jsx
src/features/search/components/TrakePathCard.jsx
src/features/search/components/TrakePathCard.test.jsx
```

The existing backend `src/hcmai/common/schemas/` remains for unrelated contracts until Phase B, but Phase A search/TRAKE HTTP contracts must no longer live there.

---

### Task 1: Freeze Current DP Semantics with Characterization Tests

**Files:**

- Create: `tests/temporal/test_dp.py`
- Modify: `src/hcmai/temporal/dp.py`

**Interfaces:**

- Consumes: existing `VideoEventScores` and `align_video(...)` / `rank_paths(...)` behavior.
- Produces: tests that lock strict ordering, full alignment, gap penalty, multiple paths per video, one-event behavior, and level-wise diversity.

- [X] **Step 1: Add a helper that builds deterministic score matrices**

```python
import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores


def video_scores(video_id: str, scores: list[list[float]]) -> VideoEventScores:
    matrix = np.asarray(scores, dtype=np.float32)
    n_frames = matrix.shape[1]
    return VideoEventScores(
        video_id=video_id,
        frame_ids=np.asarray([f"{video_id}-f{i}" for i in range(n_frames)]),
        frame_idx=np.arange(n_frames, dtype=np.int64),
        timestamps_ms=np.arange(n_frames, dtype=np.int64) * 1_000,
        scores=matrix,
    )
```

- [X] **Step 2: Add strict-order and full-alignment tests**

```python
def test_align_video_is_strictly_increasing_and_full():
    video = video_scores(
        "v1",
        [
            [9.0, 1.0, 0.0, 0.0],
            [0.0, 8.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 7.0],
        ],
    )
    path = align_video(video, lambda_gap=0.0, paths=1)[0]
    assert path.frame_idx == (0, 1, 3)
    assert len(path.frame_ids) == 3


def test_align_video_returns_no_partial_path_when_video_has_too_few_frames():
    video = video_scores("v1", [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assert align_video(video, lambda_gap=0.0, paths=1) == []
```

- [X] **Step 3: Add gap-penalty, one-event, and multiple-path tests**

```python
def test_gap_penalty_prefers_shorter_chronological_path():
    video = video_scores(
        "v1",
        [
            [5.0, 4.9, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0],
        ],
    )
    path = align_video(video, lambda_gap=1e-3, paths=1)[0]
    assert path.frame_idx == (1, 2)


def test_one_event_uses_same_decoder():
    video = video_scores("v1", [[0.1, 0.8, 0.2]])
    path = align_video(video, lambda_gap=0.0, paths=1)[0]
    assert path.frame_idx == (1,)


def test_align_video_can_return_multiple_paths_from_same_video():
    video = video_scores("v1", [[5.0, 4.0, 0.0], [0.0, 4.0, 5.0]])
    paths = align_video(video, lambda_gap=0.0, paths=2)
    assert len(paths) == 2
    assert all(path.video_id == "v1" for path in paths)
```

- [X] **Step 4: Lock the current level-wise ranking behavior**

```python
def test_rank_paths_takes_first_level_across_videos_before_second_level():
    v1 = video_scores("v1", [[10.0, 9.0, 0.0], [0.0, 9.0, 10.0]])
    v2 = video_scores("v2", [[8.0, 0.0], [0.0, 8.0]])
    rows = rank_paths([v1, v2], lambda_gap=0.0, max_rows=2)
    assert [row.video_id for row in rows] == ["v1", "v2"]
```

- [X] **Step 5: Run the characterization tests**

Run:

```bash
PYTHONPATH=src pytest tests/temporal/test_dp.py -v
```

Expected: all tests pass against the existing recurrence. If a numeric fixture is wrong, adjust only fixture numbers until it characterizes current behavior; do not modify the recurrence to satisfy a preferred behavior.

- [X] **Step 6: Commit the baseline characterization**

```bash
git add tests/temporal/test_dp.py src/hcmai/temporal/dp.py
git commit -m "test: characterize temporal dp baseline"
```

---

### Task 2: Replace Video Shortlisting with Full-Corpus Visual Event Scoring

**Files:**

- Modify: `src/hcmai/retrieval/retriever/video_scores.py`
- Modify: `src/hcmai/retrieval/retriever/pipeline.py`
- Create: `tests/retrieval/test_video_scores.py`

**Interfaces:**

- Produces: `score_all_videos(index, query_vectors, chunk_size=65_536) -> list[VideoEventScores]`.
- Produces: `RetrievalService.score_event_videos(events, *, chunk_size=65_536)` with no filters, shortlist depth, RRF budget, or max-video limit.

- [ ] **Step 1: Write a fake visual index test proving every video is scored**

```python
class FakeIndex:
    def __init__(self):
        self.frame_ids = np.asarray(["a0", "a1", "b0", "b1", "c0"])
        self.frame_idx = np.asarray([0, 1, 0, 1, 0])
        self.timestamps = np.asarray([0, 1000, 0, 1000, 0])
        self.video_ids = np.asarray(["a", "a", "b", "b", "c"])

    def video_positions(self, video_id):
        return np.flatnonzero(self.video_ids == video_id)

    def score_subset(self, query_vectors, positions, chunk_size):
        assert positions.tolist() == [0, 1, 2, 3, 4]
        return np.asarray([[1, 2, 3, 4, 5]], dtype=np.float32)


def test_score_all_videos_scores_every_index_position():
    rows = score_all_videos(FakeIndex(), np.asarray([[1.0]], dtype=np.float32))
    assert [row.video_id for row in rows] == ["a", "b", "c"]
    assert [row.frame_ids.tolist() for row in rows] == [
        ["a0", "a1"],
        ["b0", "b1"],
        ["c0"],
    ]
```

- [ ] **Step 2: Run the test and verify the new function is missing**

Run:

```bash
PYTHONPATH=src pytest tests/retrieval/test_video_scores.py -v
```

Expected: FAIL because `score_all_videos` does not exist.

- [ ] **Step 3: Implement full-corpus scoring without nearest-neighbor shortlist voting**

Replace `score_videos(...)` with logic equivalent to:

```python
def score_all_videos(index, query_vectors, chunk_size: int = 65_536):
    positions = np.arange(len(index.frame_ids), dtype=np.int64)
    if len(positions) == 0:
        return []

    scores = index.score_subset(query_vectors, positions, chunk_size)
    video_ids = sorted({str(video_id) for video_id in index.video_ids})
    return [
        VideoEventScores(
            video_id=video_id,
            frame_ids=index.frame_ids[window],
            frame_idx=index.frame_idx[window],
            timestamps_ms=index.timestamps[window],
            scores=scores[:, window],
        )
        for video_id in video_ids
        if len(window := index.video_positions(video_id))
    ]
```

Do not call `search_filtered`, compute RRF votes, compute coverage, or apply `max_videos`.

- [ ] **Step 4: Simplify `RetrievalService.score_event_videos`**

Target signature:

```python
def score_event_videos(
    self,
    events: Sequence[str],
    *,
    chunk_size: int = 65_536,
) -> list[VideoEventScores]:
```

The body must continue selecting `self._retriever_for("visual")`, encode all events as one batch, and call `score_all_videos`.

- [ ] **Step 5: Run retrieval and DP tests**

```bash
PYTHONPATH=src pytest tests/retrieval/test_video_scores.py tests/temporal/test_dp.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit full-corpus scoring**

```bash
git add src/hcmai/retrieval/retriever/video_scores.py src/hcmai/retrieval/retriever/pipeline.py tests/retrieval/test_video_scores.py
git commit -m "refactor: score temporal events over full visual corpus"
```

---

### Task 3: Make the Temporal Core Task-Agnostic and Dataclass-Based

**Files:**

- Modify: `src/hcmai/temporal/dp.py`
- Modify: `src/hcmai/temporal/planner.py`
- Create: `src/hcmai/orchestration/temporal_search.py`
- Modify: `src/hcmai/temporal/__init__.py`
- Delete after references are gone: `src/hcmai/temporal/service.py`
- Delete after references are gone: `src/hcmai/common/schemas/alignment.py`
- Create: `tests/temporal/test_planner.py`
- Create: `tests/orchestration/test_temporal_search.py`

**Interfaces:**

- Produces: `split_query_events(query: str) -> tuple[str, ...]`.
- Produces: frozen `AlignedPath` dataclass in `temporal/dp.py`.
- Produces: `TemporalSearchService.search(events: Sequence[str], *, top_k: int) -> tuple[AlignedPath, ...]`.

- [ ] **Step 1: Write deterministic splitter tests**

```python
def test_multiline_query_prefers_lines():
    assert split_query_events("hold ingredient\nroll ingredient\ncoat flour") == (
        "hold ingredient",
        "roll ingredient",
        "coat flour",
    )


def test_single_line_query_splits_sentences():
    assert split_query_events("Hold ingredient. Roll ingredient! Coat flour?") == (
        "Hold ingredient",
        "Roll ingredient",
        "Coat flour",
    )


def test_single_event_stays_single():
    assert split_query_events("chef holds a bowl") == ("chef holds a bowl",)
```

- [ ] **Step 2: Replace `build_alignment_plan` with `split_query_events`**

`planner.py` must no longer import `AlignmentEvent`, `AlignmentPlan`, or `SearchFilters`. It returns normalized strings only.

- [ ] **Step 3: Keep `DPPath` private and add canonical `AlignedPath` in `temporal/dp.py`; add only the timed result in `orchestration/temporal_search.py`**

Leave the existing numerical decoder contract in `temporal/dp.py`:

```python
@dataclass(frozen=True, slots=True)
class DPPath:
    video_id: str
    score: float
    frame_idx: tuple[int, ...]
    frame_ids: tuple[str, ...]
```

Add the canonical runtime path beside the decoder in `temporal/dp.py`:

```python
@dataclass(frozen=True, slots=True)
class AlignedPath:
    video_id: str
    score: float
    frame_ids: tuple[str, ...]
    frame_idxs: tuple[int, ...]
    timestamps_ms: tuple[int, ...]
```

Add only the timed orchestration wrapper to `orchestration/temporal_search.py`:

```python
@dataclass(frozen=True, slots=True)
class TemporalSearchResult:
    paths: tuple[AlignedPath, ...]
    retrieval_ms: float
    alignment_ms: float
```

`DPPath` is never exposed to KIS/TRAKE HTTP projection code. Do not introduce a Pydantic internal path type.

- [ ] **Step 4: Write a fake-service test that validates canonical identity and returns paths**

Use a fake retrieval result whose `frame_id`, `frame_idx`, and `timestamp_ms` match fake `DataService.get_frame` records. Assert:

```python
search = service.search(["e1", "e2"], top_k=2)
assert search.paths[0].frame_ids == ("v1-f0", "v1-f1")
assert search.paths[0].frame_idxs == (0, 1)
assert search.paths[0].timestamps_ms == (0, 1000)
assert search.retrieval_ms >= 0
assert search.alignment_ms >= 0
```

Add a second test where index `frame_idx` disagrees with canonical data and assert `ValueError`.

- [ ] **Step 5: Implement `TemporalSearchService`**

Constructor:

```python
class TemporalSearchService:
    def __init__(self, data: DataService, retrieval: RetrievalService, config: AlignmentConfig):
        self.data = data
        self.retrieval = retrieval
        self.config = config
```

Search method:

```python
def search(self, events: Sequence[str], *, top_k: int) -> TemporalSearchResult:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    normalized = tuple(" ".join(event.split()) for event in events if event.strip())
    if not normalized:
        raise ValueError("events must not be empty")

    retrieval_started = perf_counter()
    scores = self.retrieval.score_event_videos(
        normalized,
        chunk_size=self.config.chunk_size,
    )
    retrieval_ms = (perf_counter() - retrieval_started) * 1_000

    # Validate every score-matrix column against canonical DataService identity
    # before decoding, then retain a video_id -> VideoEventScores lookup for
    # canonical frame_idx/timestamp materialization.
    score_by_video = {video.video_id: video for video in scores}

    alignment_started = perf_counter()
    rows = rank_paths(
        scores,
        lambda_gap=self.config.lambda_gap,
        max_rows=top_k,
        event_power=self.config.event_power,
        cluster_delta=self.config.cluster_delta,
    )
    paths = tuple(
        _materialize_aligned_path(row, score_by_video[row.video_id])
        for row in rows
    )
    alignment_ms = (perf_counter() - alignment_started) * 1_000
    return TemporalSearchResult(
        paths=paths,
        retrieval_ms=retrieval_ms,
        alignment_ms=alignment_ms,
    )
```

`_materialize_aligned_path` resolves each `DPPath.frame_id` position against the matching `VideoEventScores` arrays and checks `frame_id`, canonical `frame_idx`, and canonical `timestamp_ms` against `DataService.get_frame(frame_id)`. Any mismatch raises `ValueError`; no identity is recomputed from filenames or list position.

- [ ] **Step 6: Run temporal tests**

```bash
PYTHONPATH=src pytest tests/temporal tests/orchestration/test_temporal_search.py -v
```

Expected: PASS.

- [ ] **Step 7: Delete old alignment service/schema only after import search is empty**

Run:

```bash
rg -n "TemporalAlignmentService|AlignmentPlan|AlignmentEvent|common.schemas.alignment" src tests
```

Expected before deletion: only the files being replaced. After migration: no runtime references.

- [ ] **Step 8: Commit the shared temporal core**

```bash
git add src/hcmai/temporal src/hcmai/orchestration/temporal_search.py tests/temporal tests/orchestration/test_temporal_search.py
git rm src/hcmai/temporal/service.py src/hcmai/common/schemas/alignment.py
git commit -m "refactor: expose task agnostic temporal search core"
```

---

### Task 4: Create Thin KIS and TRAKE API Contracts

**Files:**

- Create: `src/hcmai/api/contracts/__init__.py`
- Create: `src/hcmai/api/contracts/latency.py`
- Create: `src/hcmai/api/contracts/search.py`
- Create: `src/hcmai/api/contracts/trake.py`
- Create: `tests/api/test_contracts.py`
- Delete after migration: `src/hcmai/common/schemas/search.py`
- Delete after migration: `src/hcmai/common/schemas/trake.py`

**Interfaces:**

- Produces: `SearchRequest`, `SearchResultMetadata`, `SearchResult`, `SearchResponse`, `SearchLatency`, `TRAKERequest`, `TRAKEPath`, `TRAKEResponse`.

- [ ] **Step 1: Write request contract tests proving removed legacy fields are rejected**

```python
def test_kis_request_has_only_query_and_top_k():
    request = SearchRequest(query="chef cooks", top_k=20)
    assert request.query == "chef cooks"
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "x", "search_id": "legacy"})
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "x", "filters": {"video_ids": ["v1"]}})


def test_trake_requires_explicit_events():
    request = TRAKERequest(events=["e1", "e2"], top_k=5)
    assert request.events == ["e1", "e2"]
```

Use `ConfigDict(extra="forbid")` on these public models so removed fields fail instead of being silently ignored.

- [ ] **Step 2: Implement shared latency contract**

```python
class SearchLatency(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_ms: float = Field(default=0, ge=0)
    retrieval_ms: float = Field(default=0, ge=0)
    alignment_ms: float = Field(default=0, ge=0)
    materialization_ms: float = Field(default=0, ge=0)
    total_ms: float = Field(default=0, ge=0)
```

- [ ] **Step 3: Implement KIS contracts with alignment-array invariant**

`SearchResult` includes representative `frame_idx: int` for submission and must validate:

```python
if not (
    len(self.frame_ids)
    == len(self.timestamps_ms)
    == len(self.thumbnail_urls)
):
    raise ValueError("alignment arrays must have equal lengths")
```

`SearchResponse` must validate every result path length equals `len(events)`.

Use `Field(default_factory=list)` for `objects`; never use a mutable list literal as a Pydantic default in implementation.

- [ ] **Step 4: Implement TRAKE contracts with full path invariant**

Validate:

```python
len(frame_ids) == len(frame_idxs) == len(timestamps_ms) == len(thumbnail_urls)
```

and every path length equals `len(response.events)`.

- [ ] **Step 5: Run contract tests**

```bash
PYTHONPATH=src pytest tests/api/test_contracts.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit API contracts**

```bash
git add src/hcmai/api/contracts tests/api/test_contracts.py
git commit -m "refactor: define thin kis and trake api contracts"
```

---

### Task 5: Rebuild KIS as a Thin Projection of `AlignedPath`

**Files:**

- Modify: `src/hcmai/orchestration/workflows/kis.py`
- Modify: `src/hcmai/orchestration/materializer.py`
- Create: `tests/orchestration/test_kis_pipeline.py`

**Interfaces:**

- Consumes: `split_query_events(query)` and `TemporalSearchService.search(events, top_k=...) -> TemporalSearchResult`.
- Produces: `SearchResponse` with representative middle frame, full path arrays, representative metadata, backend-owned asset URLs, and new latency fields.

- [ ] **Step 1: Write a pipeline test for middle-frame projection and path invariants**

Use a fake path with five frames and assert:

```python
response = pipeline.execute(SearchRequest(query="e1\ne2\ne3\ne4\ne5", top_k=1))
result = response.results[0]
assert response.events == ["e1", "e2", "e3", "e4", "e5"]
assert result.frame_id == "f2"
assert result.frame_idx == 2
assert result.frame_ids == ["f0", "f1", "f2", "f3", "f4"]
assert result.timestamps_ms == [0, 1000, 2000, 3000, 4000]
assert len(result.thumbnail_urls) == 5
assert result.score == pytest.approx(2.73)
```

- [ ] **Step 2: Write metadata materialization assertions**

Representative metadata test must cover:

```python
assert result.metadata.title == "Cooking Episode"
assert result.metadata.caption == "chef coats ingredient"
assert result.metadata.ocr == "FLOUR"
assert result.metadata.objects == ["bowl", "person"]
assert result.metadata.asr == "coat it with flour"
```

Materialize representative metadata from the already loaded stores without running retrieval:

```python
caption = data.get_evidence(frame.frame_id, RetrievalSource.CAPTION)
ocr = data.get_evidence(frame.frame_id, RetrievalSource.OCR)
counts = data.get_object_counts(frame.frame_id) or {}
objects = sorted(counts)
video_meta = (
    data.video_metadata_store.get(frame.video_id)
    if data.video_metadata_store is not None
    else None
)
title = video_meta.title if video_meta is not None else None
segments = data.get_transcript_segments_at_time(frame.video_id, frame.timestamp_ms)
asr = " ".join(segment.text.strip() for segment in segments if segment.text.strip()) or None
```

This exact transcript-at-representative-timestamp rule is the Phase A ASR metadata baseline; do not fall back to reranking text or fusion candidates.

- [ ] **Step 3: Replace candidate-based materialization**

Replace the old candidate/rank materializer API with exactly:

```python
def build_kis_result(self, path: AlignedPath) -> SearchResult:
    representative = len(path.frame_ids) // 2
    frame_id = path.frame_ids[representative]
    frame = self.data.get_frame(frame_id)
    # assert canonical organizer coordinate retained for submission
    if frame.frame_idx != path.frame_idxs[representative]:
        raise ValueError("aligned frame_idx disagrees with canonical frame")
    ...
```

The method returns one `SearchResult` and performs representative metadata/URL materialization. Delete `build_response`, `build_result(candidate, rank)`, and `_build_scores`; KIS workflow constructs `SearchResponse` from the list of returned results. Do not convert the path into `RetrievalCandidate` or `SearchScores`.

Build URLs with the existing frame route convention in the backend:

```python
encoded = quote(frame_id, safe="")
thumbnail_url = f"/api/v1/frames/{encoded}/thumbnail"
frame_url = f"/api/v1/frames/{encoded}/image"
```

The frontend must not reconstruct these URLs.

- [ ] **Step 4: Replace old KIS tracing fields with the shared latency contract**

Measure separately around:

```text
split_query_events -> query_ms
score_event_videos -> retrieval_ms
rank_paths -> alignment_ms
response/metadata/URL construction -> materialization_ms
whole request -> total_ms
```

`TemporalSearchService.search` returns `TemporalSearchResult`; copy `search.retrieval_ms` and `search.alignment_ms` directly into the public `SearchLatency`. KIS measures only splitter time (`query_ms`), result/metadata/URL construction (`materialization_ms`), and wall-clock `total_ms` outside the shared service. Do not rerun retrieval or DP for timing.

- [ ] **Step 5: Remove KIS dependencies on `TaskRequest`, `TaskType`, `SearchFilters`, `search_id`, `RetrievalCandidate`, `SearchScores`, and public `RetrievalTrace`**

Run:

```bash
rg -n "TaskType|TaskRequest|SearchFilters|search_id|RetrievalCandidate|SearchScores|RetrievalTrace" src/hcmai/orchestration/workflows/kis.py src/hcmai/orchestration/materializer.py
```

Expected: no matches except any explicitly internal observability helper that does not enter the API.

- [ ] **Step 6: Run KIS tests**

```bash
PYTHONPATH=src pytest tests/orchestration/test_kis_pipeline.py tests/api/test_contracts.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit KIS projection**

```bash
git add src/hcmai/orchestration/workflows/kis.py src/hcmai/orchestration/materializer.py tests/orchestration/test_kis_pipeline.py
git commit -m "refactor: project kis results from temporal paths"
```

---

### Task 6: Rebuild TRAKE as an Independent Path List

**Files:**

- Modify: `src/hcmai/orchestration/workflows/trake.py`
- Create: `tests/orchestration/test_trake_pipeline.py`

**Interfaces:**

- Consumes: explicit `TRAKERequest.events` and `TemporalSearchService`.
- Produces: `TRAKEResponse.paths`, one output element per ranked DP path, including raw score and backend thumbnail URLs.

- [ ] **Step 1: Write a test with two paths from the same video**

```python
def test_trake_keeps_same_video_paths_independent():
    response = pipeline.execute(TRAKERequest(events=["e1", "e2"], top_k=2))
    assert len(response.paths) == 2
    assert [path.video_id for path in response.paths] == ["v1", "v1"]
    assert response.paths[0].frame_ids != response.paths[1].frame_ids
```

- [ ] **Step 2: Assert full ordered arrays and raw scores**

```python
path = response.paths[0]
assert path.score == pytest.approx(2.41)
assert len(path.frame_ids) == len(response.events)
assert len(path.frame_idxs) == len(response.events)
assert len(path.timestamps_ms) == len(response.events)
assert len(path.thumbnail_urls) == len(response.events)
```

- [ ] **Step 3: Rewrite `TRAKEPipeline.execute`**

The request no longer contains `query`, `query_type`, warnings, rank, or request ID. Do not hash a synthetic query. Pass `request.events` directly to the shared temporal service.

- [ ] **Step 4: Return HTTP-domain paths rather than `TRAKESubmission` rows**

Each `AlignedPath` maps directly to one `TRAKEPath`. Use the existing canonical `frame_idxs` supplied by the temporal service and build `thumbnail_urls` from `frame_ids`.

- [ ] **Step 5: Run TRAKE and shared temporal tests**

```bash
PYTHONPATH=src pytest tests/orchestration/test_trake_pipeline.py tests/orchestration/test_temporal_search.py tests/temporal -v
```

Expected: PASS.

- [ ] **Step 6: Commit TRAKE projection**

```bash
git add src/hcmai/orchestration/workflows/trake.py tests/orchestration/test_trake_pipeline.py
git commit -m "refactor: return independent trake alignment paths"
```

---

### Task 7: Remove Generic Task Dispatch and Wire Explicit KIS/TRAKE Methods

**Files:**

- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/api/routers/search.py`
- Modify: `src/hcmai/api/routers/trake.py`
- Modify: `src/hcmai/api/routers/system.py`
- Delete: `src/hcmai/orchestration/task_router.py`
- Delete: `src/hcmai/common/schemas/task.py`
- Create: `tests/api/test_search_routes.py`
- Create: `tests/api/test_trake_routes.py`
- Create: `tests/orchestration/test_service_wiring.py`

**Interfaces:**

- Produces: `SearchService.search_kis(request: SearchRequest) -> SearchResponse`.
- Produces: `SearchService.search_trake(request: TRAKERequest) -> TRAKEResponse`.
- Keeps: frame lookup, neighbor, health, and submission helpers on the service until Phase B.

- [ ] **Step 1: Write wiring test proving KIS and TRAKE share one temporal service**

```python
service = SearchService(data=fake_data, retrieval=fake_retrieval, config=config)
assert service.kis.temporal is service.trake.temporal
```

Also assert there is no `pipeline_registry` attribute.

- [ ] **Step 2: Replace registry dispatch with explicit methods**

Target shape:

```python
class SearchService:
    def search_kis(self, request: SearchRequest) -> SearchResponse:
        return self.kis.execute(request)

    def search_trake(self, request: TRAKERequest) -> TRAKEResponse:
        return self.trake.execute(request)
```

Build one `TemporalSearchService` in `_default` setup and inject the same object into both workflows.

- [ ] **Step 3: Point routers at explicit service methods**

`POST /api/v1/search` calls `service.search_kis`.

`POST /api/v1/trake` calls `service.search_trake`.

Do not catch `SearchPipelineUnavailableError`; that abstraction no longer exists. Keep 503 for missing runtime dependencies and normal FastAPI/Pydantic 422 validation.

- [ ] **Step 4: Simplify health capability reporting**

Replace TaskType registry checks with booleans derived from loaded dependencies:

```python
search_ready = self.data is not None and self.retrieval is not None
capabilities = {
    "search": search_ready,
    "kis": search_ready,
    "trake": search_ready,
    ...
}
```

- [ ] **Step 5: Add route tests**

KIS test:

```python
response = client.post("/api/v1/search", json={"query": "chef cooks", "top_k": 3})
assert response.status_code == 200
assert "query_type" not in response.json()
assert "search_id" not in response.json()
```

TRAKE test:

```python
response = client.post("/api/v1/trake", json={"events": ["e1", "e2"], "top_k": 3})
assert response.status_code == 200
assert "paths" in response.json()
assert "submissions" not in response.json()
```

- [ ] **Step 6: Run API and wiring tests**

```bash
PYTHONPATH=src pytest tests/api tests/orchestration/test_service_wiring.py -v
```

Expected: PASS.

- [ ] **Step 7: Remove task router and task schema**

```bash
git rm src/hcmai/orchestration/task_router.py src/hcmai/common/schemas/task.py
```

Then run:

```bash
rg -n "TaskType|TaskRequest|TaskResponse|PipelineRegistry|task_router" src/hcmai tests
```

Any remaining `TaskType` match must be handled in Task 8 before Phase A closes.

- [ ] **Step 8: Commit explicit routing**

```bash
git add src/hcmai/orchestration src/hcmai/api tests/api tests/orchestration/test_service_wiring.py
git commit -m "refactor: route kis and trake explicitly"
```

---

### Task 8: Delete Filters and TaskType from Detached Retrieval Interfaces Without Deleting RRF

**Files:**

- Modify: `src/hcmai/common/config.py`
- Modify: `src/hcmai/common/schemas/enum.py`
- Modify: `src/hcmai/retrieval/retriever/models/contracts.py`
- Modify: `src/hcmai/retrieval/retriever/pipeline.py`
- Modify: `src/hcmai/retrieval/retriever/fusion/rrf.py`
- Modify: `src/hcmai/retrieval/retriever/concurrent.py`
- Modify: `src/hcmai/retrieval/retriever/dense/retriever.py`
- Modify: `src/hcmai/retrieval/retriever/text/retriever.py`
- Modify: `src/hcmai/retrieval/retriever/segment/retriever.py`
- Modify or delete if unused: `src/hcmai/retrieval/retriever/filtered.py`
- Modify: `src/hcmai/data/stores/frame.py`
- Modify: `src/hcmai/data/pipeline.py`
- Create: `tests/retrieval/test_rrf_task_agnostic.py`
- Create: `tests/config/test_search_config.py`

**Interfaces:**

- Produces: generic retrieval calls with `query`, `top_k` only; no `SearchFilters` or `TaskType`.
- Produces: `FusionConfig.source_weights: dict[RetrievalSource, float]` replacing `task_weights`.
- Keeps: RRF, Context, ASR, caption/OCR retrieval implementations available for experiments.

- [ ] **Step 1: Write a config test for task-agnostic equal source weights**

```python
def test_fusion_config_has_source_weights_not_task_weights():
    config = FusionConfig()
    assert set(config.source_weights) == set(RetrievalSource)
    assert all(weight == 1.0 for weight in config.source_weights.values())
    assert not hasattr(config, "task_weights")
```

- [ ] **Step 2: Rewrite `FusionConfig`**

Replace:

```python
task_weights: dict[TaskType, dict[RetrievalSource, float]]
```

with:

```python
source_weights: dict[RetrievalSource, float] = Field(
    default_factory=lambda: {source: 1.0 for source in FUSION_SOURCES}
)
```

Keep `required_sources`, `rrf_k`, worker count, and active-weight normalization.

- [ ] **Step 3: Remove `query_type` from RRF and retriever protocols**

Target signatures:

```python
def search(self, query: str, top_k: int = 100) -> RetrievalResult: ...
def search_batch(self, queries: list[str], top_k: int = 100) -> list[RetrievalResult]: ...
```

`RRFFusionRetriever._active_weights(active_sources)` reads `self.config.source_weights`.

- [ ] **Step 4: Remove filter arguments from all runtime retriever call chains**

Remove `filters` from `Retriever`, `VectorRetriever`, modality jobs, dense/text/segment search methods, and `RetrievalService.search/search_batch`.

Delete the public filtering surface completely from runtime search:

- delete `DenseIndex.search_filtered(...)` and `DenseIndex.filtered_positions(...)`;
- delete `SegmentDenseIndex.search_filtered(...)` and `SegmentDenseIndex.filtered_positions(...)`;
- update dense/segment retrievers to call ordinary `index.search(...)`;
- delete `DataService.filter_frame_ids(...)` and `FrameStore.filter_frame_ids(...)`;
- remove the now-unused `exact_subset_search` import and delete `src/hcmai/retrieval/retriever/filtered.py`.

Before `git rm`, run `rg -n "exact_subset_search|search_filtered|filtered_positions|filter_frame_ids" src tests` and migrate every remaining runtime call to unfiltered full-corpus/global search. No offline or evaluation interface may keep the deleted `SearchFilters` domain type.

- [ ] **Step 5: Delete `SearchFilters` and `TaskType` definitions**

After search/TRAKE contracts have moved, delete `SearchFilters` with the old `common/schemas/search.py` file and remove `TaskType` from `common/schemas/enum.py`.

- [ ] **Step 6: Write and run an RRF smoke test**

The test uses two fake retrievers with equal default weights and asserts candidate order matches equal-weight RRF. It calls:

```python
result = fusion.search("query", top_k=10)
```

with no task/filter argument.

Run:

```bash
PYTHONPATH=src pytest tests/retrieval/test_rrf_task_agnostic.py tests/config/test_search_config.py -v
```

Expected: PASS.

- [ ] **Step 7: Prove removed types are gone from runtime**

```bash
rg -n "SearchFilters|TaskType|TaskRequest|TaskResponse|search_id" src/hcmai tests
```

Expected: no runtime/test matches except historical documentation that is explicitly being updated.

- [ ] **Step 8: Commit task/filter cleanup without deleting RRF/reranking packages**

```bash
git add src/hcmai/common src/hcmai/retrieval src/hcmai/data tests/retrieval tests/config
git commit -m "refactor: remove task and filter coupling from retrieval"
```

---

### Task 9: Reduce Temporal Search Configuration and Detach Reranker Wiring

**Files:**

- Modify: `src/hcmai/common/config.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: configuration YAML files that currently set removed alignment shortlist fields
- Create: `tests/orchestration/test_default_dependencies.py`

**Interfaces:**

- Produces: KIS/TRAKE default runtime with visual retrieval + temporal search only.
- Keeps: `src/hcmai/retrieval/reranking/` package source present and importable for explicit experiments.

- [ ] **Step 1: Write a test asserting default KIS/TRAKE do not require a reranker**

```python
service = load_search_service(messages=[])
assert not hasattr(service.kis, "reranking")
assert not hasattr(service.trake, "reranking")
```

Use dependency fakes if loading production FAISS/LLM assets is unsuitable for unit tests.

- [ ] **Step 2: Remove shortlist-only fields from `AlignmentConfig`**

Delete:

```text
top_k
max_videos
rrf_k
```

Keep only fields consumed by the current full-corpus scorer/decoder, currently:

```python
class AlignmentConfig(BaseModel):
    lambda_gap: float = Field(default=1e-5, ge=0.0)
    event_power: float = Field(default=1.0, gt=0.0, le=1.0)
    chunk_size: int = Field(default=65_536, ge=1)
    cluster_delta: float = Field(default=0.0, ge=0.0)
```

- [ ] **Step 3: Remove default reranker construction/wiring**

Do not instantiate or inject `RerankingService` for KIS/TRAKE. Do not delete `src/hcmai/retrieval/reranking/`.

- [ ] **Step 4: Remove stale config keys from checked-in YAML**

Run:

```bash
rg -n "candidate_|rerank_count|global_quota|local_quota|backfill|scene_|top_m_evidence|max_videos|cluster_delta|event_power|alignment:" configs . --glob '*.yaml' --glob '*.yml'
```

For the baseline config, preserve current `lambda_gap`, `event_power`, `chunk_size`, and `cluster_delta` values exactly; remove only keys no longer represented by the config model.

- [ ] **Step 5: Run config and dependency tests**

```bash
PYTHONPATH=src pytest tests/config tests/orchestration/test_default_dependencies.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit configuration cleanup**

```bash
git add src/hcmai/common/config.py src/hcmai/orchestration/setup.py configs tests/config tests/orchestration/test_default_dependencies.py
git commit -m "refactor: minimize temporal baseline configuration"
```

---

### Task 10: Update Frontend API Client to the New Contracts

**Frontend working root:** the extracted `frontend_v2.zip` `frontend/` directory.

**Files:**

- Modify: `src/api/search.js`
- Modify: `src/api/search.test.js`

**Interfaces:**

- Produces: `searchFrames({query, topK, signal})` and `searchTrake({events, topK, signal})`.
- Removes: query type, search ID, asset URL reconstruction, Suggest Query, and query-file upload helpers.

- [ ] **Step 1: Replace KIS client contract test**

Assert the request body is exactly:

```javascript
expect(fetch).toHaveBeenCalledWith(
  expect.stringContaining('/api/v1/search'),
  expect.objectContaining({
    body: JSON.stringify({ query: 'chef cooks', top_k: 20 }),
  }),
);
```

Assert returned `frame_url`, `thumbnail_url`, and `thumbnail_urls` are preserved from the backend rather than regenerated.

- [ ] **Step 2: Replace TRAKE client contract test**

Request body must be:

```javascript
{
  events: ['e1', 'e2'],
  top_k: 20,
}
```

Validate `payload.paths`, `payload.events`, and `payload.latency`.

- [ ] **Step 3: Simplify `src/api/search.js`**

Delete exports:

```text
frameAssetUrl
materializeFrameAssets
suggestQueries
uploadQueryFiles
```

unless another non-search feature still imports `frameAssetUrl`; if so, move that generic URL helper into `src/api/client.js` but do not use it to overwrite search response URLs.

- [ ] **Step 4: Run API client tests**

```bash
CI=true npm test -- --runInBand src/api/search.test.js
```

Expected: PASS.

- [ ] **Step 5: Commit frontend client migration**

```bash
git add src/api/search.js src/api/search.test.js
git commit -m "refactor: use temporal search api contracts"
```

---

### Task 11: Render KIS Representative Metadata and Alignment Accordion

**Files:**

- Create: `src/features/alignment/components/AlignmentAccordion.jsx`
- Create: `src/features/alignment/components/AlignmentAccordion.test.jsx`
- Modify: `src/features/frames/components/FrameCard.jsx`
- Modify: `src/features/frames/components/FrameCard.test.jsx`
- Modify: `src/features/frames/components/FrameMetadata.jsx`
- Modify: `src/features/frames/components/FrameMetadata.test.jsx`
- Delete after references are gone: `src/features/frames/components/ScoreBreakdown.jsx`
- Modify: `src/styles/frames-grid.css`
- Modify: `src/styles/frame-interactions.css`

**Interfaces:**

- Consumes: KIS result `score`, representative `metadata`, `frame_ids`, `timestamps_ms`, `thumbnail_urls`, and top-level `events`.
- Produces: representative card + optional alignment sequence; no percentage normalization.

- [ ] **Step 1: Write `AlignmentAccordion` test**

```javascript
render(
  <AlignmentAccordion
    events={['hold', 'roll']}
    frameIds={['f1', 'f2']}
    timestampsMs={[1200, 2400]}
    thumbnailUrls={['/t/f1', '/t/f2']}
  />,
);
fireEvent.click(screen.getByRole('button', { name: /alignment/i }));
expect(screen.getByText('hold')).toBeTruthy();
expect(screen.getByText('00:01.200')).toBeTruthy();
expect(screen.getByAltText(/f1/i).getAttribute('src')).toBe('/t/f1');
```

Implement one shared timestamp formatter that displays `HH:MM:SS.mmm` or `MM:SS.mmm` consistently; test the chosen format.

- [ ] **Step 2: Update `FrameCard` score semantics**

Replace:

```javascript
Math.round(frame.scores.final * 100) + '%'
```

with raw DP display:

```javascript
`Alignment score: ${frame.score.toFixed(3)}`
```

Delete the score breakdown tooltip.

- [ ] **Step 3: Render representative metadata**

Use:

```javascript
frame.metadata?.title
frame.metadata?.caption
frame.metadata?.ocr
frame.metadata?.objects
frame.metadata?.asr
```

Do not add metadata filter controls.

- [ ] **Step 4: Pass top-level `events` to every KIS card**

Update `FramesBox`/`SearchWorkspace` props so each card receives the same response `events`, then render `AlignmentAccordion` inside the card or inspector.

- [ ] **Step 5: Remove Fusion/Rerank UI**

Delete `ScoreBreakdown.jsx` only after:

```bash
rg -n "ScoreBreakdown" src
```

returns no imports.

- [ ] **Step 6: Run KIS component tests**

```bash
CI=true npm test -- --runInBand \
  src/features/alignment/components/AlignmentAccordion.test.jsx \
  src/features/frames/components/FrameCard.test.jsx \
  src/features/frames/components/FrameMetadata.test.jsx
```

Expected: PASS.

- [ ] **Step 7: Commit KIS alignment UI**

```bash
git add src/features/alignment src/features/frames src/styles
git commit -m "feat: inspect kis temporal alignment paths"
```

---

### Task 12: Render TRAKE Paths Independently and Submit Exactly One Path

**Files:**

- Create: `src/features/search/components/TrakePathCard.jsx`
- Create: `src/features/search/components/TrakePathCard.test.jsx`
- Modify: `src/features/search/components/SearchWorkspace.jsx`
- Modify: `src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `src/styles/frames-grid.css`

**Interfaces:**

- Consumes: `TRAKEResponse.events` and `TRAKEResponse.paths`.
- Produces: one card per backend path, ordered event/timestamp/thumbnail rows, one submit action per path.

- [ ] **Step 1: Replace the existing grouping test with a same-video independence test**

```javascript
const paths = [
  {
    video_id: 'V01', score: 3.0,
    frame_ids: ['a1', 'a2'], frame_idxs: [10, 20],
    timestamps_ms: [1000, 2000], thumbnail_urls: ['/a1', '/a2'],
  },
  {
    video_id: 'V01', score: 2.8,
    frame_ids: ['b1', 'b2'], frame_idxs: [30, 40],
    timestamps_ms: [3000, 4000], thumbnail_urls: ['/b1', '/b2'],
  },
];
render(<TrakeResults events={['e1', 'e2']} paths={paths} ... />);
expect(screen.getAllByText(/V01/)).toHaveLength(2);
```

- [ ] **Step 2: Delete `groupTrakeFramesByVideo` and `materializeTrakeFrames`**

Backend paths are already materialized. Do not sort path frames independently of event order.

- [ ] **Step 3: Implement `TrakePathCard`**

Render path order as received. Show:

```text
Video ID
Alignment score: raw float
E1 + event text + timestamp + thumbnail
E2 + event text + timestamp + thumbnail
Submit this path
```

- [ ] **Step 4: Build submission line from only the selected path**

```javascript
const frames = path.frame_idxs.join(',');
requestSubmission({
  line: `${displayVideoId(path.video_id)},${frames}`,
  source: 'TRAKE path',
});
```

Do not combine paths sharing a video.

- [ ] **Step 5: Run TRAKE workspace tests**

```bash
CI=true npm test -- --runInBand \
  src/features/search/components/TrakePathCard.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx
```

Expected: PASS.

- [ ] **Step 6: Commit TRAKE path UI**

```bash
git add src/features/search src/styles/frames-grid.css
git commit -m "fix: preserve independent trake alignment paths"
```

---

### Task 13: Remove Progressive Session State, Suggest Query, and Query-File Parsing UI

**Files:**

- Modify: `src/features/search/components/SearchWorkspace.jsx`
- Modify: `src/features/search/components/SearchWorkspace.test.jsx`
- Delete: `src/features/search-controls/components/QuerySuggestionsPanel.jsx`
- Delete: `src/features/search-controls/components/QuerySuggestionsPanel.test.jsx`
- Modify: `src/features/submission/components/SubmissionWorktree.jsx`
- Modify: `src/features/submission/components/SubmissionWorktree.test.jsx`
- Modify: `src/api/search.js`
- Delete if now unused: `src/styles/query-suggestions.css`

**Interfaces:**

- Produces: stateless KIS/TRAKE search UI and retained submission-file editing/download behavior.
- Removes: search session fingerprints, search IDs, Suggest Query, and query-file upload/parsing.

- [ ] **Step 1: Delete progressive session constants and helpers**

Remove:

```text
SESSION_FINGERPRINT_KEY
SEARCH_ID_PREFIX
PROGRESSIVE_TASKS
sessionFingerprint
progressiveSearchIdKey
```

`handleNewSearch` now clears only component state and aborts an active request.

- [ ] **Step 2: Remove Suggest Query state and UI**

Delete `suggestions`, `isSuggesting`, `suggestError`, `handleSuggestQuery`, `handleSelectSuggestion`, Suggest Query button, and `QuerySuggestionsPanel` rendering/import.

- [ ] **Step 3: Remove query-file upload/parsing from `SubmissionWorktree`**

The worktree keeps already-loaded/local submission file editing and ZIP download, but does not offer `Upload Query Files` or `Select Folder` controls that call query-file parsing.

Replace those controls with one local **New CSV** action. It creates an empty editable submission file entirely in frontend state, named `submission.csv`; if that name already exists, choose the first free `submission-2.csv`, `submission-3.csv`, and so on. The new file starts with empty text content, becomes the active worktree file, and uses the existing editor/download path. It must not upload or call any backend query-parser endpoint.

- [ ] **Step 4: Update tests to assert removed controls are absent**

```javascript
expect(screen.queryByRole('button', { name: /suggest query/i })).toBeNull();
expect(screen.queryByRole('button', { name: /upload query files/i })).toBeNull();
```

Add a submission-worktree test:

```javascript
fireEvent.click(screen.getByRole('button', { name: /new csv/i }));
expect(screen.getByText('submission.csv')).toBeInTheDocument();
fireEvent.click(screen.getByRole('button', { name: /new csv/i }));
expect(screen.getByText('submission-2.csv')).toBeInTheDocument();
```

- [ ] **Step 5: Run frontend search/submission tests**

```bash
CI=true npm test -- --runInBand \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/features/submission/components/SubmissionWorktree.test.jsx \
  src/api/search.test.js
```

Expected: PASS.

- [ ] **Step 6: Commit removed legacy frontend workflows**

```bash
git add src/features/search src/features/submission src/api/search.js src/styles
git rm src/features/search-controls/components/QuerySuggestionsPanel.jsx src/features/search-controls/components/QuerySuggestionsPanel.test.jsx
git commit -m "refactor: remove progressive and query helper ui"
```

---

### Task 14: Complete Phase A Regression, Dead-Code, and Contract Verification

**Files:**

- Modify: `src/hcmai/temporal/README.md` (backend repo)
- Modify: `src/features/docs/components/ApiDocsModal.jsx` (frontend repo)
- Delete: stale tracked backend `__pycache__/` and `.pyc` files

**Interfaces:**

- Produces: a clean Phase A baseline ready for Phase B; no artifact-layout changes.

- [ ] **Step 1: Prove legacy runtime concepts are absent**

Backend:

```bash
rg -n "SearchFilters|search_id|TaskType|TaskRequest|TaskResponse|PipelineRegistry|MATCHED|EVALUATED_NO_MATCH|UNKNOWN|backfill|scene_top|candidate_pool|global_quota|local_quota" src/hcmai
```

Expected: no matches after `src/hcmai/temporal/README.md` is updated to the new baseline.

Frontend:

```bash
rg -n "search_id|progressiveSearchId|Fusion:|Rerank:|Suggest Query|parse-query-files|groupTrakeFramesByVideo" src
```

Expected: no matches.

- [ ] **Step 2: Prove detached research packages still exist**

```bash
test -d src/hcmai/retrieval/reranking
test -f src/hcmai/retrieval/retriever/fusion/rrf.py
```

Expected: both commands succeed.

- [ ] **Step 3: Remove tracked Python bytecode**

```bash
find src -type d -name __pycache__ -prune -exec rm -rf {} +
find src -type f -name '*.pyc' -delete
git status --short
```

Add `__pycache__/` and `*.pyc` to `.gitignore` if not already ignored.

- [ ] **Step 4: Run backend static compilation and tests**

```bash
PYTHONPATH=src python -m compileall -q src/hcmai
PYTHONPATH=src pytest tests -v
```

Expected: compile succeeds and all available backend tests pass.

- [ ] **Step 5: Run frontend tests and production build**

```bash
CI=true npm test -- --runInBand
npm run build
```

Expected: tests pass and build exits 0.

- [ ] **Step 6: Perform API smoke checks with dependency fakes or a configured local corpus**

Verify:

```text
POST /api/v1/search -> response has query, events, results, latency
POST /api/v1/trake  -> response has events, paths, latency
KIS score is raw float, not percentage
KIS alignment arrays match event count
TRAKE path arrays match event count
same-video TRAKE paths remain separate
valid no-path query returns 200 with empty array
```

- [ ] **Step 7: Confirm no data artifact was changed**

```bash
git status --short artifacts data
```

Expected: no generated artifact/data changes caused by Phase A.

- [ ] **Step 8: Commit Phase A closeout**

```bash
git add .
git commit -m "chore: close temporal search cleanup phase"
```
