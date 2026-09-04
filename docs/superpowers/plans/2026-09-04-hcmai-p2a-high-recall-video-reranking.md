# HCMAI P2a High-Recall Video-Level Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximize KIS Recall@20 by replacing temporal-only final video ranking with a deterministic union of global dense, event-level, and temporal candidates followed by the existing VLM reranker at the unique-video level.

**Architecture:** Preserve the existing temporal decoder and frame-level `RerankingService`. Add a shared `VideoCandidateRerankingService` that collects three candidate views, deduplicates them to at most 200 videos, selects `F_global` and `F_temporal`, obtains reranker scores, aggregates each video with `max`, and exposes one ordered video list to KIS and TRAKE. Feature flag off must preserve the current v16 KIS/TRAKE behavior exactly.

**Tech Stack:** Python 3, Pydantic v2, NumPy, existing `RetrievalService`, `TemporalSearchService`, `RerankingService`, `Corpus`, pytest, existing remote `LLMService.rerank()` endpoint.

**Spec:** `docs/superpowers/specs/2026-09-04-hcmai-p2a-high-recall-video-reranking-design.md`

## Global Constraints

- Primary rollout metric is **Recall@20**.
- Recall@100 and Recall@200 are secondary diagnostics.
- Do not change query preparation semantics.
- Do not change P0 multimodal evidence semantics.
- Do not change P1/P1a temporal recurrence or path coordinates.
- Do not regenerate or re-index artifacts.
- Keep event count dynamic; never assume `N=4`.
- Candidate union is video-level and bounded to **200 unique videos** by default.
- Production representative frames are exactly:
  - `F_global`;
  - `F_temporal`.
- Production video reranker aggregation is exactly `max`.
- No hard reranker threshold may drop a candidate.
- Existing `RerankingService` remains the model/provider boundary.
- KIS returns the representative frame that wins reranking.
- TRAKE keeps the existing temporal path unchanged; only video ordering may change.
- Reranking must fail open: an unavailable reranker must not delete candidates.
- `video_reranking.enabled` defaults to `false`.
- With `video_reranking.enabled=false`, existing v16 behavior must be preserved.
- Do not make latency a rollout gate in this phase.
- Every production behavior change follows RED → GREEN → REFACTOR.
- Do not tune candidate quotas or reranker behavior per query.

---

# Verified v16 Source Boundaries

The source archive inspected for this plan is `src_hcmai_v16.zip`.

Relevant current interfaces:

```text
src/hcmai/common/config.py
  AlignmentConfig
  SearchConfig
  AppConfig

src/hcmai/orchestration/pipeline.py
  SearchService
    corpus
    retrieval
    temporal_evidence
    KISPipeline
    TRAKEPipeline

src/hcmai/orchestration/setup.py
  load_search_service()
  _load_fast_track_retrieval()
  currently no default reranker wiring

src/hcmai/orchestration/workflows/kis.py
  KISPipeline.execute()
    split_query_events()
    TemporalSearchService.search()
    SearchMaterializer.build_kis_result()

src/hcmai/orchestration/workflows/trake.py
  TRAKEPipeline.execute()
    TemporalSearchService.search()
    _build_path()

src/hcmai/orchestration/workflows/temporal_search.py
  TemporalSearchResult
  TemporalSearchService.search()

src/hcmai/orchestration/materializer.py
  SearchMaterializer.build_kis_result()
    currently chooses path.frame_ids[len(path.frame_ids)//2]

src/hcmai/retrieval/retriever/pipeline.py
  RetrievalService.search()
  RetrievalService.search_batch()
  RetrievalService.source_retriever()
  RetrievalService.score_event_videos()

src/hcmai/retrieval/models.py
  RetrievalCandidate
  RetrievalResult
  RetrievalSource

src/hcmai/retrieval/reranking/pipeline.py
  RerankingService.rerank()
  categorized RerankingError subclasses

llm/pipeline.py
  LLMService.rerank(query, images)
```

Important public KIS contract constraint:

```text
SearchResponse validates that every SearchResult.frame_ids contains
exactly one aligned frame per query event.
```

Therefore P2a may choose a KIS representative frame outside the temporal path, but every emitted KIS video must still retain a valid temporal path of length `N`.

The uploaded v16 archive does not contain the repository test tree, but repository-owned historical plans and reranking README identify the current test conventions, including:

```text
tests/test_config.py
tests/test_reranker.py
tests/unit/orchestration/test_registry.py
tests/unit/orchestration/test_kis_pipeline.py
tests/unit/orchestration/test_trake_pipeline.py
tests/integration/test_kis_golden_path.py
```

New tests in this plan follow those conventions. If the live branch has renamed equivalents, use the existing live test file rather than creating a duplicate test suite.

---

# File Structure

## Create

```text
src/hcmai/orchestration/workflows/video_reranking.py

tests/unit/orchestration/test_video_reranking.py
tests/integration/test_p2a_high_recall_reranking.py

scripts/evaluate_p2a_recall.py
```

## Modify

```text
src/hcmai/common/config.py
src/hcmai/orchestration/setup.py
src/hcmai/orchestration/pipeline.py
src/hcmai/orchestration/materializer.py
src/hcmai/orchestration/workflows/__init__.py
src/hcmai/orchestration/workflows/kis.py
src/hcmai/orchestration/workflows/trake.py
src/hcmai/orchestration/workflows/temporal_search.py
src/hcmai/retrieval/reranking/pipeline.py

tests/test_config.py
tests/test_reranker.py
tests/unit/orchestration/test_registry.py
tests/unit/orchestration/test_kis_pipeline.py
tests/unit/orchestration/test_trake_pipeline.py
tests/integration/test_kis_golden_path.py
```

## Keep Behavioral Semantics Unchanged

```text
src/hcmai/temporal/dp.py
src/hcmai/retrieval/evidence/*
src/hcmai/query_preparation/*
src/hcmai/retrieval/reranking/adapters/*
```

---

# Task 1: Add P2a Runtime Configuration With Exact Rollback Defaults

**Purpose:** Introduce a feature-flagged P2a configuration without changing the default v16 search path.

**Files:**
- Modify: `src/hcmai/common/config.py:419-433`
- Modify: `tests/test_config.py`

**Interfaces produced:**

```python
class CandidateUnionConfig(BaseModel):
    global_target: int
    event_target: int
    temporal_target: int
```

```python
class VideoRerankingConfig(BaseModel):
    enabled: bool
    candidate_max_videos: int
    frame_candidate_top_k: int
    representative_frames: Literal[2]
    aggregation: Literal["max"]
    candidate_union: CandidateUnionConfig
```

`SearchConfig.video_reranking` becomes the runtime entry point.

- [ ] **Step 1: Write the failing default-config test**

Add to `tests/test_config.py`:

```python
from hcmai.common.config import SearchConfig


def test_video_reranking_defaults_are_recall_safe_and_disabled():
    config = SearchConfig().video_reranking

    assert config.enabled is False
    assert config.candidate_max_videos == 200
    assert config.frame_candidate_top_k == 500
    assert config.representative_frames == 2
    assert config.aggregation == "max"

    assert config.candidate_union.global_target == 120
    assert config.candidate_union.event_target == 50
    assert config.candidate_union.temporal_target == 30
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=.:src python -m pytest \
  tests/test_config.py::test_video_reranking_defaults_are_recall_safe_and_disabled \
  -q
```

Expected: FAIL because `SearchConfig.video_reranking` does not exist.

- [ ] **Step 3: Add the new config models**

In `src/hcmai/common/config.py`, immediately before `SearchConfig`, add:

```python
class CandidateUnionConfig(BaseModel):
    """Video quotas reserved for each independent recall source."""

    model_config = ConfigDict(extra="forbid")

    global_target: int = Field(default=120, ge=0)
    event_target: int = Field(default=50, ge=0)
    temporal_target: int = Field(default=30, ge=0)


class VideoRerankingConfig(BaseModel):
    """Competition-safe P2a video-level reranking configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    candidate_max_videos: int = Field(default=200, ge=1)
    frame_candidate_top_k: int = Field(default=500, ge=1)

    representative_frames: Literal[2] = 2
    aggregation: Literal["max"] = "max"

    candidate_union: CandidateUnionConfig = Field(
        default_factory=CandidateUnionConfig
    )

    @model_validator(mode="after")
    def validate_candidate_budget(self) -> "VideoRerankingConfig":
        reserved = (
            self.candidate_union.global_target
            + self.candidate_union.event_target
            + self.candidate_union.temporal_target
        )
        if reserved > self.candidate_max_videos:
            raise ValueError(
                "video reranking candidate source targets must not exceed "
                "candidate_max_videos"
            )
        if reserved == 0:
            raise ValueError(
                "video reranking candidate union must reserve at least one source"
            )
        return self
```

Then extend:

```python
class SearchConfig(BaseModel):
    ...
    video_reranking: VideoRerankingConfig = Field(
        default_factory=VideoRerankingConfig
    )
```

Do not change existing `alignment` or `hybrid_temporal` defaults.

- [ ] **Step 4: Add config validation tests**

```python
import pytest
from pydantic import ValidationError

from hcmai.common.config import (
    CandidateUnionConfig,
    SearchConfig,
    VideoRerankingConfig,
)


def test_video_reranking_rejects_source_targets_above_video_cap():
    with pytest.raises(ValidationError):
        VideoRerankingConfig(
            candidate_max_videos=100,
            candidate_union=CandidateUnionConfig(
                global_target=80,
                event_target=30,
                temporal_target=10,
            ),
        )


def test_video_reranking_requires_at_least_one_candidate_source():
    with pytest.raises(ValidationError):
        VideoRerankingConfig(
            candidate_union=CandidateUnionConfig(
                global_target=0,
                event_target=0,
                temporal_target=0,
            )
        )


def test_video_reranking_contract_is_fixed_to_two_frames_and_max():
    with pytest.raises(ValidationError):
        VideoRerankingConfig(representative_frames=1)

    with pytest.raises(ValidationError):
        VideoRerankingConfig(aggregation="mean")
```

- [ ] **Step 5: Run GREEN**

```bash
PYTHONPATH=.:src python -m pytest tests/test_config.py -q
```

Required: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hcmai/common/config.py tests/test_config.py
git commit -m "feat(search): add high-recall video reranking config"
```

---

# Task 2: Expose One Best Temporal Path Per Video Without Changing Existing Search

**Purpose:** Candidate union must not be gated by the current temporal Top-K, while KIS still requires a valid event-aligned path for every emitted result. Expose one best path for every alignable video using the existing score matrices and decoder.

**Files:**
- Modify: `src/hcmai/orchestration/workflows/temporal_search.py`
- Modify or create live equivalent: `tests/unit/orchestration/test_temporal_search.py`

**Interfaces produced:**

```python
@dataclass(frozen=True, slots=True)
class TemporalCandidateResult:
    paths: tuple[AlignedPath, ...]
    retrieval_ms: float
    alignment_ms: float
```

```python
TemporalSearchService.search_best_per_video(
    original_events: Sequence[str],
    *,
    retrieval_events: Sequence[str] | None = None,
    caption_events: Sequence[str] | None = None,
    use_dense: bool = True,
    use_bm25: bool = False,
) -> TemporalCandidateResult
```

Contract:

- exactly zero or one path per video;
- paths sorted by temporal score descending;
- all path identities are canonical;
- existing `TemporalSearchService.search()` output is unchanged.

- [ ] **Step 1: Write a failing one-best-path-per-video test**

Create or extend `tests/unit/orchestration/test_temporal_search.py` with the repository's existing fake corpus/evidence style:

```python
def test_search_best_per_video_returns_one_best_path_for_every_alignable_video(
    temporal_service,
):
    result = temporal_service.search_best_per_video(
        ["event one", "event two"],
        retrieval_events=["event one", "event two"],
        use_dense=True,
        use_bm25=False,
    )

    assert len(result.paths) >= 2
    assert len({path.video_id for path in result.paths}) == len(result.paths)
    assert [path.score for path in result.paths] == sorted(
        (path.score for path in result.paths),
        reverse=True,
    )
```

The fixture must contain at least two videos with valid paths.

- [ ] **Step 2: Add a regression test freezing `search(top_k=...)`**

Use the same fixture:

```python
def test_existing_temporal_search_ranking_is_unchanged_by_candidate_interface(
    temporal_service,
):
    before = temporal_service.search(
        ["event one", "event two"],
        retrieval_events=["event one", "event two"],
        use_dense=True,
        use_bm25=False,
        top_k=2,
    )

    all_best = temporal_service.search_best_per_video(
        ["event one", "event two"],
        retrieval_events=["event one", "event two"],
        use_dense=True,
        use_bm25=False,
    )

    after = temporal_service.search(
        ["event one", "event two"],
        retrieval_events=["event one", "event two"],
        use_dense=True,
        use_bm25=False,
        top_k=2,
    )

    assert after.paths == before.paths
    assert {path.video_id for path in all_best.paths}.issuperset(
        {path.video_id for path in before.paths}
    )
```

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/unit/orchestration/test_temporal_search.py \
  -q
```

Expected: new candidate-interface test FAILS because the method does not exist.

- [ ] **Step 4: Refactor shared validation/scoring into a private helper**

In `TemporalSearchService`, introduce a private result:

```python
@dataclass(frozen=True, slots=True)
class _ScoredTemporalRequest:
    original_events: tuple[str, ...]
    scores: tuple[VideoEventScores, ...]
    retrieval_ms: float
```

Add:

```python
def _score_request(
    self,
    original_events: Sequence[str],
    *,
    retrieval_events: Sequence[str] | None,
    caption_events: Sequence[str] | None,
    use_dense: bool,
    use_bm25: bool,
) -> _ScoredTemporalRequest:
    ...
```

Move only the existing Steps 1–3 behavior from `search()` into this helper:

- event normalization;
- max event validation;
- retrieval/caption normalization;
- `evidence.score_events(...)`;
- metadata validation;
- retrieval timing.

Do not change scoring arguments.

- [ ] **Step 5: Make existing `search()` consume `_score_request()`**

The existing decoder remains:

```python
rows = rank_paths(
    scored.scores,
    lambda_gap=self.config.lambda_gap,
    max_rows=top_k,
    event_power=self.config.event_power,
    cluster_delta=self.config.cluster_delta,
)
```

Materialize exactly as before.

- [ ] **Step 6: Implement `search_best_per_video()`**

Import existing `align_video` alongside `rank_paths`:

```python
from hcmai.temporal.dp import AlignedPath, DPPath, align_video, rank_paths
```

Implementation shape:

```python
def search_best_per_video(...):
    scored = self._score_request(...)

    alignment_started = perf_counter()
    paths: list[AlignedPath] = []

    for video in scored.scores:
        rows = align_video(
            video,
            lambda_gap=self.config.lambda_gap,
            paths=1,
            event_power=self.config.event_power,
            cluster_delta=self.config.cluster_delta,
        )
        if not rows:
            continue
        paths.append(self._materialize_aligned_path(rows[0], video))

    paths.sort(key=lambda path: (-path.score, path.video_id))

    return TemporalCandidateResult(
        paths=tuple(paths),
        retrieval_ms=scored.retrieval_ms,
        alignment_ms=(perf_counter() - alignment_started) * 1_000,
    )
```

Do not touch `src/hcmai/temporal/dp.py`.

- [ ] **Step 7: Test canonical path length**

```python
def test_best_per_video_paths_keep_one_frame_per_event(temporal_service):
    result = temporal_service.search_best_per_video(
        ["e1", "e2", "e3"],
        retrieval_events=["e1", "e2", "e3"],
    )

    assert all(len(path.frame_ids) == 3 for path in result.paths)
```

- [ ] **Step 8: Run GREEN**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/unit/orchestration/test_temporal_search.py \
  tests/unit/orchestration/test_kis_pipeline.py \
  tests/unit/orchestration/test_trake_pipeline.py \
  -q
```

Required: PASS.

- [ ] **Step 9: Commit**

```bash
git add \
  src/hcmai/orchestration/workflows/temporal_search.py \
  tests/unit/orchestration/test_temporal_search.py
git commit -m "feat(temporal): expose best aligned path per video"
```

---

# Task 3: Add Immutable Video Candidate Contracts and Deterministic Union

**Purpose:** Build the recall reservoir independently of reranker/model behavior so candidate Recall@200 can be tested by itself.

**Files:**
- Create: `src/hcmai/orchestration/workflows/video_reranking.py`
- Create: `tests/unit/orchestration/test_video_reranking.py`

**Interfaces produced:**

```python
@dataclass(frozen=True, slots=True)
class EventCandidateHit:
    event_index: int
    frame_id: str
    score: float | None
    rank: int
```

```python
@dataclass(frozen=True, slots=True)
class VideoCandidate:
    video_id: str
    union_rank: int

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
    winner_frame_source: Literal["global", "temporal", "fallback"] | None = None
```

```python
build_video_union(
    *,
    corpus: Corpus,
    global_result: RetrievalResult | None,
    event_results: Sequence[RetrievalResult],
    temporal_paths: Sequence[AlignedPath],
    config: VideoRerankingConfig,
) -> tuple[VideoCandidate, ...]
```

- [ ] **Step 1: Write a test proving one video consumes one slot**

```python
def test_video_union_deduplicates_frames_from_the_same_video(
    fake_corpus,
    candidate,
):
    global_result = retrieval_result(
        candidate("V1:F1"),
        candidate("V1:F2"),
        candidate("V2:F1"),
    )

    union = build_video_union(
        corpus=fake_corpus,
        global_result=global_result,
        event_results=(),
        temporal_paths=(),
        config=VideoRerankingConfig(
            candidate_max_videos=2,
            candidate_union=CandidateUnionConfig(
                global_target=2,
                event_target=0,
                temporal_target=0,
            ),
        ),
    )

    assert [item.video_id for item in union] == ["V1", "V2"]
```

Fake corpus must resolve:

```text
V1:F1 -> video_id V1
V1:F2 -> video_id V1
V2:F1 -> video_id V2
```

- [ ] **Step 2: Write event round-robin test**

Construct event lists:

```text
E1: A, B, C
E2: D, E, F
```

and:

```python
config = VideoRerankingConfig(
    candidate_max_videos=4,
    candidate_union=CandidateUnionConfig(
        global_target=0,
        event_target=4,
        temporal_target=0,
    ),
)
```

Assert first four event-derived videos are:

```text
A, D, B, E
```

not:

```text
A, B, C, D
```

- [ ] **Step 3: Write source-overlap refill test**

Use:

```text
global:   A, B, C
events:   A, D
temporal: B, E
```

with cap 5 and targets `3/1/1`.

Assert:

```text
A, B, C, D, E
```

contains five unique videos and overlap does not waste capacity.

- [ ] **Step 4: Write deterministic cap test**

Call `build_video_union()` twice with identical inputs and assert exact candidate equality and ranks `1..cap`.

- [ ] **Step 5: Run RED**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/unit/orchestration/test_video_reranking.py \
  -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 6: Implement score extraction helpers**

Inside `video_reranking.py`:

```python
def _retrieval_score(candidate: RetrievalCandidate) -> float | None:
    for score in (
        candidate.final_score,
        candidate.fusion_score,
    ):
        if score is not None:
            return float(score)

    if candidate.source_scores:
        return max(float(score) for score in candidate.source_scores.values())

    return None
```

Ranks remain one-based.

- [ ] **Step 7: Implement unique global stream**

For each ranked frame candidate:

1. resolve `frame = corpus.frame(candidate.frame_id)`;
2. use the first occurrence of each `frame.video_id`;
3. retain:
   - `global_frame_id`;
   - extracted score;
   - one-based unique-video global rank.

Do not let a video with 20 nearby frames occupy 20 entries.

- [ ] **Step 8: Implement event streams and round-robin order**

For each event result:

- dedupe by video independently;
- create `EventCandidateHit(event_index, frame_id, score, rank)`;
- round-robin over event rank positions;
- retain all event hits for a video even when the video entered the union through another source.

- [ ] **Step 9: Implement temporal stream**

Deduplicate `temporal_paths` by first video occurrence. Task 2 guarantees one best path per video, but keep this defensive.

Temporal rank is one-based in the supplied score ordering.

- [ ] **Step 10: Implement quota-preserving union**

Algorithm:

```text
reservoir = ordered dict keyed by video_id

consume up to global_target from global stream
consume up to event_target from event round-robin stream
consume up to temporal_target from temporal stream

fill remaining slots in source priority:
  global remaining
  event remaining
  temporal remaining

stop at candidate_max_videos
```

When a video already exists:
- update its provenance;
- do not increase reservoir length.

After membership is final, attach all known source provenance for each selected video and assign deterministic `union_rank`.

- [ ] **Step 11: Run GREEN**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/unit/orchestration/test_video_reranking.py \
  -q
```

Required: PASS.

- [ ] **Step 12: Commit**

```bash
git add \
  src/hcmai/orchestration/workflows/video_reranking.py \
  tests/unit/orchestration/test_video_reranking.py
git commit -m "feat(search): add deterministic high-recall video union"
```

---

# Task 4: Add Fail-Open Partial Scoring to the Existing Reranker

**Purpose:** P2a must reuse the existing VLM reranker but must not lose a whole candidate reservoir because one representative image is missing or one model batch fails.

**Files:**
- Modify: `src/hcmai/retrieval/reranking/pipeline.py`
- Modify: `tests/test_reranker.py`

**Existing interface that must remain unchanged:**

```python
RerankingService.rerank(
    query: str,
    candidates: Sequence[RetrievalCandidate],
) -> list[RetrievalCandidate]
```

**New interface produced:**

```python
@dataclass(frozen=True, slots=True)
class PartialRerankingResult:
    candidates: tuple[RetrievalCandidate, ...]
    failures: dict[str, str]
```

```python
RerankingService.rerank_partial(
    query: str,
    candidates: Sequence[RetrievalCandidate],
) -> PartialRerankingResult
```

`failures` maps canonical `frame_id -> failure category`.

- [ ] **Step 1: Freeze existing all-or-nothing behavior**

Add/retain:

```python
def test_rerank_still_raises_when_one_frame_asset_is_missing(...):
    service = ...
    candidates = [
        candidate("valid"),
        candidate("missing"),
    ]

    with pytest.raises(RerankerUnavailableError) as error:
        service.rerank("query", candidates)

    assert error.value.category == "frame_asset_missing"
```

This test must pass before adding the new method.

- [ ] **Step 2: Write failing partial-asset test**

```python
def test_rerank_partial_keeps_valid_frames_when_one_asset_is_missing(...):
    service = ...
    result = service.rerank_partial(
        "query",
        [
            candidate("valid"),
            candidate("missing"),
        ],
    )

    assert [item.frame_id for item in result.candidates] == ["valid"]
    assert result.candidates[0].reranker_score is not None
    assert result.failures == {"missing": "frame_asset_missing"}
```

- [ ] **Step 3: Write failing invalid-score isolation test**

Use a fake adapter that returns:

```python
[0.9, float("nan"), 0.4]
```

for three prepared items.

Assert:
- first and third survive;
- second is recorded as `"invalid_score"`;
- no surviving candidate has a non-finite score.

- [ ] **Step 4: Write batch-outage test**

Use a fake adapter that raises a timeout on its first scoring call.

Assert:

```python
result.candidates == ()
assert set(result.failures.values()) == {"timeout"}
```

All prepared frames remain represented in failures.

- [ ] **Step 5: Run RED**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/test_reranker.py \
  -q
```

Expected: new `rerank_partial` tests FAIL.

- [ ] **Step 6: Add `PartialRerankingResult`**

In `src/hcmai/retrieval/reranking/pipeline.py`:

```python
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class PartialRerankingResult:
    candidates: tuple[RetrievalCandidate, ...]
    failures: dict[str, str]
```

Export it through `__all__`.

- [ ] **Step 7: Implement independent image preparation**

`rerank_partial()` must prepare each candidate independently:

```python
prepared: list[tuple[int, RetrievalCandidate, Any]] = []
failures: dict[str, str] = {}

for position, candidate in enumerate(original):
    try:
        image_path = self.corpus.image_path(candidate.frame_id)
        image = load_image(image_path, mode="RGB")
    except FileNotFoundError:
        failures[candidate.frame_id] = "frame_asset_missing"
        continue
    except (OSError, KeyError, RuntimeError):
        failures[candidate.frame_id] = "image_load_failure"
        continue

    prepared.append((position, candidate, image))
```

Close every opened image in `finally`.

- [ ] **Step 8: Score prepared candidates in existing configured batches**

For each batch:

```python
try:
    values = list(
        self.adapter.score(
            query,
            [image for _, _, image in batch],
        )
    )
except TimeoutError:
    # mark this batch and all still-unscored prepared items as timeout
    ...
except Exception as error:
    classified = _classified_backend_error(error)
    ...
```

Rules:

- `"timeout"` and `"unavailable"` are treated as request-level outage:
  - mark current and remaining prepared candidates;
  - stop model calls.
- `"contract_error"` marks the affected batch and continues to the next batch.
- wrong score count marks the batch `"contract_error"`.
- a non-finite individual score marks only that frame `"invalid_score"`.

- [ ] **Step 9: Preserve input identity and stable output ordering**

Successful copies get:

```python
reranker_score=score
final_score=score
```

Return successful candidates through the same `_ordered()` ranking helper used by `rerank()`.

Do not mutate input candidates.

- [ ] **Step 10: Prove strict `rerank()` did not change**

Run the complete reranker test module.

Required:
- old strict behavior tests PASS;
- new partial tests PASS.

- [ ] **Step 11: Commit**

```bash
git add \
  src/hcmai/retrieval/reranking/pipeline.py \
  tests/test_reranker.py
git commit -m "feat(reranking): add fail-open partial frame scoring"
```

---

# Task 5: Build the Shared `VideoCandidateRerankingService`

**Purpose:** Compose global visual retrieval, event retrieval, temporal paths, candidate union, two-frame reranking, `max` aggregation, and fallback ordering in one task-agnostic service.

**Files:**
- Modify: `src/hcmai/orchestration/workflows/video_reranking.py`
- Modify: `src/hcmai/orchestration/workflows/__init__.py`
- Extend: `tests/unit/orchestration/test_video_reranking.py`

**Interfaces produced:**

```python
@dataclass(frozen=True, slots=True)
class VideoCandidateSet:
    candidates: tuple[VideoCandidate, ...]
    retrieval_ms: float
    temporal_ms: float
```

```python
@dataclass(frozen=True, slots=True)
class VideoRerankingResult:
    candidates: tuple[VideoCandidate, ...]
    retrieval_ms: float
    temporal_ms: float
    reranking_ms: float
    warnings: tuple[str, ...]
```

```python
class VideoCandidateRerankingService:
    def collect_candidates(... ) -> VideoCandidateSet
    def rerank_candidates(... ) -> VideoRerankingResult
    def search_and_rerank(... ) -> VideoRerankingResult
```

Production KIS/TRAKE call only `search_and_rerank()`.

Evaluation tooling may call `collect_candidates()` and `rerank_candidates()` directly to implement R2/R3/R4 without rescoring retrieval.

- [ ] **Step 1: Write a service-construction test**

```python
def test_video_reranking_service_binds_shared_dependencies(
    fake_corpus,
    fake_retrieval,
    fake_temporal,
    fake_reranker,
):
    service = VideoCandidateRerankingService(
        corpus=fake_corpus,
        retrieval=fake_retrieval,
        temporal=fake_temporal,
        reranking=fake_reranker,
        config=VideoRerankingConfig(enabled=True),
    )

    assert service.retrieval is fake_retrieval
    assert service.temporal is fake_temporal
    assert service.reranking is fake_reranker
```

- [ ] **Step 2: Define full-query text semantics**

`search_and_rerank()` accepts both original and retrieval text:

```python
def search_and_rerank(
    self,
    query: str,
    original_events: Sequence[str],
    *,
    retrieval_events: Sequence[str],
    caption_events: Sequence[str] | None,
    use_dense: bool,
    use_bm25: bool,
    top_videos: int,
) -> VideoRerankingResult:
    ...
```

Rules:

- `query` is the original human-readable narrative used by the VLM reranker.
- global visual dense query is:

```python
global_dense_query = "\n".join(retrieval_events)
```

This preserves all prepared retrieval events while presenting one full semantic query to SigLIP/dense retrieval.
- event retrieval uses each `retrieval_event` independently.
- temporal search keeps the existing original/retrieval/caption event split.

For TRAKE, caller constructs:

```python
query = "\n".join(request.events)
```

For KIS, caller uses:

```python
query = request.query
```

- [ ] **Step 3: Write global-source test proving the visual retriever is the anchor**

Fake `RetrievalService.source_retriever(RetrievalSource.VISUAL)` separately from `RetrievalService.search()`.

Assert `collect_candidates()` calls the VISUAL retriever for the global full-query view.

This prevents fused Context/ASR text leakage from silently becoming the semantic anchor.

If a configured visual retriever is unavailable, `collect_candidates()` must add warning:

```text
global_visual_unavailable
```

and continue with the remaining sources.

- [ ] **Step 4: Write event-batch retrieval test**

When `use_dense=True`:

```python
retrieval.search_batch(
    list(retrieval_events),
    top_k=config.frame_candidate_top_k,
)
```

is called exactly once.

When `use_dense=False`, neither global visual nor event retrieval runs.

Temporal search still honors `use_dense/use_bm25`.

- [ ] **Step 5: Write temporal-all-best reuse test**

Assert:

```python
temporal.search_best_per_video(...)
```

is called once.

Its sorted paths are used:
- as temporal candidate source;
- as `temporal_path` lookup for global/event videos.

No second temporal scoring request may be issued by one `collect_candidates()` call.

- [ ] **Step 6: Implement `collect_candidates()`**

Pseudo-flow:

```python
retrieval_started = perf_counter()

global_result = None
event_results = ()

if use_dense:
    visual = retrieval.source_retriever(RetrievalSource.VISUAL)
    if visual is not None:
        global_result = visual.search(
            "\n".join(retrieval_events),
            config.frame_candidate_top_k,
        )

    event_results = retrieval.search_batch(
        list(retrieval_events),
        top_k=config.frame_candidate_top_k,
    )

dense_candidate_ms = ...

temporal_result = temporal.search_best_per_video(
    original_events,
    retrieval_events=retrieval_events,
    caption_events=caption_events,
    use_dense=use_dense,
    use_bm25=use_bm25,
)

union = build_video_union(
    corpus=corpus,
    global_result=global_result,
    event_results=event_results,
    temporal_paths=temporal_result.paths,
    config=config,
)
```

Attach `temporal_path` for every selected video with an available path, not just videos admitted through the temporal quota.

- [ ] **Step 7: Choose `F_temporal` deterministically**

Current `AlignedPath` does not expose per-event emission provenance.

Therefore P2a v1 uses the approved fallback:

```python
def _temporal_representative(path: AlignedPath) -> str:
    return path.frame_ids[len(path.frame_ids) // 2]
```

Do not fabricate an "event max score" from unavailable data.

- [ ] **Step 8: Write exact representative dedupe test**

For a video with:

```text
F_global == F_temporal == "V1:F10"
```

assert only one `RetrievalCandidate(frame_id="V1:F10")` is passed to `rerank_partial()`.

Metadata on the temporary candidate must retain both roles, e.g.:

```python
metadata={
    "video_id": "V1",
    "representative_sources": ("global", "temporal"),
}
```

- [ ] **Step 9: Build reranker frame candidates**

For every distinct representative frame:

```python
RetrievalCandidate(
    frame_id=frame_id,
    metadata={
        "video_id": video_id,
        "representative_sources": sources,
    },
)
```

Do not copy temporal score into `fusion_score` or `final_score`.

The reranker must score the image independently.

- [ ] **Step 10: Implement two-frame `max` aggregation**

Map successful frame reranker scores back to each video's roles.

Rules:

```text
both scores:
  final = max(global, temporal)

one score:
  final = available score

tie:
  global representative wins

no score:
  final_rerank_score = None
  winner frame fallback:
    global if available
    otherwise temporal
```

Store:

```text
rerank_global_score
rerank_temporal_score
final_rerank_score
winner_frame_id
winner_frame_source
```

- [ ] **Step 11: Implement final ordering**

If at least one frame reranker score succeeded:

```text
1. candidates with a finite final_rerank_score
2. final_rerank_score descending
3. global_score descending, None last
4. union_rank ascending
5. video_id ascending
```

Unscored videos remain after scored videos and keep fallback order among themselves.

If **zero** frame scores succeeded across the entire candidate set:

```text
return candidate union order unchanged
```

and add warning:

```text
reranker_unavailable_fallback
```

This is the whole-service fail-open rule.

- [ ] **Step 12: Add partial-failure tests**

Required tests:

```text
global frame fails, temporal succeeds
  -> video keeps temporal score/frame

temporal frame fails, global succeeds
  -> video keeps global score/frame

both fail for one video
  -> video remains in output

all reranker frames fail
  -> output order equals union order
```

- [ ] **Step 13: Implement `rerank_candidates(..., include_temporal: bool)`**

Production calls:

```python
include_temporal=True
```

Evaluator R3 calls:

```python
include_temporal=False
```

This is an evaluation switch on an internal service method; it is not a runtime YAML setting and does not weaken the production `representative_frames=2` contract.

- [ ] **Step 14: Implement `search_and_rerank()` as composition only**

```python
candidate_set = self.collect_candidates(...)
return self.rerank_candidates(
    query,
    candidate_set,
    include_temporal=True,
)
```

Do not duplicate retrieval logic in this method.

- [ ] **Step 15: Run GREEN**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/unit/orchestration/test_video_reranking.py \
  tests/test_reranker.py \
  -q
```

Required: PASS.

- [ ] **Step 16: Commit**

```bash
git add \
  src/hcmai/orchestration/workflows/video_reranking.py \
  src/hcmai/orchestration/workflows/__init__.py \
  tests/unit/orchestration/test_video_reranking.py
git commit -m "feat(search): add shared video candidate reranking service"
```

---

# Task 6: Let KIS Return the Reranker Winner Frame While Retaining Temporal Evidence

**Purpose:** KIS ordering becomes video-reranker ordering, but its public response still contains one aligned temporal frame per event.

**Files:**
- Modify: `src/hcmai/orchestration/materializer.py:24-64`
- Modify: `src/hcmai/orchestration/workflows/kis.py`
- Modify: `tests/unit/orchestration/test_kis_pipeline.py`
- Modify: `tests/integration/test_kis_golden_path.py`

**Interfaces changed compatibly:**

```python
SearchMaterializer.build_kis_result(
    path: AlignedPath,
    *,
    representative_frame_id: str | None = None,
    score_override: float | None = None,
) -> SearchResult
```

Default arguments preserve the current upper-middle/path-score behavior.

- [ ] **Step 1: Freeze old materializer behavior**

```python
def test_build_kis_result_without_override_still_uses_upper_middle(...):
    result = materializer.build_kis_result(path)

    assert result.frame_id == path.frame_ids[len(path.frame_ids) // 2]
    assert result.score == path.score
```

Run this before modifying the materializer.

- [ ] **Step 2: Write representative override test**

Use a valid frame from the same video that is **not** in the aligned path:

```python
result = materializer.build_kis_result(
    path,
    representative_frame_id="V1:global-best",
    score_override=0.93,
)

assert result.frame_id == "V1:global-best"
assert result.video_id == path.video_id
assert result.score == pytest.approx(0.93)

assert result.frame_ids == list(path.frame_ids)
assert result.timestamps_ms == list(path.timestamps_ms)
```

This proves KIS can submit a global winner frame while retaining the full temporal evidence arrays.

- [ ] **Step 3: Reject cross-video representative overrides**

```python
with pytest.raises(ValueError, match="representative"):
    materializer.build_kis_result(
        path,
        representative_frame_id="OTHER:F1",
        score_override=0.9,
    )
```

- [ ] **Step 4: Implement compatible materializer extension**

Behavior:

```python
if representative_frame_id is None:
    representative = len(path.frame_ids) // 2
    frame_id = path.frame_ids[representative]
else:
    frame_id = representative_frame_id

frame = corpus.frame(frame_id)

if frame.video_id != path.video_id:
    raise ValueError(
        "representative frame must belong to the aligned video"
    )

score = path.score if score_override is None else score_override
```

Keep canonical frame metadata checks.

For the old midpoint path, retain the strict path index/timestamp checks exactly.

For an external same-video representative, do not pretend it occupies an event position in `path.frame_ids`.

- [ ] **Step 5: Add optional video reranking dependency to KIS**

Constructor:

```python
def __init__(
    self,
    corpus: Corpus | None,
    temporal: TemporalSearchService | None,
    query_preparation: QueryPreparationService | None = None,
    max_temporal_event_count: int = DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    video_reranking: VideoCandidateRerankingService | None = None,
) -> None:
    ...
```

Store:

```python
self.video_reranking = video_reranking
```

- [ ] **Step 6: Write KIS feature-flag-off regression**

Create KIS with:

```python
video_reranking=None
```

and assert:
- `temporal.search(... top_k=request.top_k)` is called as before;
- `SearchMaterializer.build_kis_result(path)` is used without override;
- result equality matches the existing fixture/golden path.

- [ ] **Step 7: Write P2a KIS ordering test**

Fake shared service returns:

```text
Video B:
  final rerank 0.95
  winner frame B-global
  valid path B

Video A:
  final rerank 0.80
  winner frame A-temporal
  valid path A
```

Assert KIS results order:

```text
B, A
```

and frame IDs:

```text
B-global
A-temporal
```

while each result still contains exactly `N` temporal path frames.

- [ ] **Step 8: Handle candidates without an alignable path**

The shared union may retain a global/event candidate with:

```python
temporal_path is None
```

KIS cannot emit it because `SearchResponse` requires one aligned frame per event.

KIS must:
- skip such a candidate at materialization;
- continue to the next ranked video;
- return up to `request.top_k` candidates with valid paths.

Do not synthesize fake temporal arrays.

Add a test where rank 1 has no path and rank 2 does; rank 2 must still be emitted.

- [ ] **Step 9: Wire P2a branch inside `execute()`**

Keep event preparation unchanged.

Branch only after events/retrieval/caption events are frozen:

```python
if self.video_reranking is None:
    # exact legacy v16 temporal path
else:
    reranked = self.video_reranking.search_and_rerank(
        request.query,
        events,
        retrieval_events=retrieval_events,
        caption_events=caption_events,
        use_dense=request.use_dense,
        use_bm25=request.use_bm25,
        top_videos=request.top_k,
    )
```

Materialize ranked candidates with:

```python
self.materializer.build_kis_result(
    candidate.temporal_path,
    representative_frame_id=candidate.winner_frame_id,
    score_override=(
        candidate.final_rerank_score
        if candidate.final_rerank_score is not None
        else candidate.global_score
        if candidate.global_score is not None
        else candidate.temporal_path.score
    ),
)
```

Limit after skipping pathless candidates.

- [ ] **Step 10: Preserve KIS latency contract**

Do not add HTTP fields.

Map internal timings:

```text
SearchLatency.retrieval_ms =
  candidate retrieval time + reranking time

SearchLatency.alignment_ms =
  temporal candidate scoring/alignment time

SearchLatency.materialization_ms =
  unchanged materialization timing
```

Latency is telemetry only; do not use it as a rollout gate.

- [ ] **Step 11: Run KIS tests**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/unit/orchestration/test_kis_pipeline.py \
  tests/integration/test_kis_golden_path.py \
  -q
```

Required: PASS.

- [ ] **Step 12: Commit**

```bash
git add \
  src/hcmai/orchestration/materializer.py \
  src/hcmai/orchestration/workflows/kis.py \
  tests/unit/orchestration/test_kis_pipeline.py \
  tests/integration/test_kis_golden_path.py
git commit -m "feat(kis): rank videos with shared visual reranker"
```

---

# Task 7: Reorder TRAKE Videos Without Changing Temporal Paths

**Purpose:** TRAKE shares P2a candidate/reranking evidence but preserves every path's temporal coordinates and raw path score.

**Files:**
- Modify: `src/hcmai/orchestration/workflows/trake.py`
- Modify: `tests/unit/orchestration/test_trake_pipeline.py`

- [ ] **Step 1: Freeze current TRAKE path projection**

Add/retain a test asserting `_build_path()` preserves:

```text
video_id
path.score
frame_ids
frame_idxs
timestamps_ms
```

exactly.

- [ ] **Step 2: Add optional shared video reranking dependency**

Constructor:

```python
def __init__(
    self,
    temporal: TemporalSearchService | None,
    query_preparation: QueryPreparationService | None = None,
    max_temporal_event_count: int = DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    video_reranking: VideoCandidateRerankingService | None = None,
) -> None:
    ...
```

- [ ] **Step 3: Write feature-off regression**

With `video_reranking=None`, `execute()` must call the current:

```python
self.temporal.search(...)
```

and return existing path ordering unchanged.

- [ ] **Step 4: Write reranked-order/path-identity test**

Fake P2a service returns candidate order:

```text
B, A
```

with paths whose temporal scores are:

```text
A score 10.0
B score 4.0
```

Assert output order is:

```text
B, A
```

but returned TRAKE scores remain:

```text
4.0, 10.0
```

and their frame/timestamp arrays exactly equal the source `AlignedPath`s.

This proves reranking changes only video ordering.

- [ ] **Step 5: Define TRAKE full-query text**

TRAKE has no raw query string.

For P2a:

```python
reranker_query = "\n".join(request.events)
```

Use original events, not retrieval rewrites, for the VLM relevance query.

Pass `request.retrieval_events or request.events` separately for dense/event retrieval.

- [ ] **Step 6: Implement P2a branch**

When shared video reranking is present:

```python
reranked = self.video_reranking.search_and_rerank(
    "\n".join(events),
    events,
    retrieval_events=retrieval_events,
    caption_events=caption_events,
    use_dense=request.use_dense,
    use_bm25=request.use_bm25,
    top_videos=request.top_k,
)
```

Return up to `request.top_k` candidates with non-`None` `temporal_path`.

Do not replace the path score with reranker score.

- [ ] **Step 7: Preserve latency response schema**

Use the same mapping as KIS:
- retrieval includes candidate retrieval + reranker;
- alignment is temporal work;
- materialization is response projection.

- [ ] **Step 8: Run TRAKE tests**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/unit/orchestration/test_trake_pipeline.py \
  -q
```

Required: PASS.

- [ ] **Step 9: Commit**

```bash
git add \
  src/hcmai/orchestration/workflows/trake.py \
  tests/unit/orchestration/test_trake_pipeline.py
git commit -m "feat(trake): reuse high-recall video reranking order"
```

---

# Task 8: Wire the Existing Reranker at Startup and Share One P2a Service

**Purpose:** Restore reranking as an explicit optional runtime dependency, created once at application startup, while keeping disabled mode identical to v16.

**Files:**
- Modify: `src/hcmai/orchestration/setup.py:40-99`
- Modify: `src/hcmai/orchestration/pipeline.py:78-151, 207-308`
- Modify: `tests/unit/orchestration/test_registry.py`
- Add/modify live setup tests if present.

**Interfaces:**

`SearchService.__init__` gains:

```python
reranking: RerankingService | None = None
```

and stores:

```python
self.reranking
self.video_reranking
```

KIS and TRAKE receive the same `VideoCandidateRerankingService` instance.

- [ ] **Step 1: Write registry rollback test**

```python
def test_video_reranking_disabled_preserves_detached_reranker(
    search_config,
    fake_dependencies,
):
    search_config.video_reranking.enabled = False
    service = build_search_service(...)

    assert service.video_reranking is None
    assert service.kis.video_reranking is None
    assert service.trake.video_reranking is None
```

If Pydantic/frozen fixtures require `model_copy(update=...)`, use that instead of mutating.

- [ ] **Step 2: Write shared-service registry test**

When enabled:

```python
assert service.video_reranking is service.kis.video_reranking
assert service.video_reranking is service.trake.video_reranking
```

No request handler may construct another reranker.

- [ ] **Step 3: Add `_load_reranking_service()`**

In `setup.py`:

```python
def _load_reranking_service(
    settings: AppConfig,
    models: LLMServiceConfig,
    corpus: Corpus | None,
    llm: LLMService | None,
    messages: list[str],
) -> RerankingService | None:
    if not settings.search.video_reranking.enabled:
        return None

    if corpus is None:
        messages.append(
            "Video reranking model unavailable: canonical corpus not loaded"
        )
        return None

    if llm is None:
        messages.append(
            "Video reranking model unavailable: inference service not loaded"
        )
        return None

    return RerankingService.remote(
        corpus,
        RerankerConfig(
            batch_size=models.reranker.batch_size,
            required=False,
        ),
        llm,
    )
```

Import:

```python
from hcmai.retrieval.reranking.pipeline import (
    RerankerConfig,
    RerankingService,
)
```

Use the existing `LLMService.rerank()` remote/provider boundary.

- [ ] **Step 4: Call loader only after corpus/LLM exist**

Inside `load_search_service()`:

```python
reranking = _load_reranking_service(
    settings,
    models,
    corpus,
    llm,
    messages,
)
```

Pass:

```python
reranking=reranking
```

to `SearchService`.

Do not modify `_load_fast_track_retrieval()` to own reranking.

- [ ] **Step 5: Build one shared P2a service in `SearchService`**

After `temporal` is constructed:

```python
self.reranking = reranking

self.video_reranking = (
    VideoCandidateRerankingService(
        corpus=self.corpus,
        retrieval=self.retrieval,
        temporal=temporal,
        reranking=self.reranking,
        config=self.config.video_reranking,
    )
    if (
        self.config.video_reranking.enabled
        and self.corpus is not None
        and self.retrieval is not None
        and temporal is not None
    )
    else None
)
```

Important:

```text
reranking=None
```

does **not** prevent construction of `VideoCandidateRerankingService` when the feature is enabled.

That service must still provide candidate-union fallback ranking.

- [ ] **Step 6: Inject the exact same instance**

```python
self.kis = KISPipeline(
    ...,
    video_reranking=self.video_reranking,
)

self.trake = TRAKEPipeline(
    ...,
    video_reranking=self.video_reranking,
)
```

- [ ] **Step 7: Update health reporting**

Add capability fields without removing existing keys:

```python
"video_reranking": self.video_reranking is not None,
"reranker_model": self.reranking is not None,
```

Keep remote capability health unchanged.

This also makes the existing startup log:

```python
getattr(service, "reranking", None) is not None
```

truthful again when enabled.

- [ ] **Step 8: Test enabled-without-model fail-open wiring**

Construct `SearchService` with:
- feature enabled;
- corpus/retrieval/temporal available;
- `reranking=None`.

Assert:
- `service.video_reranking is not None`;
- KIS/TRAKE share it;
- requests return union fallback rather than raising because the model is absent.

- [ ] **Step 9: Run registry/setup tests**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/unit/orchestration/test_registry.py \
  tests/unit/orchestration/test_kis_pipeline.py \
  tests/unit/orchestration/test_trake_pipeline.py \
  -q
```

Required: PASS.

- [ ] **Step 10: Commit**

```bash
git add \
  src/hcmai/orchestration/setup.py \
  src/hcmai/orchestration/pipeline.py \
  tests/unit/orchestration/test_registry.py
git commit -m "feat(search): wire optional shared video reranking runtime"
```

---

# Task 9: Add an R0-R4 Recall Evaluation Harness

**Purpose:** Measure whether candidate union and VLM reranking actually improve Recall@20 before enabling P2a.

**Files:**
- Create: `scripts/evaluate_p2a_recall.py`
- Create: `tests/unit/orchestration/test_p2a_evaluation.py` or nearest existing evaluation-test location.
- Runtime output only after execution:
  - `artifacts/evaluation/p2a_recall_results.jsonl`
  - `artifacts/evaluation/p2a_recall_summary.json`

**Input format:**

Use a labeled JSONL file independent of `query.zip`:

```json
{
  "query_id": "query-p2-example",
  "query": "original Vietnamese query",
  "gt_video_id": "L24_V018",
  "retrieval_events": [
    "event 1 retrieval text",
    "event 2 retrieval text",
    "event 3 retrieval text"
  ]
}
```

If `retrieval_events` is absent:
- split the query with the current deterministic KIS parser;
- use the resulting events as retrieval events.

Do not invent GT labels from query text.

- [ ] **Step 1: Define exact run semantics**

```text
R0 global:
  unique-video order from full-query VISUAL dense candidates only

R1 temporal:
  TemporalCandidateResult.paths order

R2 union:
  VideoCandidateSet.candidates union order, no reranker

R3 rerank-1:
  same R2 candidate objects
  rerank global representatives only

R4 rerank-2:
  same R2 candidate objects
  rerank global + temporal representatives
```

R2/R3/R4 must reuse the exact same `collect_candidates()` result.

- [ ] **Step 2: Write a pure Recall metric test**

```python
def test_recall_at_k_counts_query_when_gt_is_within_cutoff():
    ranks = [1, 20, 21, None]

    assert recall_at_k(ranks, 20) == pytest.approx(0.5)
```

- [ ] **Step 3: Write run-isolation test**

Fake service counts retrieval calls.

Assert one query evaluation performs:

```text
collect_candidates -> once
R3 rerank -> from frozen candidate set
R4 rerank -> from same frozen candidate set
```

It must not regenerate retrieval for each run.

- [ ] **Step 4: Implement per-query diagnostics**

Write one JSONL record per query containing:

```json
{
  "query_id": "...",
  "gt_video_id": "...",

  "r0_global_rank": 8,
  "r1_temporal_rank": 1,
  "r2_union_rank": 4,
  "r3_rerank1_rank": 3,
  "r4_rerank2_rank": 1,

  "gt_in_global_pool": true,
  "gt_in_event_pool": true,
  "gt_in_temporal_pool": true,

  "global_frame_id": "...",
  "temporal_frame_id": "...",

  "rerank_global_score": 0.91,
  "rerank_temporal_score": 0.73,
  "winner_frame_source": "global"
}
```

Also record the final R4 top wrong video with source membership/ranks/scores.

- [ ] **Step 5: Implement aggregate summary**

Required metrics by run:

```text
Recall@20
Recall@100
Recall@200
median finite GT rank
MRR
```

Also report:

```text
candidate source coverage:
  global
  event
  temporal
  union
```

- [ ] **Step 6: Implement candidate gate assertion/report**

Report whether:

```python
recall_union_200 >= max(
    recall_global_200,
    recall_event_200,
    recall_temporal_200,
)
```

If false, list every query whose GT was present in a constituent source but absent from union@200.

Do not silently tune quotas inside the evaluator.

- [ ] **Step 7: Implement reranker gate report**

Report:

```python
recall_r4_20 > recall_r2_20
```

and list:
- R4 wins;
- R4 regressions;
- unchanged queries.

Do not auto-enable runtime config.

- [ ] **Step 8: Add CLI**

Example:

```bash
PYTHONPATH=.:src python scripts/evaluate_p2a_recall.py \
  --queries artifacts/evaluation/p2a_labeled_queries.jsonl \
  --output artifacts/evaluation/p2a_recall_results.jsonl \
  --summary artifacts/evaluation/p2a_recall_summary.json
```

`--help` must work without loading indexes/models.

- [ ] **Step 9: Run evaluator unit tests**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/unit/orchestration/test_p2a_evaluation.py \
  -q
```

Required: PASS.

- [ ] **Step 10: Commit**

```bash
git add \
  scripts/evaluate_p2a_recall.py \
  tests/unit/orchestration/test_p2a_evaluation.py
git commit -m "chore(eval): add P2a recall ablation harness"
```

---

# Task 10: Integration Regression and Feature-Flag Rollback

**Purpose:** Prove that P2a can be merged safely before it is enabled and that its HTTP response contracts remain valid.

**Files:**
- Create: `tests/integration/test_p2a_high_recall_reranking.py`
- Modify live integration tests only where the feature flag requires dependency setup.

- [ ] **Step 1: Add disabled-mode golden regression**

Build a service with:

```python
SearchConfig(
    video_reranking=VideoRerankingConfig(enabled=False),
)
```

Run the existing KIS golden fixture.

Assert exact equality to the current v16 expected:
- result video order;
- representative frame IDs;
- path arrays;
- scores.

This is the rollback contract.

- [ ] **Step 2: Add enabled KIS integration fixture**

Use:
- three candidate videos;
- duplicate raw frames from one false-positive video;
- one correct video found by global/event/temporal;
- fake reranker scores making the correct visual scene win.

Assert:
- one video cannot occupy multiple Top-20 slots;
- final rank follows video reranker score;
- KIS representative equals winner frame;
- each KIS result still contains exactly one path frame per event.

- [ ] **Step 3: Add lecture-false-positive regression shape**

Use synthetic metadata/candidate IDs representing:

```text
wrong lecture video:
  high temporal/source evidence
  low VLM frame score

correct narrative video:
  moderate retrieval rank
  high VLM frame score
```

Assert P2a moves the correct video above the lecture.

Do not special-case title/OCR strings in production logic. The test behavior must arise only from fake reranker scores.

- [ ] **Step 4: Add whole-reranker-failure integration test**

Fake reranker returns no successful frame scores.

Assert:
- no candidate disappears solely because reranking failed;
- KIS still returns candidate-union results;
- TRAKE still returns temporal paths.

- [ ] **Step 5: Add TRAKE path identity integration test**

Compare enabled/disabled runs for the same chosen video.

The path:

```text
frame_ids
frame_idxs
timestamps_ms
path score
```

must be identical even if its rank changes.

- [ ] **Step 6: Run integration tests**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/integration/test_kis_golden_path.py \
  tests/integration/test_p2a_high_recall_reranking.py \
  -q
```

Required: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  tests/integration/test_p2a_high_recall_reranking.py \
  tests/integration/test_kis_golden_path.py
git commit -m "test(search): verify P2a recall and rollback contracts"
```

---

# Task 11: Full Verification and Recall Rollout Gate

**Purpose:** Decide whether P2a should become the competition default. No algorithm changes belong in this task.

## 11A. Static/import verification

- [ ] **Step 1: Compile changed source**

```bash
python -m compileall -q src llm scripts
```

Required: exit code 0.

## 11B. Focused tests

- [ ] **Step 2: Run reranker and P2a unit tests**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/test_config.py \
  tests/test_reranker.py \
  tests/unit/orchestration/test_video_reranking.py \
  tests/unit/orchestration/test_temporal_search.py \
  tests/unit/orchestration/test_registry.py \
  tests/unit/orchestration/test_kis_pipeline.py \
  tests/unit/orchestration/test_trake_pipeline.py \
  tests/unit/orchestration/test_p2a_evaluation.py \
  -q
```

Required: 0 failures.

## 11C. Integration tests

- [ ] **Step 3: Run KIS/TRAKE integration tests**

```bash
PYTHONPATH=.:src python -m pytest \
  tests/integration/test_kis_golden_path.py \
  tests/integration/test_p2a_high_recall_reranking.py \
  -q
```

Required: 0 failures.

## 11D. Full repository suite

- [ ] **Step 4: Run the full test suite**

```bash
PYTHONPATH=.:src python -m pytest -q
```

Required before a completion claim: 0 unexpected failures.

If the live environment has known pre-existing failures, prove them against the base revision rather than labeling them pre-existing by assumption.

## 11E. Rollback equivalence

- [ ] **Step 5: Run a frozen KIS/TRAKE sample with P2a disabled**

Configuration:

```yaml
search:
  video_reranking:
    enabled: false
```

Compare pre-P2a branch vs P2a branch:
- KIS video IDs;
- KIS representative frames;
- KIS temporal arrays;
- KIS scores;
- TRAKE paths.

Required: exact behavior equality for the frozen fixture/query set.

## 11F. Recall evaluation

- [ ] **Step 6: Build a labeled benchmark subset**

From `query.zip`, manually prepare at least the queries for which a reliable `gt_video_id` is known.

Do not infer labels automatically.

Prioritize:
- dense-good / temporal-bad failures;
- temporal-good narrative queries;
- simple single-scene KIS queries;
- the known yellow-animal/pumpkin-style failure class.

- [ ] **Step 7: Run R0-R4**

```bash
PYTHONPATH=.:src python scripts/evaluate_p2a_recall.py \
  --queries artifacts/evaluation/p2a_labeled_queries.jsonl \
  --output artifacts/evaluation/p2a_recall_results.jsonl \
  --summary artifacts/evaluation/p2a_recall_summary.json
```

Read the actual output before making any rollout claim.

- [ ] **Step 8: Check candidate gate**

Required:

```text
union Recall@200 must not underperform the strongest constituent
source Recall@200 on the labeled benchmark without an explicitly
explained 200-video-cap eviction.
```

If union loses GT videos, do not tune the reranker. Fix candidate union first.

- [ ] **Step 9: Check reranker gate**

Primary requirement:

```text
R4 Recall@20 > R2 Recall@20
```

Also compare:

```text
R4 Recall@20 vs R0 global
R4 Recall@20 vs R1 temporal
R4 Recall@20 vs R3 one-frame
```

P2a should not be enabled if the two-frame reranker systematically hurts easy global-dense queries.

- [ ] **Step 10: Inspect query-level regressions**

For every R4 regression:
- GT global rank;
- GT event rank;
- GT temporal rank;
- GT union rank;
- global representative frame;
- temporal representative frame;
- both reranker scores;
- top false-positive scores.

Classify the failure at the actual stage before changing weights/quotas.

- [ ] **Step 11: Decide rollout**

Enable:

```yaml
search:
  video_reranking:
    enabled: true
```

only when all are true:

```text
[ ] full suite passes
[ ] disabled mode matches v16 baseline
[ ] union candidate gate passes
[ ] R4 Recall@20 improves over R2
[ ] R4 is competitive with or better than the strongest R0/R1 baseline
[ ] improvements are distributed across multiple labeled queries
[ ] simple KIS queries do not systematically regress
[ ] false-positive visual mismatches are actually suppressed
```

Latency is recorded but is **not** a gate in P2a.

- [ ] **Step 12: Record evaluation state**

Update:

```text
KNOWLEDGE.md
docs/research/2026-09-04-p2a-high-recall-reranking-evaluation.md
```

State one of:

```text
IMPLEMENTED / EVALUATION PENDING
IMPLEMENTED / NO-GO
IMPLEMENTED / ROLLOUT APPROVED
```

Include exact test counts and Recall metrics from fresh commands.

- [ ] **Step 13: Commit evaluation report only after real evaluation**

```bash
git add \
  KNOWLEDGE.md \
  docs/research/2026-09-04-p2a-high-recall-reranking-evaluation.md
git commit -m "docs(search): record P2a recall evaluation"
```

Do not fabricate this commit before the benchmark exists.

---

# Implementation Notes

## 1. Global semantic anchor is VISUAL dense, not the fused fast-track rank

P2a's global source exists specifically to protect joint visual semantics from fragmented text evidence.

Use:

```python
retrieval.source_retriever(RetrievalSource.VISUAL)
```

for the global full-query source.

Do not use the fused Context/ASR ranking as `global_rank`.

Event retrieval may use the configured batched retrieval stack because it is only one candidate-rescue source.

## 2. Use prepared retrieval events for the global dense embedding

KIS may have Vietnamese original text and model-oriented retrieval events.

Construct:

```python
global_dense_query = "\n".join(retrieval_events)
```

for visual dense retrieval.

Use the original human query for VLM reranking.

This keeps query preparation frozen while preserving whole-query context.

## 3. Candidate union is membership/rank based

Never calculate:

```python
dense_score + temporal_score + event_score
```

to decide candidate membership.

Those scales are not assumed calibrated.

## 4. KIS representative frame may be outside its aligned path

That is intentional in P2a.

The result still carries:
- a valid `N`-frame temporal path;
- a representative frame from the same video.

Do not rewrite path arrays to pretend the global representative is one of the event alignments.

## 5. Do not synthesize temporal paths

If a video has no valid temporal path:
- it may remain in candidate diagnostics/reranking;
- it cannot be emitted by current KIS/TRAKE public contracts;
- task heads skip it and backfill with the next ranked alignable video.

Changing the HTTP alignment contract is outside P2a.

## 6. Reranker score is KIS final ranking score

When reranking succeeds:

```text
SearchResult.score = final_rerank_score
```

TRAKE is different:

```text
TRAKEPath.score = existing temporal path score
```

Do not overwrite TRAKE path score with VLM score.

## 7. Exact duplicate frame is scored once

If the same canonical frame is both global and temporal representative:
- make one temporary `RetrievalCandidate`;
- retain both source roles in metadata;
- reuse its one score for both roles.

## 8. Do not add a reranker threshold

Low VLM score is a rank signal, not a hard rejection.

Candidate deletion based on a tuned threshold is P2b/research scope.

## 9. Keep R0-R4 inputs frozen per query

For one query:
- perform candidate collection once;
- derive R0/R1/R2 source order from that collection;
- rerank the same R2 candidates for R3/R4.

Do not re-run candidate generation with different settings per ablation.

## 10. Do not tune the 120/50/30 quota on the yellow-animal query alone

The quota is an initial design default.

Any quota change requires aggregate labeled-query evidence.

---

# Recommended Commit Sequence

```text
1. feat(search): add high-recall video reranking config
2. feat(temporal): expose best aligned path per video
3. feat(search): add deterministic high-recall video union
4. feat(reranking): add fail-open partial frame scoring
5. feat(search): add shared video candidate reranking service
6. feat(kis): rank videos with shared visual reranker
7. feat(trake): reuse high-recall video reranking order
8. feat(search): wire optional shared video reranking runtime
9. chore(eval): add P2a recall ablation harness
10. test(search): verify P2a recall and rollback contracts
11. docs(search): record P2a recall evaluation  # only after benchmark
```

Keep commits separate during competition development so regressions can be bisected quickly.

---

# Definition of Done

P2a implementation is complete only when:

```text
[ ] video_reranking config exists and defaults disabled
[ ] global semantic source is full-query VISUAL dense
[ ] event retrieval uses dynamic N
[ ] one best temporal path is available per alignable video
[ ] global/event/temporal candidates union at video level
[ ] max candidate reservoir is 200 unique videos
[ ] event candidate merge is round-robin
[ ] source overlap does not waste slots
[ ] F_global is retained
[ ] F_temporal is deterministic
[ ] exact representative duplicates are scored once
[ ] existing RerankingService provider boundary is reused
[ ] partial frame failures do not delete the whole candidate reservoir
[ ] R_video uses max aggregation
[ ] no reranker threshold exists
[ ] whole reranker failure falls back to union order
[ ] KIS returns the reranker winner frame
[ ] KIS still retains one temporal frame per event
[ ] TRAKE path identity and path score remain unchanged
[ ] KIS and TRAKE share one video reranking service
[ ] feature-disabled behavior matches v16
[ ] R0-R4 evaluation harness exists
[ ] candidate Recall@200 is measured
[ ] primary Recall@20 is measured
[ ] rollout is based on fresh Recall evidence
```
