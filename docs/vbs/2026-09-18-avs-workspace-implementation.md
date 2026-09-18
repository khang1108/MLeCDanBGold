# AVS Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dedicated low-latency AVS workflow that directly retrieves a large candidate pool, deterministically suppresses same-video temporal duplicates, exposes many videos early, lets the user harvest many frames into a task-scoped basket, and submits the selected batch as one DRES answer set.

**Architecture:** Extend the existing standalone retrieval service with a direct text-search boundary that returns the current shared retrieval ordering without temporal decoding or learned reranking. The web backend materializes canonical frame metadata, applies a pure `AvsCoverageSelector`, and exposes `/api/v1/avs/search`; the frontend routes AVS tasks into a dedicated workspace whose search, selection, and submission state are isolated from KIS. Existing one-answer KIS/VQA UI remains on top of a backward-compatible singular frontend wrapper while the browser-to-backend submission contract becomes an `answers[]` batch.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic v2, httpx, existing HCMAI retrieval/corpus services, React 19, Jest + React Testing Library, CSS.

**Spec:** `docs/superpowers/specs/2026-09-18-avs-workspace-design.md`

## Global Constraints

- AVS v1 must not call `KISIntentResolver`, KIS rewrite/patch logic, temporal DP/alignment, EventTrail, or an online visual reranker.
- AVS v1 uses the shared retrieval ordering as relevance evidence; post-retrieval ordering is deterministic.
- Raw AVS retrieval pool size must be larger than the visible result count.
- Hard deduplication is limited to same-video candidates inside the configured temporal window; cross-video visual similarity is not part of v1.
- The system exposes candidates; the user decides correctness and selection.
- Grid search state, pending selection state, and submission outcome state remain independent.
- Selecting/deselecting a card is a local frontend state change and must not call the backend.
- Canonical selection identity is `candidate_id == frame_id`; list position is never an identity.
- Pending selections survive repeated AVS searches within the same frozen DRES task scope.
- KIS and VQA remain exactly-one-answer submission tasks; AVS accepts one-or-more temporal answers.
- One AVS browser submission request must map to exactly one DRES `ApiClientAnswerSet` containing all selected answers.
- `NOT_RECORDED` and `UNKNOWN` outcomes never clear pending selections; `UNKNOWN` never auto-retries.
- Do not add cross-video embedding diversity, clustering, learned AVS reranking, AVS-specific LLM logic, or select-all-visible in this plan.
- The extracted `/mnt/data/src_v1_7_extracted` tree is not a git checkout. Commit steps below are for execution in the real repository checkout; do not initialize a new repository merely to satisfy them.

---

## File Structure Map

### Backend / retrieval service

- `src/hcmai/common/config.py` — add server-side AVS tuning values.
- `src/hcmai/orchestration/workflows/avs.py` — pure coverage candidate and deterministic selector.
- `src/hcmai/retrieval/serving/schemas.py` — direct text-search wire schemas.
- `src/hcmai/retrieval/serving/runtime.py` — retain loaded `RetrievalService` and expose direct text search.
- `src/hcmai/retrieval/serving/server.py` — add standalone retrieval `/search_text` route.
- `src/hcmai/retrieval/serving/client.py` — add main-backend client mapping for `/search_text`.
- `src/hcmai/api/contracts/avs.py` — public AVS request/result/latency contracts.
- `src/hcmai/api/contracts/__init__.py` — export AVS contracts.
- `src/hcmai/orchestration/workflows/avs_search.py` — direct retrieval + coverage + canonical materialization.
- `src/hcmai/orchestration/pipeline.py` — construct/delegate the AVS workflow without entering KIS.
- `src/hcmai/api/routers/avs.py` — expose `POST /api/v1/avs/search`.
- `src/hcmai/api/routers/__init__.py`, `src/hcmai/app.py` — register the AVS router.
- `src/hcmai/api/result_logging.py` — shared best-effort DRES result-log helper used by existing search and AVS.
- `src/hcmai/api/routers/search.py` — consume the extracted result-log helper.

### Submission backend + transport

- `src/hcmai/api/contracts/vbs.py` — change browser submission request from `answer` to `answers`.
- `src/hcmai/api/routers/vbs.py` — task-family validation, list mapping, one DRES answer set.
- `frontend/src/api/submissions.js` — add batch transport while retaining the singular KIS/VQA wrapper.
- `frontend/src/api/submissions.test.js` — batch/singular transport contract tests.

### AVS frontend

- `frontend/src/api/avs.js`, `frontend/src/api/avs.test.js` — AVS search client and response validation.
- `frontend/src/features/avs/selectionState.js`, `selectionState.test.js` — pure task-scoped basket reducer.
- `frontend/src/features/avs/gridNavigation.js`, `gridNavigation.test.js` — directional keyboard focus helper.
- `frontend/src/features/avs/components/AvsQueryControls.jsx` — AVS query + task selector.
- `frontend/src/features/avs/components/AvsCandidateCard.jsx` — one selectable/inspectable result.
- `frontend/src/features/avs/components/AvsHarvestGrid.jsx` — grid + keyboard navigation.
- `frontend/src/features/avs/components/AvsSelectionBar.jsx` — sticky pending count/actions.
- `frontend/src/features/avs/components/AvsSelectionDrawer.jsx` — review/remove selected frames.
- `frontend/src/features/avs/components/AvsSubmitDialog.jsx` — lightweight multi-answer confirmation and uncertain-outcome actions.
- `frontend/src/features/avs/components/AvsWorkspace.jsx` — search, scope, selection, and submission orchestration.
- `frontend/src/features/avs/components/AvsWorkspace.test.jsx` — end-to-end AVS interaction tests.
- `frontend/src/features/avs/hooks/useAvsSubmission.js`, `useAvsSubmission.test.js` — failure-safe batch submission state machine.
- `frontend/src/features/avs/index.js` — feature exports.
- `frontend/src/features/vbs/taskFamily.js`, `taskFamily.test.js` — deterministic frontend AVS task classification.
- `frontend/src/features/vbs/components/VbsTaskSelector.jsx`, `VbsTaskSelector.test.jsx` — reusable task selector extracted from KIS toolbox.
- `frontend/src/features/search-controls/components/ToolBox.jsx` — use shared task selector instead of owning task-change UI.
- `frontend/src/App.jsx`, `frontend/src/App.test.jsx`, `frontend/src/App.avs.test.jsx` — route AVS tasks to `AvsWorkspace`, preserve KIS modal/submission behavior.
- `frontend/src/styles/avs.css`, `frontend/src/styles/index.css` — AVS grid, sticky bar, drawer, dialog, focus states.

### New backend tests

The archive currently has no Python test tree. Create focused pytest modules under `tests/`; the full backend verification command is `PYTHONPATH=src python -m pytest tests -v`.

---

### Task 1: Add AVS configuration and deterministic coverage selection

**Files:**
- Modify: `src/hcmai/common/config.py:422-435`
- Create: `src/hcmai/orchestration/workflows/avs.py`
- Create: `tests/orchestration/workflows/test_avs_coverage.py`

**Interfaces:**
- Produces: `AvsConfig(candidate_pool_size, default_page_size, maximum_page_size, temporal_dedup_window_ms, coverage_policy_version)`.
- Produces: `AvsCoverageCandidate(frame_id, video_id, timestamp_ms, retrieval_rank, retrieval_score)`.
- Produces: `AvsCoverageSelector(config: AvsConfig).order(candidates: Sequence[AvsCoverageCandidate]) -> list[AvsCoverageCandidate]`.
- Consumed later by: `AvsSearchService` in Task 3.

- [ ] **Step 1: Write failing configuration and selector tests**

```python
# tests/orchestration/workflows/test_avs_coverage.py
from hcmai.common.config import AvsConfig
from hcmai.orchestration.workflows.avs import AvsCoverageCandidate, AvsCoverageSelector


def c(frame_id: str, video_id: str, ts: int, rank: int, score: float) -> AvsCoverageCandidate:
    return AvsCoverageCandidate(
        frame_id=frame_id,
        video_id=video_id,
        timestamp_ms=ts,
        retrieval_rank=rank,
        retrieval_score=score,
    )


def selector(window_ms: int = 3000) -> AvsCoverageSelector:
    return AvsCoverageSelector(AvsConfig(temporal_dedup_window_ms=window_ms))


def test_config_requires_pool_larger_than_visible_page():
    try:
        AvsConfig(candidate_pool_size=80, maximum_page_size=80)
    except ValueError:
        pass
    else:
        raise AssertionError("candidate pool must be larger than maximum visible page")


def test_same_video_nearby_candidates_keep_strongest_retrieval_rank():
    ordered = selector().order([
        c("v1-a", "V1", 10_000, 1, .99),
        c("v1-b", "V1", 11_000, 2, .98),
        c("v2-a", "V2", 20_000, 3, .97),
    ])
    assert [item.frame_id for item in ordered] == ["v1-a", "v2-a"]


def test_same_video_candidates_outside_window_remain_eligible():
    ordered = selector(window_ms=2000).order([
        c("v1-a", "V1", 10_000, 1, .99),
        c("v1-b", "V1", 13_000, 2, .98),
    ])
    assert [item.frame_id for item in ordered] == ["v1-a", "v1-b"]


def test_video_first_pass_exposes_other_videos_before_second_v1_candidate():
    ordered = selector(window_ms=0).order([
        c("v1-a", "V1", 10_000, 1, .99),
        c("v1-b", "V1", 20_000, 2, .98),
        c("v1-c", "V1", 30_000, 3, .97),
        c("v2-a", "V2", 40_000, 4, .96),
        c("v3-a", "V3", 50_000, 5, .95),
    ])
    assert [item.frame_id for item in ordered] == [
        "v1-a", "v2-a", "v3-a", "v1-b", "v1-c",
    ]


def test_selector_is_deterministic_and_preserves_all_non_suppressed_candidates():
    values = [
        c("v2-b", "V2", 40_000, 4, .70),
        c("v1-a", "V1", 10_000, 1, .99),
        c("v2-a", "V2", 20_000, 2, .90),
        c("v1-b", "V1", 30_000, 3, .80),
    ]
    first = selector(window_ms=0).order(values)
    second = selector(window_ms=0).order(list(reversed(values)))
    assert first == second
    assert {item.frame_id for item in first} == {item.frame_id for item in values}
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
PYTHONPATH=src python -m pytest tests/orchestration/workflows/test_avs_coverage.py -v
```

Expected: FAIL because `AvsConfig` and `hcmai.orchestration.workflows.avs` do not exist.

- [ ] **Step 3: Add the server-side AVS configuration**

Add immediately before `SearchConfig` in `src/hcmai/common/config.py`:

```python
class AvsConfig(BaseModel):
    """Server-owned parameters for latency-first AVS candidate exposure."""

    model_config = ConfigDict(extra="forbid")

    candidate_pool_size: int = Field(default=500, ge=2)
    default_page_size: int = Field(default=80, ge=1)
    maximum_page_size: int = Field(default=100, ge=1)
    temporal_dedup_window_ms: int = Field(default=3000, ge=0)
    coverage_policy_version: Literal["video-pass-v1"] = "video-pass-v1"

    @model_validator(mode="after")
    def validate_pool_bounds(self) -> "AvsConfig":
        if self.default_page_size > self.maximum_page_size:
            raise ValueError("AVS default_page_size must not exceed maximum_page_size")
        if self.maximum_page_size >= self.candidate_pool_size:
            raise ValueError("AVS candidate_pool_size must be greater than maximum_page_size")
        return self
```

Add to `SearchConfig`:

```python
avs: AvsConfig = Field(default_factory=AvsConfig)
```

- [ ] **Step 4: Implement the pure selector**

Create `src/hcmai/orchestration/workflows/avs.py` with this public shape:

```python
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from hcmai.common.config import AvsConfig


@dataclass(frozen=True, slots=True)
class AvsCoverageCandidate:
    frame_id: str
    video_id: str
    timestamp_ms: int
    retrieval_rank: int
    retrieval_score: float | None = None

    def __post_init__(self) -> None:
        if not self.frame_id.strip() or not self.video_id.strip():
            raise ValueError("AVS candidate identity must be non-empty")
        if self.timestamp_ms < 0:
            raise ValueError("AVS timestamp_ms must be non-negative")
        if self.retrieval_rank < 1:
            raise ValueError("AVS retrieval_rank must be one-based")


class AvsCoverageSelector:
    def __init__(self, config: AvsConfig) -> None:
        self.config = config

    def order(self, candidates: Sequence[AvsCoverageCandidate]) -> list[AvsCoverageCandidate]:
        ranked = sorted(candidates, key=lambda item: (item.retrieval_rank, item.frame_id))
        kept_by_video: dict[str, list[AvsCoverageCandidate]] = defaultdict(list)

        for candidate in ranked:
            if any(
                abs(candidate.timestamp_ms - kept.timestamp_ms)
                <= self.config.temporal_dedup_window_ms
                for kept in kept_by_video[candidate.video_id]
            ):
                continue
            kept_by_video[candidate.video_id].append(candidate)

        ordered: list[AvsCoverageCandidate] = []
        depth = 0
        while True:
            pass_items = [
                items[depth]
                for items in kept_by_video.values()
                if depth < len(items)
            ]
            if not pass_items:
                break
            ordered.extend(sorted(pass_items, key=lambda item: (item.retrieval_rank, item.frame_id)))
            depth += 1
        return ordered
```

- [ ] **Step 5: Run selector tests**

Run:

```bash
PYTHONPATH=src python -m pytest tests/orchestration/workflows/test_avs_coverage.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit in the real git checkout**

```bash
git add src/hcmai/common/config.py src/hcmai/orchestration/workflows/avs.py tests/orchestration/workflows/test_avs_coverage.py
git commit -m "feat: add deterministic AVS coverage selector"
```

---

### Task 2: Expose direct text retrieval from the standalone retrieval process

**Files:**
- Modify: `src/hcmai/retrieval/serving/schemas.py:13-161`
- Modify: `src/hcmai/retrieval/serving/runtime.py:52-172, 186+`
- Modify: `src/hcmai/retrieval/serving/server.py:20-31, 97+`
- Modify: `src/hcmai/retrieval/serving/client.py:22-30, 47+`
- Create: `tests/retrieval/serving/test_text_search.py`

**Interfaces:**
- Produces wire schema: `TextSearchRequestSchema(query: str, top_k: int)`.
- Produces wire schema: `TextCandidateSchema(frame_id: str, rank: int, score: float | None)`.
- Produces wire schema: `TextSearchCandidatesSchema(candidates, retrieval_ms, warnings)`.
- Produces runtime method: `RetrievalRuntime.search_text(query: str, *, top_k: int) -> RetrievalResult`.
- Produces client method: `RetrievalHttpClient.search_text(query: str, *, top_k: int) -> RemoteTextSearchResult`.
- No temporal/KIS method is involved in this call path.

- [ ] **Step 1: Write failing direct-text retrieval tests**

```python
# tests/retrieval/serving/test_text_search.py
from fastapi.testclient import TestClient

from hcmai.common.observability.models import RetrievalTrace
from hcmai.retrieval.models import RetrievalCandidate, RetrievalResult, RetrievalSource
from hcmai.retrieval.serving.server import create_app


class FakeRuntime:
    def __init__(self):
        self.search_text_calls = []

    def search_text(self, query: str, *, top_k: int) -> RetrievalResult:
        self.search_text_calls.append((query, top_k))
        return RetrievalResult(
            candidates=[
                RetrievalCandidate(
                    frame_id="f1",
                    source_scores={RetrievalSource.VISUAL: .8},
                    source_ranks={RetrievalSource.VISUAL: 1},
                    fusion_score=.02,
                ),
                RetrievalCandidate(
                    frame_id="f2",
                    source_scores={RetrievalSource.VISUAL: .7},
                    source_ranks={RetrievalSource.VISUAL: 2},
                    fusion_score=.01,
                ),
            ],
            trace=RetrievalTrace(),
            warnings=["context unavailable"],
        )


def test_search_text_endpoint_preserves_retrieval_order_and_rank():
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime))

    response = client.post("/search_text", json={"query": "seafood", "top_k": 50})

    assert response.status_code == 200
    assert runtime.search_text_calls == [("seafood", 50)]
    assert response.json()["candidates"] == [
        {"frame_id": "f1", "rank": 1, "score": .02},
        {"frame_id": "f2", "rank": 2, "score": .01},
    ]
    assert response.json()["warnings"] == ["context unavailable"]
```

Add this client-mapping test in the same module so the wire contract and the main-backend mapping fail together before implementation:

```python
import httpx

from hcmai.retrieval.serving.client import RetrievalHttpClient
from hcmai.retrieval.serving.utils.config import RetrievalClientSettings


class StubHttpClient:
    def post(self, path: str, *, json: dict):
        assert path == "/search_text"
        assert json == {"query": "seafood", "top_k": 50}
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"frame_id": "f1", "rank": 1, "score": .02},
                    {"frame_id": "f2", "rank": 2, "score": .01},
                ],
                "retrieval_ms": 12.5,
                "warnings": ["context unavailable"],
            },
            request=httpx.Request("POST", "http://127.0.0.1:8002/search_text"),
        )

    def close(self) -> None:
        return None


def test_search_text_client_maps_rank_order_and_diagnostics():
    client = RetrievalHttpClient(RetrievalClientSettings(target="127.0.0.1:8002"))
    client._client.close()
    client._client = StubHttpClient()

    result = client.search_text("seafood", top_k=50)

    assert [(item.frame_id, item.rank, item.score) for item in result.candidates] == [
        ("f1", 1, .02),
        ("f2", 2, .01),
    ]
    assert result.retrieval_ms == 12.5
    assert result.warnings == ("context unavailable",)
```

- [ ] **Step 2: Run the test and verify the endpoint/client are missing**

Run:

```bash
PYTHONPATH=src python -m pytest tests/retrieval/serving/test_text_search.py -v
```

Expected: FAIL because text-search schemas, endpoint, runtime method, and client mapping do not exist.

- [ ] **Step 3: Add the text-search wire schemas**

Add to `src/hcmai/retrieval/serving/schemas.py`:

```python
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TextSearchRequestSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: NonBlank
    top_k: int = Field(default=100, ge=1)


class TextCandidateSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_id: NonBlank
    rank: int = Field(ge=1)
    score: float | None = None


class TextSearchCandidatesSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: list[TextCandidateSchema]
    retrieval_ms: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Retain the already-loaded `RetrievalService` in `RetrievalRuntime`**

Add a `retrieval` field to the runtime dataclass and pass the existing `retrieval` local created at `runtime.py:91` as the `retrieval=retrieval` keyword in the existing `return cls(...)` constructor call. Add:

```python
from hcmai.retrieval.models import RetrievalResult
from hcmai.retrieval.retriever.pipeline import RetrievalService

# dataclass field
retrieval: RetrievalService

# method
def search_text(self, query: str, *, top_k: int) -> RetrievalResult:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not query.strip():
        raise ValueError("query must not be blank")
    return self.retrieval.search(query.strip(), top_k=top_k)
```

This reuses the already loaded indexes/encoders; it must not load a second `RetrievalService`.

- [ ] **Step 5: Add `/search_text` on the retrieval server**

Import the new schemas and add before `/search_image`:

```python
@app.post("/search_text", response_model=TextSearchCandidatesSchema)
def search_text(req: TextSearchRequestSchema) -> TextSearchCandidatesSchema:
    rt = _get_runtime()
    try:
        result = rt.search_text(req.query, top_k=req.top_k)
        candidates = []
        for rank, candidate in enumerate(result.candidates, start=1):
            score = candidate.fusion_score
            if score is None and candidate.source_scores:
                score = max(candidate.source_scores.values())
            candidates.append(TextCandidateSchema(
                frame_id=candidate.frame_id,
                rank=rank,
                score=score,
            ))
        return TextSearchCandidatesSchema(
            candidates=candidates,
            retrieval_ms=result.trace.total_duration_ms,
            warnings=list(result.warnings),
        )
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except Exception as err:
        logger.exception("Error in /search_text: %s", err)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err)) from err
```

`rank` is the authoritative AVS relevance ordering; `score` is diagnostic only.

- [ ] **Step 6: Add main-backend HTTP client mapping**

In `client.py`, add:

```python
@dataclass(frozen=True, slots=True)
class RemoteTextCandidate:
    frame_id: str
    rank: int
    score: float | None


@dataclass(frozen=True, slots=True)
class RemoteTextSearchResult:
    candidates: tuple[RemoteTextCandidate, ...]
    retrieval_ms: float
    warnings: tuple[str, ...]


def search_text(self, query: str, *, top_k: int = 100) -> RemoteTextSearchResult:
    req = TextSearchRequestSchema(query=query, top_k=top_k)
    try:
        resp = self._client.post("/search_text", json=req.model_dump())
        if resp.status_code != 200:
            raise map_http_error(resp.status_code, resp.text)
        data = TextSearchCandidatesSchema.model_validate(resp.json())
        return RemoteTextSearchResult(
            candidates=tuple(
                RemoteTextCandidate(frame_id=item.frame_id, rank=item.rank, score=item.score)
                for item in data.candidates
            ),
            retrieval_ms=data.retrieval_ms,
            warnings=tuple(data.warnings),
        )
    except httpx.RequestError as err:
        raise RetrievalUnavailableError(
            f"Failed to search_text at {self.base_url}: {err}"
        ) from err
```

- [ ] **Step 7: Run direct-text retrieval tests**

Run:

```bash
PYTHONPATH=src python -m pytest tests/retrieval/serving/test_text_search.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit in the real git checkout**

```bash
git add src/hcmai/retrieval/serving tests/retrieval/serving/test_text_search.py
git commit -m "feat: expose direct text retrieval service"
```

---

### Task 3: Add the AVS search contract, workflow, and public router

**Files:**
- Create: `src/hcmai/api/contracts/avs.py`
- Modify: `src/hcmai/api/contracts/__init__.py`
- Create: `src/hcmai/orchestration/workflows/avs_search.py`
- Modify: `src/hcmai/orchestration/pipeline.py:101-230, 690+`
- Create: `src/hcmai/api/routers/avs.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/app.py:20-35, 198-207`
- Create: `tests/orchestration/workflows/test_avs_search.py`
- Create: `tests/api/test_avs_router.py`

**Interfaces:**
- Consumes: `RetrievalHttpClient.search_text()` from Task 2.
- Consumes: `AvsCoverageSelector` from Task 1.
- Produces: `AvsSearchRequest`, `AvsSearchResult`, `AvsSearchLatency`, `AvsSearchResponse`.
- Produces: `AvsSearchService.search(request: AvsSearchRequest) -> AvsSearchResponse`.
- Produces: `SearchService.search_avs(request: AvsSearchRequest) -> AvsSearchResponse`.
- Produces HTTP: `POST /api/v1/avs/search`.

- [ ] **Step 1: Write failing AVS workflow tests**

Create `tests/orchestration/workflows/test_avs_search.py` with a minimal canonical corpus and a gateway that exposes only the direct text-search interface from Task 2:

```python
from hcmai.api.contracts.avs import AvsSearchRequest
from hcmai.common.config import AvsConfig
from hcmai.corpus.models import Frame
from hcmai.orchestration.workflows.avs_search import AvsSearchService
from hcmai.retrieval.serving.client import RemoteTextCandidate, RemoteTextSearchResult


class FakeCorpus:
    def __init__(self) -> None:
        self.frames = {
            "f1": Frame("f1", "V1", 10, 10_000, "/f1.jpg", fps=25.0),
            "f2": Frame("f2", "V1", 11, 11_000, "/f2.jpg", fps=25.0),
            "f3": Frame("f3", "V1", 30, 30_000, "/f3.jpg", fps=25.0),
            "f4": Frame("f4", "V2", 40, 40_000, "/f4.jpg", fps=25.0),
            "f5": Frame("f5", "V3", 50, 50_000, "/f5.jpg", fps=25.0),
        }

    def frame(self, frame_id: str) -> Frame:
        return self.frames[frame_id]

    def title(self, video_id: str):
        return f"title-{video_id}"

    def caption(self, frame_id: str):
        return None

    def ocr(self, frame_id: str):
        return None

    def objects(self, frame_id: str):
        return ()

    def transcript(self, video_id: str, start_ms: int, end_ms: int):
        return None


class FakeTextGateway:
    def __init__(self) -> None:
        self.calls = []

    def search_text(self, query: str, *, top_k: int) -> RemoteTextSearchResult:
        self.calls.append((query, top_k))
        return RemoteTextSearchResult(
            candidates=(
                RemoteTextCandidate("f1", 1, .99),
                RemoteTextCandidate("f2", 2, .98),
                RemoteTextCandidate("f3", 3, .97),
                RemoteTextCandidate("f4", 4, .96),
                RemoteTextCandidate("f5", 5, .95),
            ),
            retrieval_ms=8.0,
            warnings=(),
        )


def test_avs_search_uses_direct_retrieval_then_coverage():
    gateway = FakeTextGateway()
    service = AvsSearchService(
        corpus=FakeCorpus(),
        retrieval=gateway,
        config=AvsConfig(
            candidate_pool_size=500,
            default_page_size=3,
            maximum_page_size=100,
            temporal_dedup_window_ms=3000,
        ),
    )

    response = service.search(AvsSearchRequest(query="seafood", page_size=3))

    assert gateway.calls == [("seafood", 500)]
    assert [item.video_id for item in response.results] == ["V1", "V2", "V3"]
    assert [item.candidate_id for item in response.results] == ["f1", "f4", "f5"]
    assert response.candidate_pool_size == 5
    assert response.deduplicated_candidate_count == 4
    assert response.unique_videos == 3
```

This fake gateway intentionally has no KIS, temporal-search, EventTrail, or reranker methods. If `AvsSearchService` reaches for any of those interfaces, this test fails structurally. Task 11 adds the higher-level `SearchService.search_avs()` sentinel regression.

- [ ] **Step 2: Write a failing router contract test**

```python
# tests/api/test_avs_router.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hcmai.api.contracts.avs import (
    AvsSearchLatency,
    AvsSearchResponse,
    AvsSearchResult,
)
from hcmai.api.contracts.search import SearchResultMetadata
from hcmai.api.routers.avs import create_avs_router


class FakeSearchService:
    def search_avs(self, request):
        assert request.query == "seafood"
        assert request.page_size == 80
        return AvsSearchResponse(
            results=[AvsSearchResult(
                candidate_id="f1",
                frame_id="f1",
                video_id="V1",
                frame_idx=10,
                timestamp_ms=1000,
                fps=25.0,
                retrieval_rank=1,
                retrieval_score=.9,
                metadata=SearchResultMetadata(),
            )],
            latency=AvsSearchLatency(
                retrieval_ms=1.0,
                coverage_ms=.1,
                materialization_ms=.1,
                total_ms=1.2,
            ),
            candidate_pool_size=5,
            deduplicated_candidate_count=4,
            unique_videos=1,
            warnings=[],
        )


def test_avs_router_returns_dedicated_contract():
    app = FastAPI()
    app.include_router(create_avs_router({"service": FakeSearchService()}))

    response = TestClient(app).post(
        "/api/v1/avs/search",
        json={"query": "seafood", "page_size": 80},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["candidate_id"] == "f1"
    assert response.json()["candidate_pool_size"] == 5
```

- [ ] **Step 3: Run the new tests and verify the AVS API is absent**

```bash
PYTHONPATH=src python -m pytest \
  tests/orchestration/workflows/test_avs_search.py \
  tests/api/test_avs_router.py -v
```

Expected: FAIL on missing AVS modules/contracts.

- [ ] **Step 4: Add public AVS contracts**

Create `src/hcmai/api/contracts/avs.py`:

```python
from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .search import SearchResultMetadata

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AvsSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: NonBlank
    page_size: int = Field(default=80, ge=1)


class AvsSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: NonBlank
    frame_id: NonBlank
    video_id: NonBlank
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    fps: float | None = Field(default=None, gt=0)
    retrieval_rank: int = Field(ge=1)
    retrieval_score: float | None = None
    metadata: SearchResultMetadata


class AvsSearchLatency(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retrieval_ms: float = Field(ge=0)
    coverage_ms: float = Field(ge=0)
    materialization_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class AvsSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[AvsSearchResult]
    latency: AvsSearchLatency
    candidate_pool_size: int = Field(ge=0)
    deduplicated_candidate_count: int = Field(ge=0)
    unique_videos: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
```

Export these from `api/contracts/__init__.py`.

- [ ] **Step 5: Implement the AVS search workflow**

Create `src/hcmai/orchestration/workflows/avs_search.py` with constructor:

```python
class AvsSearchService:
    def __init__(
        self,
        *,
        corpus: Corpus,
        retrieval: RetrievalHttpClient,
        config: AvsConfig,
    ) -> None:
        self.corpus = corpus
        self.retrieval = retrieval
        self.config = config
        self.selector = AvsCoverageSelector(config)
        self.materializer = SearchMaterializer(corpus)
```

`search()` must:

```python
def search(self, request: AvsSearchRequest) -> AvsSearchResponse:
    if request.page_size > self.config.maximum_page_size:
        raise InvalidQueryInputError(
            f"page_size must be <= {self.config.maximum_page_size}"
        )

    started = perf_counter()
    remote = self.retrieval.search_text(
        request.query,
        top_k=self.config.candidate_pool_size,
    )

    canonical = []
    for item in remote.candidates:
        frame = self.corpus.frame(item.frame_id)
        canonical.append(AvsCoverageCandidate(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            timestamp_ms=frame.timestamp_ms,
            retrieval_rank=item.rank,
            retrieval_score=item.score,
        ))

    coverage_started = perf_counter()
    ordered = self.selector.order(canonical)
    coverage_ms = (perf_counter() - coverage_started) * 1000

    materialization_started = perf_counter()
    visible = ordered[:request.page_size]
    results = []
    for candidate in visible:
        frame = self.corpus.frame(candidate.frame_id)
        results.append(AvsSearchResult(
            candidate_id=frame.frame_id,
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            timestamp_ms=frame.timestamp_ms,
            fps=frame.fps,
            retrieval_rank=candidate.retrieval_rank,
            retrieval_score=candidate.retrieval_score,
            metadata=self.materializer.build_frame_metadata(frame),
        ))
    materialization_ms = (perf_counter() - materialization_started) * 1000

    return AvsSearchResponse(
        results=results,
        latency=AvsSearchLatency(
            retrieval_ms=remote.retrieval_ms,
            coverage_ms=coverage_ms,
            materialization_ms=materialization_ms,
            total_ms=(perf_counter() - started) * 1000,
        ),
        candidate_pool_size=len(remote.candidates),
        deduplicated_candidate_count=len(ordered),
        unique_videos=len({item.video_id for item in results}),
        warnings=list(remote.warnings),
    )
```

Do not add fallback to KIS if `search_text()` fails.

- [ ] **Step 6: Wire AVS into `SearchService` without branching inside `KISPipeline`**

In `SearchService.__init__`:

```python
self.avs = (
    AvsSearchService(
        corpus=corpus,
        retrieval=remote_retrieval,
        config=self.config.avs,
    )
    if corpus is not None and remote_retrieval is not None
    else None
)
```

Add:

```python
def search_avs(self, request: AvsSearchRequest) -> AvsSearchResponse:
    if self.avs is None:
        raise SearchServiceUnavailableError("AVS direct retrieval is unavailable")
    try:
        return self.avs.search(request)
    except RetrievalUnavailableError as error:
        raise SearchServiceUnavailableError(str(error)) from error
    except RetrievalInvalidRequestError as error:
        raise InvalidQueryInputError(str(error)) from error
    except (RetrievalProtocolError, RetrievalClientError) as error:
        raise SearchServiceGatewayError(str(error)) from error
```

This method must not call `_resolve_operation()` or `self.kis`.

- [ ] **Step 7: Add and register `/api/v1/avs/search`**

Create `api/routers/avs.py` with `run_in_threadpool(service.search_avs, request)`, map `InvalidQueryInputError` to HTTP 422, `SearchServiceUnavailableError` to 503, and `SearchServiceGatewayError` to 502. Register `create_avs_router` in `api/routers/__init__.py` and `app.py`.

- [ ] **Step 8: Run AVS workflow/router tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/orchestration/workflows/test_avs_search.py \
  tests/api/test_avs_router.py -v
```

Expected: PASS; the AVS workflow test uses a gateway that exposes only `search_text()`, so no KIS/DP/EventTrail interface is available on that path.

- [ ] **Step 9: Commit in the real git checkout**

```bash
git add src/hcmai/api/contracts src/hcmai/api/routers src/hcmai/orchestration src/hcmai/app.py tests/api/test_avs_router.py tests/orchestration/workflows/test_avs_search.py
git commit -m "feat: add direct AVS search API"
```

---

### Task 4: Log only the coverage-ordered AVS result list to DRES

**Files:**
- Create: `src/hcmai/api/result_logging.py`
- Modify: `src/hcmai/api/routers/search.py:184-242`
- Modify: `src/hcmai/api/routers/avs.py`
- Create: `tests/api/test_result_logging.py`

**Interfaces:**
- Produces: `record_dres_result_log(service_container, response, *, user_id, category, event_value, results, rank_offset=0) -> None`.
- Existing image/filter logging and new AVS logging consume the same helper.

- [ ] **Step 1: Write a failing result-logging test**

Use a fake connected DRES service whose `log_results()` records the payload. Call `record_dres_result_log()` with AVS results already ordered `[V2@2s, V1@1s]` and assert the `RankedAnswer.rank` values are `1, 2` in exactly that order. Add a failure fixture where `log_results()` raises and assert retrieval remains successful while `X-DRES-Log-Status == "failed"`.

- [ ] **Step 2: Run the test and verify the shared helper is absent**

```bash
PYTHONPATH=src python -m pytest tests/api/test_result_logging.py -v
```

Expected: FAIL because `hcmai.api.result_logging` does not exist.

- [ ] **Step 3: Move the existing helper out of `routers/search.py`**

Move `_record_dres_result_log()` and `_now_ms()` unchanged in behavior to `src/hcmai/api/result_logging.py`, rename the public helper to `record_dres_result_log`, and update image/filter routes to import it. Do not alter best-effort semantics.

- [ ] **Step 4: Log AVS results after coverage ordering**

In `api/routers/avs.py`, accept:

```python
user_id: Annotated[str | None, Header(alias="X-VBS-User-ID")] = None
```

After `service.search_avs()` succeeds, call:

```python
await record_dres_result_log(
    service_container,
    http_response,
    user_id=user_id,
    category="TEXT",
    event_value=request.query,
    results=result.results,
)
```

Pass `result.results`, never the raw pre-dedup retrieval candidates.

- [ ] **Step 5: Run logging and existing search-router tests**

```bash
PYTHONPATH=src python -m pytest tests/api/test_result_logging.py tests/api/test_avs_router.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit in the real git checkout**

```bash
git add src/hcmai/api/result_logging.py src/hcmai/api/routers/search.py src/hcmai/api/routers/avs.py tests/api
git commit -m "refactor: share DRES result logging with AVS"
```

---

### Task 5: Change direct submission to a task-aware answer list while preserving singular KIS/VQA callers

**Files:**
- Modify: `src/hcmai/api/contracts/vbs.py:108-118`
- Modify: `src/hcmai/api/routers/vbs.py:159-268`
- Create: `tests/api/test_vbs_submission.py`
- Modify: `frontend/src/api/submissions.js:40-150`
- Create: `frontend/src/api/submissions.test.js`

**Interfaces:**
- Backend request becomes `VbsDirectSubmissionRequest.answers: list[VbsAnswer]` with at least one item.
- Produces frontend `submitDresAnswers(options)` where `options.answers` is the complete answer array.
- Preserves frontend `submitDresAnswer(options)` as a wrapper that forwards a one-item `answers` array to `submitDresAnswers`.
- KIS/TRAKE/generic temporal tasks remain exactly one temporal answer; VQA remains exactly one text answer; AVS allows one-or-more temporal answers.

- [ ] **Step 1: Write failing backend submission policy tests**

Create `tests/api/test_vbs_submission.py` with a real FastAPI router fixture and a fake DRES service. These tests define both cardinality and kind rules before the contract changes:

```python
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hcmai.api.routers.vbs import create_vbs_router
from hcmai.vbs.models import ApiClientAnswer, DresSubmissionStatus


class FakeDresService:
    def __init__(self, *, task_group: str, task_type: str, scope_key: str = "scope-1") -> None:
        self.scope = SimpleNamespace(
            evaluation_id="eval-1",
            task_scope_key=scope_key,
            task_name="Task 1",
            task_group=task_group,
            task_type=task_type,
            duration=300,
        )
        self.submit = AsyncMock(
            return_value=DresSubmissionStatus(
                status=True,
                submission="CORRECT",
                description="recorded",
            )
        )

    def session_status(self, user_id: str) -> dict[str, object]:
        return {"user_id": user_id, "connected": True}

    async def resolve_scope(self, user_id: str, *, evaluation_id=None, task_name=None):
        return self.scope

    def temporal_range_answer(self, video_id: str, start_ms: int, end_ms: int) -> ApiClientAnswer:
        return ApiClientAnswer(
            media_item_name=video_id,
            start=start_ms,
            end=end_ms,
        )


def make_client(*, task_group: str, task_type: str, scope_key: str = "scope-1"):
    service = FakeDresService(
        task_group=task_group,
        task_type=task_type,
        scope_key=scope_key,
    )
    app = FastAPI()
    app.include_router(create_vbs_router({"vbs_service": service}))
    return TestClient(app), service


def temporal(video_id: str, timestamp_ms: int) -> dict[str, object]:
    return {
        "kind": "TEMPORAL",
        "video_id": video_id,
        "start_ms": timestamp_ms,
        "end_ms": timestamp_ms,
    }


def submit_body(answers: list[dict[str, object]], *, scope_key: str = "scope-1") -> dict[str, object]:
    return {
        "user_id": "team-a",
        "expected_task_scope_key": scope_key,
        "answers": answers,
    }


def test_kis_accepts_exactly_one_temporal_answer():
    client, service = make_client(task_group="KIS", task_type="KNOWN ITEM SEARCH")

    response = client.post("/api/v1/vbs/submit", json=submit_body([temporal("V1", 1000)]))

    assert response.status_code == 200
    assert response.json()["state"] == "RECORDED"
    assert service.submit.await_count == 1
    payload = service.submit.await_args.args[2]
    assert len(payload.answer_sets) == 1
    assert len(payload.answer_sets[0].answers) == 1


def test_kis_rejects_multiple_answers_before_dres_call():
    client, service = make_client(task_group="KIS", task_type="KNOWN ITEM SEARCH")

    response = client.post(
        "/api/v1/vbs/submit",
        json=submit_body([temporal("V1", 1000), temporal("V2", 2000)]),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ANSWER_KIND_MISMATCH"
    service.submit.assert_not_awaited()


def test_avs_accepts_multiple_temporal_answers_in_one_answer_set():
    client, service = make_client(task_group="AVS", task_type="AD-HOC VIDEO SEARCH")
    answers = [temporal(f"V{i}", i * 1000) for i in range(1, 21)]

    response = client.post("/api/v1/vbs/submit", json=submit_body(answers))

    assert response.status_code == 200
    assert service.submit.await_count == 1
    payload = service.submit.await_args.args[2]
    assert len(payload.answer_sets) == 1
    assert payload.answer_sets[0].task_name == "Task 1"
    assert len(payload.answer_sets[0].answers) == 20


def test_avs_rejects_text_answer_before_dres_call():
    client, service = make_client(task_group="AVS", task_type="AVS")

    response = client.post(
        "/api/v1/vbs/submit",
        json=submit_body([{"kind": "TEXT", "text": "not temporal"}]),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ANSWER_KIND_MISMATCH"
    service.submit.assert_not_awaited()


def test_vqa_accepts_exactly_one_text_answer():
    client, service = make_client(task_group="QA", task_type="VQA")

    response = client.post(
        "/api/v1/vbs/submit",
        json=submit_body([{"kind": "TEXT", "text": "blue"}]),
    )

    assert response.status_code == 200
    assert service.submit.await_count == 1
    payload = service.submit.await_args.args[2]
    assert payload.answer_sets[0].answers[0].text == "blue"


def test_vqa_rejects_multiple_answers_before_dres_call():
    client, service = make_client(task_group="QA", task_type="VQA")

    response = client.post(
        "/api/v1/vbs/submit",
        json=submit_body([
            {"kind": "TEXT", "text": "blue"},
            {"kind": "TEXT", "text": "green"},
        ]),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ANSWER_KIND_MISMATCH"
    service.submit.assert_not_awaited()


def test_stale_scope_is_rejected_before_dres_call():
    client, service = make_client(task_group="AVS", task_type="AVS", scope_key="live-scope")

    response = client.post(
        "/api/v1/vbs/submit",
        json=submit_body([temporal("V1", 1000)], scope_key="stale-scope"),
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "TASK_SCOPE_MISMATCH"
    service.submit.assert_not_awaited()
```

- [ ] **Step 2: Write failing frontend transport tests**

Create `frontend/src/api/submissions.test.js` with an explicit successful DRES response fixture:

```javascript
import { submitDresAnswer, submitDresAnswers } from './submissions';

const recordedResponse = () => ({
  ok: true,
  status: 200,
  headers: { get: () => null },
  json: async () => ({
    state: 'RECORDED',
    recorded: true,
    verdict: 'CORRECT',
    message: 'DRES recorded the answer batch',
  }),
});

describe('submission transport', () => {
  beforeEach(() => {
    global.fetch = jest.fn().mockResolvedValue(recordedResponse());
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  test('posts AVS answers as one answers array', async () => {
    await submitDresAnswers({
      userId: 'team-a',
      expectedTaskScopeKey: 'scope-1',
      evaluationId: 'eval-1',
      taskName: 'AVS task',
      answers: [
        { kind: 'TEMPORAL', video_id: 'V1', start_ms: 1000, end_ms: 1000 },
        { kind: 'TEMPORAL', video_id: 'V2', start_ms: 2000, end_ms: 2000 },
      ],
    });

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const body = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(body.expected_task_scope_key).toBe('scope-1');
    expect(body.answers).toHaveLength(2);
    expect(body).not.toHaveProperty('answer');
  });

  test('singular wrapper still sends a one-item answers array', async () => {
    await submitDresAnswer({
      userId: 'team-a',
      expectedTaskScopeKey: 'scope-1',
      answer: { kind: 'TEXT', text: 'answer' },
    });

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const body = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(body.answers).toEqual([{ kind: 'TEXT', text: 'answer' }]);
    expect(body).not.toHaveProperty('answer');
  });
});
```

- [ ] **Step 3: Run backend and frontend focused tests; verify current one-answer contract fails them**

```bash
PYTHONPATH=src python -m pytest tests/api/test_vbs_submission.py -v
cd frontend && npm test -- --watchAll=false --runTestsByPath src/api/submissions.test.js
```

Expected: FAIL because the backend expects `answer` and the frontend has no batch function.

- [ ] **Step 4: Change the Pydantic request to `answers`**

Replace the singular field with:

```python
class VbsDirectSubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank
    expected_task_scope_key: NonBlank
    answers: list[VbsAnswer] = Field(min_length=1)
    evaluation_id: NonBlank | None = None
    task_name: NonBlank | None = None
```

Update the module docstring from “one-answer” to “task-aware direct submission”.

- [ ] **Step 5: Add explicit task-family classification in the VBS router**

Add a private helper whose returned family is one of `"VQA"`, `"AVS"`, `"TEMPORAL_SINGLE"`, or `None`:

```python
def _task_family(scope: Any) -> str | None:
    group = scope.task_group.strip().upper()
    task_type = scope.task_type.strip().upper()
    if task_type == "VQA" or group in {"VQA", "QA"} or "QUESTION ANSWERING" in task_type:
        return "VQA"
    if task_type == "AVS" or group == "AVS" or "AD-HOC" in task_type or "ADHOC" in task_type:
        return "AVS"
    if (
        task_type in {"KIS", "TRAKE"}
        or group in {"KIS", "TRAKE"}
        or "KNOWN ITEM" in task_type
        or "VIDEO SEARCH" in task_type
    ):
        return "TEMPORAL_SINGLE"
    return None
```

Validate counts/kinds before mapping any answer:

```python
family = _task_family(scope)
if family is None:
    _raise_api_error(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "UNSUPPORTED_DRES_TASK_TYPE",
        "The active DRES task type is not supported for direct submission",
    )
if family == "VQA" and (len(data.answers) != 1 or data.answers[0].kind != "TEXT"):
    _raise_api_error(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "ANSWER_KIND_MISMATCH",
        "The active VQA task requires exactly one text answer",
    )
if family == "TEMPORAL_SINGLE" and (
    len(data.answers) != 1 or data.answers[0].kind != "TEMPORAL"
):
    _raise_api_error(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "ANSWER_KIND_MISMATCH",
        "The active KIS/TRAKE task requires exactly one temporal answer",
    )
if family == "AVS" and any(answer.kind != "TEMPORAL" for answer in data.answers):
    _raise_api_error(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "ANSWER_KIND_MISMATCH",
        "The active AVS task requires one or more temporal answers",
    )
```

- [ ] **Step 6: Map all validated answers and submit exactly one DRES answer set**

```python
mapped_answers = []
for answer in data.answers:
    if answer.kind == "TEMPORAL":
        mapped_answers.append(service.temporal_range_answer(
            answer.video_id,
            answer.start_ms,
            answer.end_ms,
        ))
    else:
        mapped_answers.append(ApiClientAnswer(text=answer.text))

payload = ApiClientSubmission(
    answer_sets=[ApiClientAnswerSet(
        task_name=scope.task_name,
        answers=mapped_answers,
    )],
)
```

Keep the existing `RECORDED` / `NOT_RECORDED` / `UNKNOWN` mapping. Change success copy to “DRES recorded the answer batch” so it is correct for both one and many answers.

- [ ] **Step 7: Add frontend batch transport and singular wrapper**

Refactor `safeAnswer()` to validate one item and add:

```javascript
const safeAnswers = (answers) => {
  if (!Array.isArray(answers) || answers.length === 0) {
    throw new Error('At least one answer object is required');
  }
  return answers.map(safeAnswer);
};

export const submitDresAnswers = async ({
  userId: value,
  expectedTaskScopeKey,
  answers,
  evaluationId,
  taskName,
  signal,
} = {}) => {
  const userId = normalizeUserId(value);
  if (!nonBlank(expectedTaskScopeKey)) throw new Error('A DRES task scope key is required');
  const body = {
    user_id: userId,
    expected_task_scope_key: expectedTaskScopeKey,
    answers: safeAnswers(answers),
  };
  if (nonBlank(evaluationId)) body.evaluation_id = evaluationId.trim();
  if (nonBlank(taskName)) body.task_name = taskName.trim();
  const payload = await requestJson('/api/v1/vbs/submit', {
    method: 'POST', body, signal,
  });
  return safeOutcome(payload);
};

export const submitDresAnswer = ({ answer, ...rest } = {}) => (
  submitDresAnswers({ ...rest, answers: [answer] })
);
```

Existing `useDirectSubmission` does not need to change its public API.

- [ ] **Step 8: Run submission tests plus existing App submission test**

```bash
PYTHONPATH=src python -m pytest tests/api/test_vbs_submission.py -v
cd frontend && npm test -- --watchAll=false --runTestsByPath \
  src/api/submissions.test.js src/App.test.jsx
```

Expected: PASS; existing KIS one-answer App test now observes an `answers: [answer]` request body but still calls `submitDresAnswer()` once.

- [ ] **Step 9: Commit in the real git checkout**

```bash
git add src/hcmai/api/contracts/vbs.py src/hcmai/api/routers/vbs.py tests/api/test_vbs_submission.py frontend/src/api/submissions.js frontend/src/api/submissions.test.js frontend/src/App.test.jsx
git commit -m "feat: support task-aware multi-answer submission"
```

---

### Task 6: Add the AVS browser API and pure task-scoped selection reducer

**Files:**
- Create: `frontend/src/api/avs.js`
- Create: `frontend/src/api/avs.test.js`
- Create: `frontend/src/features/avs/selectionState.js`
- Create: `frontend/src/features/avs/selectionState.test.js`
- Create: `frontend/src/features/avs/index.js`

**Interfaces:**
- Produces: `searchAvs({ query, pageSize, userId, signal })`.
- Produces: `candidateToTemporalAnswer(candidate)`.
- Produces: `createInitialAvsSelectionState()` and `avsSelectionReducer(state, action)`.
- Selection reducer state contains `taskScopeKey`, `pending`, `submitted`, and `unknownBatch`.

- [ ] **Step 1: Write failing AVS API tests**

```javascript
// frontend/src/api/avs.test.js
import { searchAvs } from './avs';

test('posts AVS text search with optional VBS logging identity', async () => {
  global.fetch = jest.fn().mockResolvedValue(response({
    results: [{
      candidate_id: 'f1', frame_id: 'f1', video_id: 'V1', frame_idx: 1,
      timestamp_ms: 1000, fps: 25, retrieval_rank: 1, retrieval_score: .2,
      metadata: { title: null, caption: null, ocr: null, objects: [], asr: null },
    }],
    latency: { retrieval_ms: 10, coverage_ms: 1, materialization_ms: 2, total_ms: 13 },
    candidate_pool_size: 500,
    deduplicated_candidate_count: 300,
    unique_videos: 1,
    warnings: [],
  }));

  const result = await searchAvs({ query: 'seafood', pageSize: 80, userId: 'team-a' });
  expect(result.results[0].candidate_id).toBe('f1');
  expect(global.fetch.mock.calls[0][1].headers['X-VBS-User-ID']).toBe('team-a');
});
```

Also reject malformed responses whose `results` is not an array or whose candidate identity fields are missing.

- [ ] **Step 2: Write failing selection-reducer tests**

```javascript
// frontend/src/features/avs/selectionState.test.js
import {
  avsSelectionReducer,
  candidateToTemporalAnswer,
  createInitialAvsSelectionState,
} from './selectionState';

const candidate = (id, videoId = 'V1', timestampMs = 1000) => ({
  candidate_id: id,
  frame_id: id,
  video_id: videoId,
  timestamp_ms: timestampMs,
});

test('reselecting the same canonical frame does not duplicate pending state', () => {
  let state = createInitialAvsSelectionState();
  state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
  state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
  state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
  expect(state.pending.size).toBe(1);
});

test('recorded batch clears only pending items and marks them submitted', () => {
  let state = createInitialAvsSelectionState();
  state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
  state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
  state = avsSelectionReducer(state, { type: 'RECORDED', candidateIds: ['f1'] });
  expect(state.pending.size).toBe(0);
  expect(state.submitted.has('f1')).toBe(true);
});

test('unknown batch keeps pending and stores immutable retry snapshot', () => {
  let state = createInitialAvsSelectionState();
  state = avsSelectionReducer(state, { type: 'BIND_SCOPE', taskScopeKey: 'scope-1' });
  state = avsSelectionReducer(state, { type: 'SELECT', candidate: candidate('f1') });
  state = avsSelectionReducer(state, { type: 'UNKNOWN', candidateIds: ['f1'] });
  expect(state.pending.has('f1')).toBe(true);
  expect(state.unknownBatch).toEqual(['f1']);
});
```

- [ ] **Step 3: Run tests and verify modules are missing**

```bash
cd frontend && npm test -- --watchAll=false --runTestsByPath \
  src/api/avs.test.js src/features/avs/selectionState.test.js
```

Expected: FAIL.

- [ ] **Step 4: Implement `searchAvs()` response validation**

Create `frontend/src/api/avs.js` using `requestJson()`:

```javascript
export const searchAvs = async ({ query, pageSize = 80, userId, signal } = {}) => {
  const normalizedQuery = typeof query === 'string' ? query.trim() : '';
  if (!normalizedQuery) throw new Error('An AVS text query is required');
  if (!Number.isSafeInteger(pageSize) || pageSize < 1) throw new Error('AVS page size must be positive');

  const payload = await requestJson('/api/v1/avs/search', {
    method: 'POST',
    body: { query: normalizedQuery, page_size: pageSize },
    signal,
    headers: userId?.trim() ? { 'X-VBS-User-ID': userId.trim() } : {},
  });
  // Validate response object, results array, candidate_id/frame_id/video_id/timestamp_ms,
  // latency object, and integer counters before returning it.
  return payload;
};
```

The validation must reject `candidate_id !== frame_id`, because the v1 selection key is canonical frame identity.

- [ ] **Step 5: Implement the pure selection reducer**

Use `Map` for pending/submitted identity and copy maps on every mutation:

```javascript
export const createInitialAvsSelectionState = () => ({
  taskScopeKey: null,
  pending: new Map(),
  submitted: new Map(),
  unknownBatch: null,
});

export const candidateToTemporalAnswer = (candidate) => ({
  kind: 'TEMPORAL',
  video_id: candidate.video_id,
  start_ms: candidate.timestamp_ms,
  end_ms: candidate.timestamp_ms,
});
```

Reducer actions must be exactly:

```text
BIND_SCOPE         -> bind empty state to one scope; same scope is a no-op
SELECT             -> add candidate by candidate_id unless already submitted
DESELECT           -> remove one pending candidate
CLEAR_PENDING      -> clear pending and unknownBatch
RECORDED           -> move listed candidate IDs from pending to submitted; clear unknownBatch
UNKNOWN            -> keep pending and set unknownBatch to an array snapshot of candidate IDs
MARK_UNKNOWN_RECORDED -> move unknownBatch IDs to submitted without network replay
RESET_FOR_SCOPE    -> new empty state bound to explicitly confirmed new scope
```

`SELECT` must throw or ignore attempts whose candidate lacks canonical identity/video/timestamp.

- [ ] **Step 6: Run API/reducer tests**

```bash
cd frontend && npm test -- --watchAll=false --runTestsByPath \
  src/api/avs.test.js src/features/avs/selectionState.test.js
```

Expected: PASS.

- [ ] **Step 7: Commit in the real git checkout**

```bash
git add frontend/src/api/avs.js frontend/src/api/avs.test.js frontend/src/features/avs
git commit -m "feat: add AVS browser API and basket state"
```

---

### Task 7: Extract a reusable DRES task selector and route AVS tasks to a dedicated workspace shell

**Files:**
- Create: `frontend/src/features/vbs/taskFamily.js`
- Create: `frontend/src/features/vbs/taskFamily.test.js`
- Create: `frontend/src/features/vbs/components/VbsTaskSelector.jsx`
- Create: `frontend/src/features/vbs/components/VbsTaskSelector.test.jsx`
- Modify: `frontend/src/features/search-controls/components/ToolBox.jsx:37-80, 234-263`
- Create: `frontend/src/features/avs/components/AvsQueryControls.jsx`
- Create: `frontend/src/features/avs/components/AvsWorkspace.jsx`
- Create: `frontend/src/features/avs/components/AvsWorkspace.test.jsx`
- Modify: `frontend/src/features/avs/index.js`
- Modify: `frontend/src/App.jsx:1-120, main workspace branch`
- Modify: `frontend/src/App.test.jsx`
- Create: `frontend/src/App.avs.test.jsx`

**Interfaces:**
- Produces: `taskFamily(task) -> 'AVS' | 'VQA' | 'KIS' | 'TRAKE' | 'OTHER'`.
- Produces: `<VbsTaskSelector onRequestChange(nextTask) />`.
- Produces the initial `<AvsWorkspace />` search shell.
- `App` routes only AVS-family tasks to AVS; all other task families retain the existing `SearchWorkspace` path in this plan.

- [ ] **Step 1: Write task-family tests matching backend semantics**

Create `frontend/src/features/vbs/taskFamily.test.js`:

```javascript
import { taskFamily } from './taskFamily';

describe('taskFamily', () => {
  test.each([
    [{ taskGroup: 'AVS', taskType: 'VIDEO SEARCH' }, 'AVS'],
    [{ taskGroup: 'misc', taskType: 'AD-HOC VIDEO SEARCH' }, 'AVS'],
    [{ taskGroup: 'misc', taskType: 'ADHOC VIDEO SEARCH' }, 'AVS'],
    [{ taskGroup: 'KIS', taskType: 'KNOWN ITEM SEARCH' }, 'KIS'],
    [{ taskGroup: 'TRAKE', taskType: 'TRAKE' }, 'TRAKE'],
    [{ taskGroup: 'QA', taskType: 'VQA' }, 'VQA'],
    [{ taskGroup: 'misc', taskType: 'something else' }, 'OTHER'],
  ])('classifies %j as %s', (task, expected) => {
    expect(taskFamily(task)).toBe(expected);
  });
});
```

- [ ] **Step 2: Write a failing selector extraction test**

Create `frontend/src/features/vbs/components/VbsTaskSelector.test.jsx`:

```javascript
import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import VbsTaskSelector from './VbsTaskSelector';

const evaluations = [{
  id: 'eval-1',
  name: 'VBS',
  taskTemplates: [
    { name: 'KIS task', taskGroup: 'KIS', taskType: 'KIS', duration: 300 },
    { name: 'AVS task', taskGroup: 'AVS', taskType: 'AVS', duration: 300 },
  ],
}];

test('returns the complete selected task object through onRequestChange', () => {
  const onRequestChange = jest.fn();
  render(
    <VbsTaskSelector
      connectedUserId="team-a"
      evaluations={evaluations}
      selectedTask={{
        evaluationId: 'eval-1', evaluationName: 'VBS', taskName: 'KIS task',
        taskGroup: 'KIS', taskType: 'KIS', duration: 300,
      }}
      onRequestChange={onRequestChange}
    />,
  );

  fireEvent.change(screen.getByLabelText('Select evaluation task'), {
    target: { value: 'eval-1:AVS task' },
  });

  expect(onRequestChange).toHaveBeenCalledWith({
    evaluationId: 'eval-1',
    evaluationName: 'VBS',
    taskName: 'AVS task',
    taskGroup: 'AVS',
    taskType: 'AVS',
    duration: 300,
  });
});
```

- [ ] **Step 3: Write a failing App routing test isolated from session-handshake behavior**

Create `frontend/src/App.avs.test.jsx` so existing `App.test.jsx` can keep testing real session behavior:

```javascript
import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

const mockSession = {
  connectedUserId: 'team-a',
  draftUserId: 'team-a',
  invalidateSession: jest.fn(),
  evaluations: [],
  selectedTask: {
    evaluationId: 'eval-1', evaluationName: 'VBS', taskName: 'AVS task',
    taskGroup: 'AVS', taskType: 'AVS', duration: 300,
  },
  setSelectedTask: jest.fn(),
};

jest.mock('./features/vbs/contexts/VbsSessionContext', () => ({
  VbsSessionProvider: ({ children }) => children,
  useVbsSession: () => mockSession,
}));

jest.mock('./features/avs', () => ({
  AvsWorkspace: () => <div>Dedicated AVS workspace</div>,
}));

jest.mock('./features/search', () => ({
  SearchWorkspace: () => <div>KIS search workspace</div>,
}));

jest.mock('./features/health', () => ({
  useHealthCheck: () => ({ isHealthy: true, healthData: {} }),
}));

jest.mock('./features/event-trail', () => ({
  useEventTrail: () => ({
    session: null, pending: false, error: '', close: jest.fn(),
  }),
}));

jest.mock('./features/submission', () => ({
  SubmissionDialog: () => null,
  useDirectSubmission: () => ({ dialog: null, openError: '', opening: false }),
}));

test('routes an AVS task to the dedicated workspace', () => {
  render(<App />);

  expect(screen.getByText('Dedicated AVS workspace')).toBeTruthy();
  expect(screen.queryByText('KIS search workspace')).toBeNull();
});
```

The companion KIS-preservation assertion stays in existing `App.test.jsx`: when `selectedTask` is KIS, `SearchWorkspace` remains the Query workspace and its one-answer submission behavior still passes.

- [ ] **Step 4: Run focused tests and verify the new task routing does not exist**

```bash
cd frontend && npm test -- --watchAll=false --runTestsByPath \
  src/features/vbs/taskFamily.test.js \
  src/features/vbs/components/VbsTaskSelector.test.jsx \
  src/features/avs/components/AvsWorkspace.test.jsx \
  src/App.avs.test.jsx \
  src/App.test.jsx
```

Expected: FAIL.

- [ ] **Step 5: Implement shared task-family classification**

Create `taskFamily.js` with frontend logic equivalent to Task 5 backend classification; AVS must match exact `AVS` group/type and `AD-HOC` / `ADHOC` task types before generic KIS/TRAKE checks.

- [ ] **Step 6: Extract `VbsTaskSelector` from `ToolBox`**

`VbsTaskSelector` accepts:

```javascript
{
  connectedUserId,
  evaluations,
  selectedTask,
  onRequestChange,
  disabled = false,
}
```

It owns flattening evaluation/task templates and selected key formatting. On `<select>` change it finds the full matching task and calls `onRequestChange(match)`; it never calls context setters directly.

Update `ToolBox` to render:

```jsx
<VbsTaskSelector
  connectedUserId={connectedUserId}
  evaluations={propEvaluations ?? vbsSession.evaluations}
  selectedTask={selectedTask}
  onRequestChange={setSelectedTask}
/>
```

Remove the duplicate task-flattening/select code from `ToolBox`.

- [ ] **Step 7: Add the initial AVS query shell**

`AvsQueryControls` renders the shared task selector plus one text input and Search button. `AppContent` passes `evaluations` and `setSelectedTask` from `useVbsSession()` into `AppShell`, and `AppShell` passes `connectedUserId`, `evaluations`, `selectedTask`, `setSelectedTask`, `onFrameClick`, and `onSessionRejected` into `AvsWorkspace`. `AvsWorkspace` initially owns:

```javascript
const [query, setQuery] = useState('');
const [searchState, setSearchState] = useState({
  status: 'idle', results: [], response: null, error: '',
});
const [selectionState, dispatchSelection] = useReducer(
  avsSelectionReducer,
  undefined,
  createInitialAvsSelectionState,
);
```

Search uses `searchAvs()` and replaces only `searchState`; it must not recreate `selectionState`.

- [ ] **Step 8: Route AVS tasks in `App.jsx`**

Compute:

```javascript
const activeTaskFamily = taskFamily(selectedTask);
const isAvsTask = activeTaskFamily === 'AVS';
```

Render `AvsWorkspace` in the Query panel when `isAvsTask`; render the existing `SearchWorkspace` otherwise. Do not delete or branch deeply inside `SearchWorkspace`.

- [ ] **Step 9: Run routing tests**

```bash
cd frontend && npm test -- --watchAll=false --runTestsByPath \
  src/features/vbs/taskFamily.test.js \
  src/features/vbs/components/VbsTaskSelector.test.jsx \
  src/features/avs/components/AvsWorkspace.test.jsx \
  src/App.avs.test.jsx \
  src/App.test.jsx
```

Expected: PASS.

- [ ] **Step 10: Commit in the real git checkout**

```bash
git add frontend/src/features/vbs frontend/src/features/search-controls frontend/src/features/avs frontend/src/App.jsx frontend/src/App.test.jsx frontend/src/App.avs.test.jsx
git commit -m "feat: route AVS tasks to dedicated workspace"
```

---

### Task 8: Build the harvest grid with separate selection, inspection, and keyboard focus

**Files:**
- Create: `frontend/src/features/avs/gridNavigation.js`
- Create: `frontend/src/features/avs/gridNavigation.test.js`
- Create: `frontend/src/features/avs/components/AvsCandidateCard.jsx`
- Create: `frontend/src/features/avs/components/AvsHarvestGrid.jsx`
- Modify: `frontend/src/features/avs/components/AvsWorkspace.jsx`
- Modify: `frontend/src/features/avs/components/AvsWorkspace.test.jsx`
- Create: `frontend/src/styles/avs.css`
- Modify: `frontend/src/styles/index.css`
- Modify: `frontend/src/App.jsx`

**Interfaces:**
- `AvsCandidateCard` props: `{ candidate, selected, submitted, selectionDisabled, onToggle, onInspect, cardRef, tabIndex }`.
- `AvsHarvestGrid` props: `{ candidates, pending, submitted, selectionDisabled, onToggle, onInspect }`.
- `getDirectionalIndex(cards, currentIndex, key)` chooses the nearest card in the requested visual direction using card rectangles.
- App inspection reuses `ImageModal`; AVS inspection never receives the one-answer submission callback.

- [ ] **Step 1: Write failing navigation tests**

```javascript
// gridNavigation.test.js
const rect = (left, top, width = 100, height = 100) => ({
  left, top, right: left + width, bottom: top + height,
  width, height, x: left, y: top,
});

test('ArrowDown chooses the nearest card below the current card', () => {
  const cards = [rect(0, 0), rect(120, 0), rect(0, 120), rect(120, 120)];
  expect(getDirectionalIndex(cards, 0, 'ArrowDown')).toBe(2);
});
```

Cover all four arrow directions and edge no-op behavior.

- [ ] **Step 2: Add failing workspace interaction tests**

At the top of `AvsWorkspace.test.jsx`, mock only the network boundaries and use the real AVS reducer/grid components:

```javascript
import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchAvs } from '../../../api/avs';
import { getCurrentDresTask } from '../../../api/submissions';
import AvsWorkspace from './AvsWorkspace';

jest.mock('../../../api/avs', () => ({ searchAvs: jest.fn() }));
jest.mock('../../../api/submissions', () => ({
  getCurrentDresTask: jest.fn(),
  submitDresAnswers: jest.fn(),
}));

const task = {
  evaluationId: 'eval-1', evaluationName: 'VBS', taskName: 'AVS task',
  taskGroup: 'AVS', taskType: 'AVS', duration: 300,
};

const candidate = (id, videoId, timestampMs) => ({
  candidate_id: id,
  frame_id: id,
  video_id: videoId,
  frame_idx: timestampMs / 1000,
  timestamp_ms: timestampMs,
  fps: 25,
  retrieval_rank: 1,
  retrieval_score: 0.9,
  metadata: { title: null, caption: null, ocr: null, objects: [], asr: null },
});

const responseWith = (results) => ({
  results,
  latency: { retrieval_ms: 1, coverage_ms: 0, materialization_ms: 0, total_ms: 1 },
  candidate_pool_size: results.length,
  deduplicated_candidate_count: results.length,
  unique_videos: new Set(results.map((item) => item.video_id)).size,
  warnings: [],
});

const renderWorkspace = ({ onInspect = jest.fn() } = {}) => {
  const setSelectedTask = jest.fn();
  render(
    <AvsWorkspace
      connectedUserId="team-a"
      evaluations={[]}
      selectedTask={task}
      setSelectedTask={setSelectedTask}
      onInspect={onInspect}
      onSessionRejected={jest.fn()}
    />,
  );
  return { onInspect, setSelectedTask };
};

beforeEach(() => {
  searchAvs.mockReset().mockResolvedValue(responseWith([
    candidate('f1', 'V1', 1000),
    candidate('f2', 'V2', 2000),
  ]));
  getCurrentDresTask.mockReset().mockResolvedValue({
    user_id: 'team-a', evaluation_id: 'eval-1', task_scope_key: 'scope-1',
    task_name: 'AVS task', task_group: 'AVS', task_type: 'AVS', duration: 300,
  });
});

test('checkbox selection changes only local basket state', async () => {
  renderWorkspace();
  fireEvent.change(screen.getByRole('textbox', { name: 'AVS query' }), {
    target: { value: 'seafood' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search AVS' }));
  await screen.findByRole('button', { name: 'Inspect V1 at 1000 ms' });

  const checkbox = screen.getAllByRole('checkbox')[0];
  fireEvent.click(checkbox);

  expect(checkbox.checked).toBe(true);
});

test('thumbnail inspection does not select the candidate', async () => {
  const onInspect = jest.fn();
  renderWorkspace({ onInspect });
  fireEvent.change(screen.getByRole('textbox', { name: 'AVS query' }), {
    target: { value: 'seafood' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search AVS' }));
  const inspect = await screen.findByRole('button', { name: 'Inspect V1 at 1000 ms' });

  fireEvent.click(inspect);

  expect(onInspect).toHaveBeenCalledWith(expect.objectContaining({ candidate_id: 'f1' }));
  expect(screen.getAllByRole('checkbox')[0].checked).toBe(false);
});

test('Space toggles the focused card while Enter inspects without selecting', async () => {
  const onInspect = jest.fn();
  renderWorkspace({ onInspect });
  fireEvent.change(screen.getByRole('textbox', { name: 'AVS query' }), {
    target: { value: 'seafood' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search AVS' }));
  const card = await screen.findByTestId('avs-card-f1');

  card.focus();
  fireEvent.keyDown(card, { key: ' ' });
  expect(screen.getAllByRole('checkbox')[0].checked).toBe(true);

  fireEvent.keyDown(card, { key: 'Enter' });
  expect(onInspect).toHaveBeenCalledTimes(1);
  expect(screen.getAllByRole('checkbox')[0].checked).toBe(true);
});
```

Task 9 extends the same file with cross-search persistence and submitted-state cases after the basket controls exist.

- [ ] **Step 3: Run focused frontend tests and verify grid components are absent**

```bash
cd frontend && npm test -- --watchAll=false --runTestsByPath \
  src/features/avs/gridNavigation.test.js \
  src/features/avs/components/AvsWorkspace.test.jsx
```

Expected: FAIL.

- [ ] **Step 4: Implement directional focus helper**

For each requested direction, filter candidate rectangles to those whose center lies in that half-plane, then minimize a tuple of `(primary-axis distance, secondary-axis distance, index)` so behavior is deterministic and responsive without hard-coding column counts.

- [ ] **Step 5: Implement `AvsCandidateCard`**

Required DOM semantics:

```jsx
<article
  ref={cardRef}
  data-testid={`avs-card-${candidate.candidate_id}`}
  data-candidate-id={candidate.candidate_id}
  tabIndex={tabIndex}
>
  <label>
    <input
      type="checkbox"
      checked={selected}
      disabled={selectionDisabled || submitted}
      onChange={() => onToggle(candidate)}
    />
    <span>{submitted ? 'Submitted' : selected ? 'Selected' : 'Select'}</span>
  </label>
  <button type="button" onClick={() => onInspect(candidate)} aria-label={`Inspect ${candidate.video_id} at ${candidate.timestamp_ms} ms`}>
    <img src={keyframeUrl(candidate.frame_id)} alt="" loading="lazy" />
  </button>
  <span>{candidate.video_id}</span>
  <span>{formatTimestamp(candidate.timestamp_ms)}</span>
</article>
```

Do not display retrieval score as a primary UI element.

- [ ] **Step 6: Implement `AvsHarvestGrid` keyboard behavior**

Use roving `tabIndex`: one card is `0`, others `-1`. The grid handles:

```text
Arrow keys -> focus nearest visual neighbor
Space      -> onToggle(focusedCandidate)
Enter      -> onInspect(focusedCandidate)
```

Ignore keyboard shortcuts originating from input/button/select/textarea elements other than the card focus surface so checkbox/button native behavior is preserved.

- [ ] **Step 7: Connect grid selection to the reducer and inspection to App**

`AvsWorkspace` dispatches `SELECT` or `DESELECT` based on `selectionState.pending.has(candidate.candidate_id)`. `onInspect(candidate)` forwards a plain frame-like object to App. In `App.jsx`, when the active task is AVS, render `ImageModal` without `onOpenSubmission`; KIS continues to receive the existing one-answer submission callback.

- [ ] **Step 8: Add AVS styles**

`avs.css` must include:

```text
avs-workspace
avs-query-controls
avs-harvest-grid (responsive CSS grid)
avs-candidate-card
avs-candidate-card.is-selected
avs-candidate-card.is-submitted
avs-candidate-card:focus-visible
```

Use existing CSS variables/tokens; do not introduce a second design system.

- [ ] **Step 9: Run harvest-grid tests and App tests**

```bash
cd frontend && npm test -- --watchAll=false --runTestsByPath \
  src/features/avs/gridNavigation.test.js \
  src/features/avs/components/AvsWorkspace.test.jsx \
  src/App.avs.test.jsx \
  src/App.test.jsx
```

Expected: PASS.

- [ ] **Step 10: Commit in the real git checkout**

```bash
git add frontend/src/features/avs frontend/src/styles frontend/src/App.jsx frontend/src/App.test.jsx frontend/src/App.avs.test.jsx
git commit -m "feat: add AVS harvest grid interactions"
```

---

### Task 9: Bind the basket to a live task scope and add review/clear/task-switch guards

**Files:**
- Create: `frontend/src/features/avs/components/AvsSelectionBar.jsx`
- Create: `frontend/src/features/avs/components/AvsSelectionDrawer.jsx`
- Modify: `frontend/src/features/avs/components/AvsQueryControls.jsx`
- Modify: `frontend/src/features/avs/components/AvsWorkspace.jsx`
- Modify: `frontend/src/features/avs/components/AvsWorkspace.test.jsx`

**Interfaces:**
- AVS workspace resolves live scope through existing `getCurrentDresTask()` whenever a connected AVS task becomes active.
- First usable scope dispatches `BIND_SCOPE`.
- Intentional task switch while pending selections exist requires explicit discard before `setSelectedTask(nextTask)`.
- Unexpected live-scope mismatch never clears pending selections; it blocks submission/selection until the user explicitly clears or returns to the matching task.

- [ ] **Step 1: Add failing lifecycle tests**

Extend `AvsWorkspace.test.jsx` with these concrete helpers and lifecycle assertions. They reuse the `candidate()`, `responseWith()`, `task`, `searchAvs`, and `getCurrentDresTask` fixtures created in Task 8:

```javascript
const runSearch = async (text) => {
  fireEvent.change(screen.getByRole('textbox', { name: 'AVS query' }), {
    target: { value: text },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search AVS' }));
  await waitFor(() => expect(searchAvs).toHaveBeenCalled());
};

test('pending selections survive a later AVS search and remain reviewable', async () => {
  searchAvs
    .mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]))
    .mockResolvedValueOnce(responseWith([candidate('f2', 'V2', 2000)]));
  renderWorkspace();

  await runSearch('first query');
  fireEvent.click(await screen.findByRole('checkbox'));
  await runSearch('second query');

  fireEvent.click(screen.getByRole('button', { name: 'Review selections' }));
  expect(screen.getByRole('dialog', { name: 'Selected AVS answers' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Remove f1 from selection' })).toBeTruthy();
});

test('review remove deletes only the chosen candidate', async () => {
  searchAvs.mockResolvedValueOnce(responseWith([
    candidate('f1', 'V1', 1000), candidate('f2', 'V2', 2000),
  ]));
  renderWorkspace();
  await runSearch('seafood');
  screen.getAllByRole('checkbox').forEach((checkbox) => fireEvent.click(checkbox));

  fireEvent.click(screen.getByRole('button', { name: 'Review selections' }));
  fireEvent.click(screen.getByRole('button', { name: 'Remove f1 from selection' }));

  expect(screen.queryByRole('button', { name: 'Remove f1 from selection' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Remove f2 from selection' })).toBeTruthy();
});

test('Clear asks for confirmation and preserves pending items when cancelled', async () => {
  window.confirm = jest.fn().mockReturnValue(false);
  searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
  renderWorkspace();
  await runSearch('seafood');
  fireEvent.click(screen.getByRole('checkbox'));

  fireEvent.click(screen.getByRole('button', { name: 'Clear selections' }));

  expect(window.confirm).toHaveBeenCalledWith('Clear 1 pending AVS selection?');
  expect(screen.getByRole('checkbox').checked).toBe(true);
});

test('Clear removes pending items only after explicit confirmation', async () => {
  window.confirm = jest.fn().mockReturnValue(true);
  searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
  renderWorkspace();
  await runSearch('seafood');
  fireEvent.click(screen.getByRole('checkbox'));

  fireEvent.click(screen.getByRole('button', { name: 'Clear selections' }));

  expect(screen.getByRole('checkbox').checked).toBe(false);
});

const avsEvaluations = [{
  id: 'eval-1',
  name: 'VBS',
  taskTemplates: [
    { name: 'AVS task', taskGroup: 'AVS', taskType: 'AVS', duration: 300 },
    { name: 'AVS task 2', taskGroup: 'AVS', taskType: 'AVS', duration: 300 },
  ],
}];

const TaskHarness = () => {
  const [selectedTask, setSelectedTask] = React.useState(task);
  return (
    <AvsWorkspace
      connectedUserId="team-a"
      evaluations={avsEvaluations}
      selectedTask={selectedTask}
      setSelectedTask={setSelectedTask}
      onInspect={jest.fn()}
      onSessionRejected={jest.fn()}
    />
  );
};

test('task switch cancellation keeps the current task and basket', async () => {
  window.confirm = jest.fn().mockReturnValue(false);
  searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
  render(<TaskHarness />);
  await runSearch('seafood');
  fireEvent.click(screen.getByRole('checkbox'));

  fireEvent.change(screen.getByLabelText('Select evaluation task'), {
    target: { value: 'eval-1:AVS task 2' },
  });

  expect(window.confirm).toHaveBeenCalled();
  expect(screen.getByLabelText('Select evaluation task').value).toBe('eval-1:AVS task');
  expect(screen.getByRole('checkbox').checked).toBe(true);
});

test('confirmed task switch waits for the new scope before clearing pending state', async () => {
  window.confirm = jest.fn().mockReturnValue(true);
  getCurrentDresTask.mockImplementation(async (_userId, { taskName }) => ({
    user_id: 'team-a',
    evaluation_id: 'eval-1',
    task_scope_key: taskName === 'AVS task 2' ? 'scope-2' : 'scope-1',
    task_name: taskName,
    task_group: 'AVS',
    task_type: 'AVS',
    duration: 300,
  }));
  searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
  render(<TaskHarness />);
  await runSearch('seafood');
  fireEvent.click(screen.getByRole('checkbox'));

  fireEvent.change(screen.getByLabelText('Select evaluation task'), {
    target: { value: 'eval-1:AVS task 2' },
  });

  await waitFor(() => {
    expect(screen.getByLabelText('Select evaluation task').value).toBe('eval-1:AVS task 2');
  });
  await waitFor(() => expect(screen.queryByText(/1 selected/)).toBeNull());
});

const ExternalTaskChangeHarness = () => {
  const [selectedTask, setSelectedTask] = React.useState(task);
  return (
    <>
      <button
        type="button"
        onClick={() => setSelectedTask({ ...task, taskName: 'AVS task 2' })}
      >
        Simulate external task change
      </button>
      <AvsWorkspace
        connectedUserId="team-a"
        evaluations={avsEvaluations}
        selectedTask={selectedTask}
        setSelectedTask={setSelectedTask}
        onInspect={jest.fn()}
        onSessionRejected={jest.fn()}
      />
    </>
  );
};

test('unexpected task-scope change preserves the old basket and blocks mutation', async () => {
  getCurrentDresTask.mockImplementation(async (_userId, { taskName }) => ({
    user_id: 'team-a', evaluation_id: 'eval-1',
    task_scope_key: taskName === 'AVS task 2' ? 'scope-2' : 'scope-1',
    task_name: taskName, task_group: 'AVS', task_type: 'AVS', duration: 300,
  }));
  searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
  render(<ExternalTaskChangeHarness />);
  await runSearch('seafood');
  fireEvent.click(screen.getByRole('checkbox'));

  fireEvent.click(screen.getByRole('button', { name: 'Simulate external task change' }));

  expect(await screen.findByRole('alert')).toHaveTextContent(/task scope changed/i);
  expect(screen.getByRole('checkbox')).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Review selections' }));
  expect(screen.getByRole('button', { name: 'Remove f1 from selection' })).toBeTruthy();
});
```

- [ ] **Step 2: Run workspace tests and verify lifecycle UI is missing**

```bash
cd frontend && npm test -- --watchAll=false --runTestsByPath src/features/avs/components/AvsWorkspace.test.jsx
```

Expected: FAIL.

- [ ] **Step 3: Resolve and bind the live task scope outside selection clicks**

In `AvsWorkspace`, add an effect triggered by `connectedUserId`, `selectedTask.evaluationId`, and `selectedTask.taskName`:

```javascript
getCurrentDresTask(connectedUserId, {
  evaluationId: selectedTask?.evaluationId,
  taskName: selectedTask?.taskName,
})
```

If the reducer has no scope and no pending items, dispatch `BIND_SCOPE`. If a different live scope appears while pending exists, set a local `scopeConflict` object; do not dispatch reset and do not clear pending.

Selection controls are disabled while scope is loading, absent, conflicted, or `selectionState.unknownBatch` is non-null.

- [ ] **Step 4: Guard intentional task changes in `AvsQueryControls`**

Pass an `onRequestTaskChange(nextTask)` callback from `AvsWorkspace`. If pending is empty, switch immediately. Otherwise call:

```javascript
window.confirm(`Discard ${selectionState.pending.size} pending AVS selections and switch task?`)
```

On confirm: store `nextTask` in local `pendingTaskSwitch`, disable selection/submission controls, and call `setSelectedTask(nextTask)` **without clearing the current basket yet**. The scope-resolution effect then resolves `nextTask`; only after it receives the new `task_scope_key` does it dispatch `RESET_FOR_SCOPE` with that key and clear `pendingTaskSwitch`. On cancel: do nothing. This ordering prevents both accidental early basket loss and binding the old basket to a guessed scope key.

Do not add this AVS basket guard to KIS `ToolBox`; KIS has no AVS pending basket.

- [ ] **Step 5: Implement sticky selection summary and review drawer**

`AvsSelectionBar` shows:

```text
{pendingCount} selected · {uniqueVideoCount} videos
[Review] [Clear] [Submit N]
```

`AvsSelectionDrawer` groups display by video ID for awareness but renders every selected candidate with a Remove action. It must not rerank or auto-delete.

There is no Select All button.

- [ ] **Step 6: Run lifecycle tests**

```bash
cd frontend && npm test -- --watchAll=false --runTestsByPath src/features/avs/components/AvsWorkspace.test.jsx
```

Expected: PASS.

- [ ] **Step 7: Commit in the real git checkout**

```bash
git add frontend/src/features/avs
git commit -m "feat: add task-scoped AVS selection basket"
```

---

### Task 10: Add failure-safe AVS batch submission and uncertain-outcome handling

**Files:**
- Create: `frontend/src/features/avs/hooks/useAvsSubmission.js`
- Create: `frontend/src/features/avs/hooks/useAvsSubmission.test.js`
- Create: `frontend/src/features/avs/components/AvsSubmitDialog.jsx`
- Modify: `frontend/src/features/avs/components/AvsSelectionBar.jsx`
- Modify: `frontend/src/features/avs/components/AvsWorkspace.jsx`
- Modify: `frontend/src/features/avs/components/AvsWorkspace.test.jsx`

**Interfaces:**
- Consumes: `submitDresAnswers()` from Task 5.
- Produces hook state: `{ status: 'IDLE'|'SUBMITTING'|'RECORDED'|'NOT_RECORDED'|'UNKNOWN', outcome, error }`.
- Produces methods: `submitBatch(batch)`, `retryUnknown(batch)`, `resetOutcome()`.
- `retryUnknown()` is only reachable from an explicit user action after an `UNKNOWN` outcome; the hook never schedules or performs automatic retry.

- [ ] **Step 1: Write failing hook tests for all outcome states**

Create `frontend/src/features/avs/hooks/useAvsSubmission.test.js` with a fixed batch and explicit hook context:

```javascript
import { act, renderHook } from '@testing-library/react';
import { submitDresAnswers } from '../../../api/submissions';
import { useAvsSubmission } from './useAvsSubmission';

jest.mock('../../../api/submissions', () => ({
  submitDresAnswers: jest.fn(),
}));

const batch = {
  candidateIds: ['f1', 'f2'],
  answers: [
    { kind: 'TEMPORAL', video_id: 'V1', start_ms: 1000, end_ms: 1000 },
    { kind: 'TEMPORAL', video_id: 'V2', start_ms: 2000, end_ms: 2000 },
  ],
};

const selectedTask = { evaluationId: 'eval-1', taskName: 'AVS task' };

const renderSubmission = (onSessionRejected = jest.fn()) => renderHook(() => useAvsSubmission({
  userId: 'team-a',
  selectedTask,
  taskScopeKey: 'scope-1',
  onSessionRejected,
}));

beforeEach(() => {
  submitDresAnswers.mockReset();
});

test('RECORDED sends the entire batch exactly once', async () => {
  submitDresAnswers.mockResolvedValue({
    state: 'RECORDED', recorded: true, verdict: 'CORRECT', message: 'ok',
  });
  const { result } = renderSubmission();

  await act(async () => result.current.submitBatch(batch));

  expect(submitDresAnswers).toHaveBeenCalledTimes(1);
  expect(submitDresAnswers).toHaveBeenCalledWith(expect.objectContaining({
    userId: 'team-a',
    expectedTaskScopeKey: 'scope-1',
    evaluationId: 'eval-1',
    taskName: 'AVS task',
    answers: batch.answers,
  }));
  expect(result.current.status).toBe('RECORDED');
});

test('NOT_RECORDED is definitive and does not retry', async () => {
  submitDresAnswers.mockResolvedValue({
    state: 'NOT_RECORDED',
    recorded: false,
    reason: 'DRES_REJECTED',
    message: 'rejected',
  });
  const { result } = renderSubmission();

  await act(async () => result.current.submitBatch(batch));

  expect(result.current.status).toBe('NOT_RECORDED');
  expect(submitDresAnswers).toHaveBeenCalledTimes(1);
});

test('UNKNOWN does not trigger an automatic second call', async () => {
  submitDresAnswers.mockResolvedValue({
    state: 'UNKNOWN', recorded: null, message: 'check DRES',
  });
  const { result } = renderSubmission();

  await act(async () => result.current.submitBatch(batch));

  expect(result.current.status).toBe('UNKNOWN');
  expect(submitDresAnswers).toHaveBeenCalledTimes(1);
});

test('TASK_SCOPE_MISMATCH remains a definite local error', async () => {
  const error = Object.assign(new Error('scope changed'), {
    status: 409,
    code: 'TASK_SCOPE_MISMATCH',
  });
  submitDresAnswers.mockRejectedValue(error);
  const { result } = renderSubmission();

  await act(async () => result.current.submitBatch(batch));

  expect(result.current.status).toBe('IDLE');
  expect(result.current.error).toMatchObject({ code: 'TASK_SCOPE_MISMATCH' });
  expect(submitDresAnswers).toHaveBeenCalledTimes(1);
});

test('DRES_AUTH_REJECTED notifies the session owner once', async () => {
  const onSessionRejected = jest.fn();
  submitDresAnswers.mockResolvedValue({
    state: 'NOT_RECORDED',
    recorded: false,
    reason: 'DRES_AUTH_REJECTED',
    message: 'reconnect',
  });
  const { result } = renderSubmission(onSessionRejected);

  await act(async () => result.current.submitBatch(batch));

  expect(result.current.status).toBe('NOT_RECORDED');
  expect(onSessionRejected).toHaveBeenCalledTimes(1);
});
```

- [ ] **Step 2: Add failing workspace tests for basket mutation**

Extend `AvsWorkspace.test.jsx`; import the already mocked `submitDresAnswers` and reuse the Task 8/9 `runSearch()`, `candidate()`, `responseWith()`, and `renderWorkspace()` helpers:

```javascript
import { getCurrentDresTask, submitDresAnswers } from '../../../api/submissions';

const selectOneForSubmission = async () => {
  searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
  renderWorkspace();
  await runSearch('seafood');
  fireEvent.click(screen.getByRole('checkbox'));
  fireEvent.click(screen.getByRole('button', { name: 'Submit 1 answer' }));
  await screen.findByRole('dialog', { name: 'Submit AVS answers' });
};

test('RECORDED moves the submitted candidate out of pending state', async () => {
  submitDresAnswers.mockResolvedValueOnce({
    state: 'RECORDED', recorded: true, verdict: 'CORRECT', message: 'recorded',
  });
  await selectOneForSubmission();

  fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));

  await screen.findByText('Submitted');
  expect(screen.getByRole('checkbox')).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Submit 1 answer' })).toBeNull();
  expect(submitDresAnswers).toHaveBeenCalledTimes(1);
});

test('NOT_RECORDED leaves the selected candidate pending', async () => {
  submitDresAnswers.mockResolvedValueOnce({
    state: 'NOT_RECORDED', recorded: false, reason: 'DRES_REJECTED', message: 'rejected',
  });
  await selectOneForSubmission();

  fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));

  await screen.findByText('rejected');
  expect(screen.getByRole('checkbox').checked).toBe(true);
  expect(screen.getByRole('button', { name: 'Submit 1 answer' })).toBeTruthy();
  expect(submitDresAnswers).toHaveBeenCalledTimes(1);
});

test('UNKNOWN freezes normal mutation and never retries automatically', async () => {
  submitDresAnswers.mockResolvedValueOnce({
    state: 'UNKNOWN', recorded: null, message: 'check DRES',
  });
  await selectOneForSubmission();

  fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));

  await screen.findByText('check DRES');
  expect(screen.getByRole('checkbox')).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Retry after verification' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Mark recorded after verification' })).toBeTruthy();
  expect(submitDresAnswers).toHaveBeenCalledTimes(1);
});

test('UNKNOWN retries only after the explicit verification action', async () => {
  window.confirm = jest.fn().mockReturnValue(true);
  submitDresAnswers
    .mockResolvedValueOnce({ state: 'UNKNOWN', recorded: null, message: 'check DRES' })
    .mockResolvedValueOnce({ state: 'RECORDED', recorded: true, verdict: 'CORRECT', message: 'recorded' });
  await selectOneForSubmission();
  fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));
  await screen.findByRole('button', { name: 'Retry after verification' });

  fireEvent.click(screen.getByRole('button', { name: 'Retry after verification' }));

  expect(window.confirm).toHaveBeenCalledWith(
    'I verified DRES state and want to retry this exact batch.',
  );
  await waitFor(() => expect(submitDresAnswers).toHaveBeenCalledTimes(2));
  await screen.findByText('Submitted');
});

test('mark recorded after verification performs no second network request', async () => {
  submitDresAnswers.mockResolvedValueOnce({
    state: 'UNKNOWN', recorded: null, message: 'check DRES',
  });
  await selectOneForSubmission();
  fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));
  await screen.findByRole('button', { name: 'Mark recorded after verification' });

  fireEvent.click(screen.getByRole('button', { name: 'Mark recorded after verification' }));

  expect(submitDresAnswers).toHaveBeenCalledTimes(1);
  await screen.findByText('Submitted');
});

test('TASK_SCOPE_MISMATCH preserves pending selection and surfaces conflict', async () => {
  submitDresAnswers.mockRejectedValueOnce(Object.assign(new Error('scope changed'), {
    status: 409,
    code: 'TASK_SCOPE_MISMATCH',
  }));
  await selectOneForSubmission();

  fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));

  expect(await screen.findByRole('alert')).toHaveTextContent(/task scope changed/i);
  expect(screen.getByRole('checkbox').checked).toBe(true);
});
```

These tests fix the dialog/button accessible names used by the implementation: the batch dialog is named `Submit AVS answers`, its network action is `Confirm submit`, and the sticky action uses singular/plural copy `Submit 1 answer` / `Submit N answers`.

- [ ] **Step 3: Run tests and verify AVS batch submission UI/hook are absent**

```bash
cd frontend && npm test -- --watchAll=false --runTestsByPath \
  src/features/avs/hooks/useAvsSubmission.test.js \
  src/features/avs/components/AvsWorkspace.test.jsx
```

Expected: FAIL.

- [ ] **Step 4: Implement `useAvsSubmission`**

The hook receives `{ userId, selectedTask, taskScopeKey, onSessionRejected }` and freezes those values into each submitted batch. `submitBatch()` calls `submitDresAnswers()` exactly once.

On thrown `error.code === 'TASK_SCOPE_MISMATCH'`, return to `IDLE` with a dedicated error object; do not transform it into `UNKNOWN`, because the backend definitively did not contact DRES.

On returned outcomes, set status directly from `outcome.state`. If the returned outcome is `NOT_RECORDED` with `reason === 'DRES_AUTH_REJECTED'`, call `onSessionRejected()` exactly once after storing the outcome; pending-selection mutation remains the workspace's responsibility. `retryUnknown(batch)` calls the same one-request submission path only when current status is `UNKNOWN`.

- [ ] **Step 5: Implement lightweight batch confirmation**

`AvsSubmitDialog` has `role="dialog"`, `aria-label="Submit AVS answers"`, and copy:

```text
Submit {N} answers from {M} videos?
[Cancel] [Confirm submit]
```

It does not render or edit CSV/JSON lines. Detailed review stays in the selection drawer.

- [ ] **Step 6: Wire outcome-specific reducer transitions**

Before submit, freeze:

```javascript
const candidateIds = [...selectionState.pending.keys()];
const answers = [...selectionState.pending.values()].map(({ candidate }) => (
  candidateToTemporalAnswer(candidate)
));
```

Then:

```text
RECORDED     -> dispatch { type: 'RECORDED', candidateIds }
NOT_RECORDED -> no reducer mutation
UNKNOWN      -> dispatch { type: 'UNKNOWN', candidateIds }
```

When `UNKNOWN` is active, normal selection mutation and normal Submit are disabled. Show two explicit actions:

```text
Retry after verification
Mark recorded after verification
```

`Retry after verification` first calls `window.confirm('I verified DRES state and want to retry this exact batch.')`; only on confirm call `retryUnknown()` with the frozen unknown batch.

`Mark recorded after verification` dispatches `MARK_UNKNOWN_RECORDED` and performs no network call.

- [ ] **Step 7: Run hook/workspace tests**

```bash
cd frontend && npm test -- --watchAll=false --runTestsByPath \
  src/features/avs/hooks/useAvsSubmission.test.js \
  src/features/avs/components/AvsWorkspace.test.jsx
```

Expected: PASS.

- [ ] **Step 8: Commit in the real git checkout**

```bash
git add frontend/src/features/avs
git commit -m "feat: add failure-safe AVS batch submission"
```

---

### Task 11: Add end-to-end regression coverage for latency path and task isolation

**Files:**
- Modify: `tests/orchestration/workflows/test_avs_search.py`
- Modify: `tests/api/test_avs_router.py`
- Modify: `tests/api/test_vbs_submission.py`
- Modify: `frontend/src/features/avs/components/AvsWorkspace.test.jsx`
- Modify: `frontend/src/App.test.jsx`

**Interfaces:**
- No new runtime interface; this task locks the approved AVS v1 behavior against regressions.

- [ ] **Step 1: Add backend fast-path sentinels**

Construct a `SearchService` with exploding sentinels:

```python
service.intent_resolver = Mock(side_effect=AssertionError("KIS intent resolver called"))
service.scoped_resolver = Mock(side_effect=AssertionError("KIS scoped resolver called"))
service.global_rewriter = Mock(side_effect=AssertionError("KIS global rewriter called"))
service.kis.execute = Mock(side_effect=AssertionError("KIS temporal pipeline called"))
service.event_trail = Mock(side_effect=AssertionError("EventTrail called"))
```

Call `search_avs()` and assert it returns normally. Add a retrieval-gateway fake exposing only `search_text()` so any attempt to use temporal search fails structurally.

- [ ] **Step 2: Add coverage acceptance fixture**

Use the spec fixture:

```text
V1 .99
V1 .98
V1 .97
V2 .96
V3 .95
```

with non-neighbor V1 timestamps so Stage A does not remove them. Assert early output is `V1, V2, V3, V1, V1`, proving video-first ordering independent of dedup.

- [ ] **Step 3: Add one-request/one-answer-set integration assertion**

Submit 20 AVS temporal answers through the FastAPI VBS router and assert:

```python
assert fake_dres.submit.call_count == 1
payload = fake_dres.submit.call_args.args[2]
assert len(payload.answer_sets) == 1
assert len(payload.answer_sets[0].answers) == 20
```

Then submit two KIS answers and assert HTTP 422 plus `fake_dres.submit.call_count == 0` for that request.

- [ ] **Step 4: Add frontend search/selection/submission isolation regression**

In one `AvsWorkspace` test:

```text
search A -> select f1/f2
search B -> f1 reappears selected, f2 remains in drawer although not in grid
inspect f3 -> pending remains f1/f2 only
submit -> one submitDresAnswers call with f1/f2
RECORDED -> both render as Submitted if they reappear
```

Assert no `submitDresAnswers` call occurs on selection, deselection, search, or inspection.

- [ ] **Step 5: Run the complete new backend suite**

```bash
PYTHONPATH=src python -m pytest tests -v
```

Expected: PASS.

- [ ] **Step 6: Run the complete frontend test suite**

```bash
cd frontend && npm test -- --watchAll=false
```

Expected: PASS.

- [ ] **Step 7: Build the frontend**

```bash
cd frontend && npm run build
```

Expected: successful production build with no compile errors.

- [ ] **Step 8: Commit in the real git checkout**

```bash
git add tests frontend/src
git commit -m "test: lock AVS workflow regressions"
```

---

### Task 12: Manual acceptance pass against the approved AVS workflow

**Files:**
- No new implementation files. If an acceptance check fails, return to the owning task, add a regression test there, fix the bug, rerun that task's focused suite, and commit that focused fix before resuming this checklist.

**Interfaces:**
- Verifies the approved AVS flow end-to-end without adding product behavior.

- [ ] **Step 1: Start the three local processes with explicit commands**

Terminal 1 — retrieval service:

```bash
PYTHONPATH=.:src python -m hcmai.retrieval.serving.server --host 127.0.0.1 --port 8002
```

Terminal 2 — web backend:

```bash
PYTHONPATH=.:src python -m uvicorn hcmai.app:app --host 127.0.0.1 --port 8000
```

Terminal 3 — frontend:

```bash
cd frontend
npm start
```

Verify readiness before interacting with the browser:

```bash
curl --fail http://127.0.0.1:8002/capabilities
curl --fail http://127.0.0.1:8000/health
```

Run the commands from the repository root; no command in this step enables or starts an AVS reranker.

- [ ] **Step 2: Verify the direct AVS latency path**

Select an AVS task and issue one text search. In backend/retrieval logs, confirm that the request reaches `POST /api/v1/avs/search` and `/search_text`. For that request, there must be no KIS intent-resolution, KIS rewrite, temporal search/alignment, EventTrail, or online reranker log entry.

- [ ] **Step 3: Verify coverage ordering using the deterministic acceptance fixture**

Rerun the focused selector acceptance test so the manual build is checked against the exact duplicate-heavy fixture rather than an ad-hoc corpus example:

```bash
PYTHONPATH=src python -m pytest \
  tests/orchestration/workflows/test_avs_coverage.py::test_video_first_pass_exposes_other_videos_before_second_v1_candidate \
  -v
```

Expected order: `V1, V2, V3, V1, V1` when the V1 candidates are outside the temporal suppression window; the neighboring-frame test must still collapse local V1 duplicates.

- [ ] **Step 4: Verify harvest interaction in the browser**

Select at least 10 candidates spanning multiple video IDs, inspect at least one candidate without selecting it, issue a second AVS query, then open Review. The original pending selections must still be present; inspection must not have added the inspected-only frame.

- [ ] **Step 5: Verify task-scope guard**

With pending selections, request an AVS task switch. Cancel the discard confirmation and verify both the current task and basket remain unchanged. Request the switch again, confirm discard, and verify the basket is cleared only as part of the explicit switch flow.

- [ ] **Step 6: Verify one-request multi-answer submission**

Select at least two AVS frames and submit the batch. Confirm the browser issues one `POST /api/v1/vbs/submit` request whose JSON body contains one `answers` array, and confirm backend logging shows one DRES `ApiClientSubmission` with one `ApiClientAnswerSet` containing the entire selected list.

- [ ] **Step 7: Re-run the automated failure-semantics coverage instead of injecting ambiguous failures into a live DRES task**

```bash
cd frontend
npm test -- --watchAll=false --runTestsByPath \
  src/features/avs/hooks/useAvsSubmission.test.js \
  src/features/avs/components/AvsWorkspace.test.jsx
```

Expected: tests for `RECORDED`, `NOT_RECORDED`, `UNKNOWN`, explicit unknown retry, and `TASK_SCOPE_MISMATCH` all pass. Do not manufacture a network timeout against the competition DRES instance for acceptance testing.

- [ ] **Step 8: Run final verification commands**

```bash
PYTHONPATH=src python -m pytest tests -v
cd frontend
npm test -- --watchAll=false
npm run build
```

Expected: backend tests pass, frontend tests pass, and the production frontend build succeeds. If this step exposes a defect, fix and commit it in the owning task before declaring the AVS workspace complete.

