# Unified Temporal Alignment Cleanup Implementation Plan

> **Implementation workflow:** Execute the checked tasks in order. Preserve the
> Task 0 decision record, use focused TDD before each behavior change, and do
> not start destructive cleanup before its stated migration gate passes.

**Goal:** Replace the progressive KIS scene/state pipeline with one stateless monotonic DP alignment core shared by KIS and TRAKE, while preserving public response contracts and deleting legacy temporal abstractions.

**Architecture:** Keep the existing visual embeddings, `DenseIndex`, `RetrievalService`, `VideoEventScores`, and monotonic DP as the reference baseline. Add a deterministic query planner and a task-agnostic `TemporalAlignmentService`; KIS projects an aligned path to one representative frame while TRAKE exposes the full path. Remove progressive state, backfill, scene clustering, soft temporal-relation scoring, and default single-frame reranking only after both task heads pass characterization tests on the new core.

**Tech Stack:** Python 3.14, Pydantic, NumPy, FAISS-backed `DenseIndex`, pytest, existing HCMAI retrieval/data services.

**Spec:** `docs/superpowers/specs/2026-08-30-unified-temporal-alignment-design.md`

**Decision status:** PROPOSED. The final architecture deliberately changes KIS
from progressive scene localization to ordered event/frame alignment. It is not
a semantics-preserving cleanup and must not be described as a verified accuracy
improvement. Complete Task 0 and record an explicit cut-over decision before
Task 10 deletes the legacy implementation.

## Execution Status — 2026-08-30

- **Task 0:** completed as a no-cut-over gate. The local organizer document
  confirms complete KIS queries and ordered TRAKE event frames, but neither a
  labeled development set nor a frame-reuse rule is available. The recorded
  outcome permits preparatory work only; see
  `docs/research/2026-08-30-temporal-migration-gate.md`.
- **Tasks 1–5:** implemented and tested. They add characterization, shared
  contracts/configuration, deterministic planning, filter-aware visual event
  scoring, and a parity-tested duplicate DP module. The legacy runtime remained
  active while these preparatory tasks completed.
- **Tasks 6–8:** implemented and tested under explicit user authorization.
  KIS and TRAKE now share the stateless alignment service; KIS retains its
  public response shape, projects a deterministic path midpoint, and rejects
  ambiguous `min_score` filters. This code migration is not a measured release
  decision: the Task 0 evaluation gate still applies before accepting the
  competition trade-off or deleting legacy code.
- **Task 9:** implemented and tested under explicit user authorization. The
  default service no longer constructs or owns a reranker, while the standalone
  reranking package remains available for explicit experiments.
- **Tasks 10–12:** implemented under the same authorization. The progressive
  runtime, schema/configuration surface, compatibility scorer, and legacy-only
  tests are removed; current docs describe the stateless baseline.
- **Task 13:** compile, source audits, and the complete Python suite pass
  (851 tests). Loaded-service smoke execution remains pending because startup
  did not complete within the 30-second local execution window while waiting
  on configured remote inference. Baseline metrics remain pending: this
  workspace has no versioned development-query manifest or labelled results.
- **Commits:** intentionally not created; the working tree contained unrelated
  user changes and the request did not authorize a commit.

## Global Constraints

- Preserve existing KIS `SearchResponse` and TRAKE `TRAKEResponse` shapes.
- Keep `search_id` in the KIS API for compatibility, but make temporal execution stateless.
- Baseline alignment uses existing visual event/frame scores only; multimodal dense scoring is a later experiment.
- Do not add entity tracking, state-transition verification, VQA, incremental DP, or multi-frame VLM verification in this refactor.
- KIS `SearchFilters` must still constrain alignment by video and timestamp; non-null `min_score` must be rejected explicitly rather than silently ignored.
- Default KIS ordering must use DP path score; no single-frame reranker may overwrite it.
- Delete legacy progressive/scene code only after KIS and TRAKE are running through the new service and tests pass.
- Use TDD for every behavior change and commit after each task.
- Every new or materially rewritten Python module, public class, and public
  function must carry the meaningful ownership/invariant docstrings required by
  `AGENTS.md`; code snippets below show only the semantic core when the
  docstrings would obscure it.
- Work in the repository's existing test layers (`tests/unit`,
  `tests/integration`, and targeted top-level compatibility tests); do not
  create a parallel `tests/temporal` or `tests/retrieval` convention.
- Do not stage unrelated user changes. Every commit command in this document
  names its owned paths; replace any broad staging command with its concrete
  touched files.

---

### Task 0: Establish Migration Authority and a Measured Decision Gate

**Files:**
- Create or update: the repository's established versioned evaluation-result
  location (do not add a runtime subsystem)
- Update when the gate closes: `KNOWLEDGE.md`

**Interfaces:**
- Consumes: the current public KIS/TRAKE contracts, the current organizer
  specification/scorer when available, and a frozen development query set.
- Produces: a reproducible comparison record and an explicit decision to
  proceed, revise the design, or retain the current KIS behavior.

- [x] **Step 1: Reconcile the current 2026 organizer contract before coding**

Record whether the scorer permits one strict-increasing frame per event, frame
reuse, or a progressive-search interaction. Organizer rules override this
plan's strict-DP assumption. If the contract is unavailable, label this
assumption **PROPOSED** rather than silently treating it as a requirement.

- [ ] **Step 2: Freeze a small, versioned development set and capture current outputs**

For each query/event sequence, record the dataset/query-set version, current
KIS and TRAKE outputs, canonical identities, model/index/config versions, and
P50/P95 latency. Preserve the results outside `src/`; use the project's
existing evaluation location or `artifacts/evaluation/temporal_migration/` if
none exists.

- [x] **Step 3: Define the cut-over rule before seeing new results**

At minimum, compare the appropriate official task metric (or an explicitly
labeled proxy), retrieval/localization failure cases, candidate-video count,
and P50/P95 latency. A rank-identical KIS result is not expected because its
semantics change. The recorded decision must say whether the measured trade-off
is accepted; tests alone are not sufficient authority to delete working KIS
logic.

- [x] **Step 4: Record the gate outcome in `KNOWLEDGE.md`**

Use **SOURCE** for current-code/contract facts, **PAPER** for literature
support, and **PROPOSED**, **VERIFIED**, or **REJECTED** for the HCMAI outcome.
Do not call the DP baseline an improvement unless the frozen evaluation supports
that claim.

- [ ] **Step 5: Commit only the reproducibility artifacts that are safe to version**

Do not commit private data, large generated results, or credentials. Commit a
small query manifest/config and a result summary only when repository policy
permits them; otherwise record the artifact paths and checksums in the run
report.

---

## File Structure After Refactor

```text
src/hcmai/common/schemas/alignment.py
    AlignmentEvent, AlignmentPlan, AlignmentPath

src/hcmai/temporal/planner.py
    deterministic query -> ordered event plan

src/hcmai/temporal/dp.py
    pure monotonic dynamic programming; no DataService or HTTP concerns

src/hcmai/temporal/service.py
    visual score acquisition + canonical path materialization

src/hcmai/temporal/__init__.py
    export only planner/service/alignment-facing symbols

src/hcmai/orchestration/workflows/kis.py
    thin KIS output head over TemporalAlignmentService

src/hcmai/orchestration/workflows/trake.py
    thin TRAKE output head over TemporalAlignmentService
```

Files intentionally removed by the end of the migration:

```text
src/hcmai/temporal/core.py
src/hcmai/temporal/ports.py
src/hcmai/temporal/query.py
src/hcmai/temporal/settings.py
src/hcmai/temporal/providers/dense.py
src/hcmai/temporal/providers/sparse.py
src/hcmai/temporal/aligners/monotonic.py
src/hcmai/temporal/aligners/monotonic_dp.py
src/hcmai/temporal/aligners/scene.py
src/hcmai/temporal/state/evidence.py
src/hcmai/temporal/state/state.py
src/hcmai/temporal/utils/relations.py
src/hcmai/temporal/utils/scoring.py
```

The `retrieval/reranking/` package remains available for experiments, but the default KIS registry no longer initializes or calls it.

---

### Task 1: Add Python Characterization Tests Before Refactoring

**Files:**
- Modify: `tests/unit/temporal/test_monotonic_dp.py`
- Create: `tests/unit/orchestration/test_public_contracts.py`

**Interfaces:**
- Consumes: existing `align_video()`, `VideoEventScores`, `SearchRequest`, `TRAKERequest`.
- Produces: tests that freeze chronological DP behavior and public request/response assumptions before any deletion.

- [x] **Step 1: Write the DP chronological-path test**

```python
# tests/unit/temporal/test_monotonic_dp.py
import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.aligners.monotonic_dp import align_video


def test_align_video_chooses_best_chronological_path():
    video = VideoEventScores(
        video_id="V01",
        frame_ids=np.array(["f0", "f1", "f2", "f3"], dtype=object),
        frame_idx=np.array([0, 1, 2, 3]),
        timestamps_ms=np.array([0, 1000, 2000, 3000]),
        scores=np.array([
            [0.90, 0.20, 0.10, 0.05],
            [0.10, 0.85, 0.30, 0.10],
            [0.05, 0.10, 0.40, 0.95],
        ]),
    )

    [path] = align_video(video, lambda_gap=0.0, paths=1)

    assert path.frame_ids == ("f0", "f1", "f3")
    assert path.frame_idx == (0, 1, 3)
```

- [x] **Step 2: Write the gap-penalty characterization test**

```python
def test_align_video_gap_penalty_can_prefer_nearer_frame():
    video = VideoEventScores(
        video_id="V01",
        frame_ids=np.array(["f0", "f1", "f2"], dtype=object),
        frame_idx=np.array([0, 1, 2]),
        timestamps_ms=np.array([0, 1000, 100_000]),
        scores=np.array([
            [0.90, 0.10, 0.10],
            [0.10, 0.80, 0.99],
        ]),
    )

    [path] = align_video(video, lambda_gap=1e-5, paths=1)

    assert path.frame_ids == ("f0", "f1")
```

- [x] **Step 3: Run the DP tests before changing code**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/temporal/test_monotonic_dp.py -v
```

Expected: PASS against the current `monotonic_dp.py`.

- [x] **Step 4: Add a request-contract test proving KIS accepts no events today and TRAKE does**

```python
# tests/unit/orchestration/test_public_contracts.py
from hcmai.common.schemas import SearchRequest, TRAKERequest


def test_current_public_requests_are_constructible():
    kis = SearchRequest(query="chef coats food with flour")
    trake = TRAKERequest(
        query="ordered cooking events",
        events=["chef holds skewer", "chef coats skewer"],
    )

    assert kis.query == "chef coats food with flour"
    assert trake.events == ["chef holds skewer", "chef coats skewer"]
```

- [x] **Step 5: Run public-contract tests**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/orchestration/test_public_contracts.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit characterization tests**

```bash
git add tests/unit/temporal/test_monotonic_dp.py tests/unit/orchestration/test_public_contracts.py
git commit -m "test: characterize temporal alignment baseline"
```

---

### Task 2: Introduce Task-Agnostic Alignment Contracts and Configuration

**Files:**
- Create: `src/hcmai/common/schemas/alignment.py`
- Modify: `src/hcmai/common/schemas/__init__.py`
- Modify: `src/hcmai/common/config.py`
- Modify: `configs/baseline.yaml`
- Create: `tests/unit/common/test_alignment_contracts.py`

**Interfaces:**
- Produces: `AlignmentEvent`, `AlignmentPlan`, `AlignmentPath`, `AlignmentConfig`.
- Later tasks consume these exact types; they must not reference `TaskType`, progressive state, scenes, or alignment modes.

- [x] **Step 1: Write failing contract tests**

```python
# tests/unit/common/test_alignment_contracts.py
import pytest

from hcmai.common.schemas.alignment import AlignmentEvent, AlignmentPlan


def test_alignment_plan_requires_consecutive_event_order():
    with pytest.raises(ValueError, match="consecutive"):
        AlignmentPlan(
            events=(
                AlignmentEvent(event_id="e0", text="first", order=0),
                AlignmentEvent(event_id="e1", text="second", order=2),
            )
        )


def test_alignment_plan_is_task_agnostic():
    plan = AlignmentPlan(
        events=(
            AlignmentEvent(event_id="e0", text="first", order=0),
            AlignmentEvent(event_id="e1", text="second", order=1),
        )
    )
    assert [event.text for event in plan.events] == ["first", "second"]
```

- [x] **Step 2: Run the new tests and verify import failure**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/common/test_alignment_contracts.py -v
```

Expected: FAIL because `hcmai.common.schemas.alignment` does not exist.

- [x] **Step 3: Add the minimal alignment contracts**

```python
# src/hcmai/common/schemas/alignment.py
from __future__ import annotations

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString
from .frame import FrameRecord
from .search import SearchFilters


class AlignmentEvent(ContractModel):
    event_id: NonEmptyString
    text: NonEmptyString
    order: int = Field(ge=0)


class AlignmentPlan(ContractModel):
    events: tuple[AlignmentEvent, ...] = Field(min_length=1)
    filters: SearchFilters | None = None

    @model_validator(mode="after")
    def validate_event_order(self):
        orders = [event.order for event in self.events]
        if orders != list(range(len(self.events))):
            raise ValueError("alignment event order must be consecutive")
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("alignment event IDs must be unique")
        return self


class AlignmentPath(ContractModel):
    path_id: NonEmptyString
    video_id: NonEmptyString
    frames: tuple[FrameRecord, ...] = Field(min_length=1)
    event_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    score: float

    @model_validator(mode="after")
    def validate_path(self):
        if len(self.frames) != len(self.event_ids):
            raise ValueError("alignment path must contain one frame per event")
        if any(frame.video_id != self.video_id for frame in self.frames):
            raise ValueError("alignment path frames must share video_id")
        if any(
            current.timestamp_ms < previous.timestamp_ms
            for previous, current in zip(self.frames, self.frames[1:])
        ):
            raise ValueError("alignment path frames must be chronological")
        return self
```

- [x] **Step 4: Add `AlignmentConfig` and a transitional `SearchConfig.alignment` field**

Add before `SearchConfig` in `src/hcmai/common/config.py`:

```python
class AlignmentConfig(BaseModel):
    top_k: int = Field(default=500, ge=1)
    max_videos: int = Field(default=200, ge=1)
    rrf_k: int = Field(default=60, gt=0)
    lambda_gap: float = Field(default=1e-5, ge=0.0)
    event_power: float = Field(default=1.0, gt=0.0, le=1.0)
    chunk_size: int = Field(default=65_536, ge=1)
    cluster_delta: float = Field(default=0.0, ge=0.0)
```

Then add, without deleting progressive fields yet:

```python
alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
```

Add the matching `search.alignment` block to `configs/baseline.yaml` with the
same explicit values. The YAML is a runtime source of truth; relying on model
defaults would make the first baseline irreproducible.

- [x] **Step 5: Export the alignment contracts from `common.schemas`**

Add imports and `__all__` entries for:

```python
AlignmentEvent
AlignmentPlan
AlignmentPath
```

- [x] **Step 6: Run contract tests**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/common/test_alignment_contracts.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit the contracts and config**

```bash
git add src/hcmai/common/schemas/alignment.py src/hcmai/common/schemas/__init__.py src/hcmai/common/config.py configs/baseline.yaml tests/unit/common/test_alignment_contracts.py
git commit -m "refactor: add task agnostic alignment contracts"
```

---

### Task 3: Add a Deterministic Query Planner and Optional KIS Events

**Files:**
- Create: `src/hcmai/temporal/planner.py`
- Modify: `src/hcmai/common/schemas/search.py`
- Create: `tests/unit/temporal/test_planner.py`

**Interfaces:**
- Consumes: `query: str`, `events: list[str] | None`, `filters: SearchFilters | None`.
- Produces: `build_alignment_plan(query, events=None, filters=None) -> AlignmentPlan`.

- [x] **Step 1: Write planner tests for explicit events, multi-line queries, sentences, and single-event fallback**

```python
# tests/unit/temporal/test_planner.py
from hcmai.temporal.planner import build_alignment_plan


def texts(plan):
    return [event.text for event in plan.events]


def test_explicit_events_are_authoritative():
    plan = build_alignment_plan(
        "ignored for event splitting",
        [" chef holds skewer ", "chef coats it"],
    )
    assert texts(plan) == ["chef holds skewer", "chef coats it"]


def test_multiline_query_becomes_ordered_events():
    plan = build_alignment_plan("first event\nsecond event\nthird event")
    assert texts(plan) == ["first event", "second event", "third event"]


def test_sentence_query_becomes_ordered_events():
    plan = build_alignment_plan("First action. Then second action. Finally third action.")
    assert texts(plan) == ["First action", "Then second action", "Finally third action"]


def test_single_sentence_remains_one_event():
    plan = build_alignment_plan("person splashes water on face")
    assert texts(plan) == ["person splashes water on face"]
```

- [x] **Step 2: Run planner tests and verify failure**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/temporal/test_planner.py -v
```

Expected: FAIL because planner does not exist.

- [x] **Step 3: Implement deterministic planning**

```python
# src/hcmai/temporal/planner.py
from __future__ import annotations

import re

from hcmai.common.schemas import AlignmentEvent, AlignmentPlan, SearchFilters

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _normalize(values: list[str]) -> list[str]:
    return [value.strip().rstrip(".!?").strip() for value in values if value.strip()]


def build_alignment_plan(
    query: str,
    events: list[str] | None = None,
    filters: SearchFilters | None = None,
) -> AlignmentPlan:
    if events:
        parts = _normalize(events)
    else:
        lines = _normalize(query.splitlines())
        if len(lines) >= 2:
            parts = lines
        else:
            sentences = _normalize(_SENTENCE_BOUNDARY.split(query.strip()))
            parts = sentences if len(sentences) >= 2 else [query.strip()]

    return AlignmentPlan(
        events=tuple(
            AlignmentEvent(event_id=f"e{index}", text=text, order=index)
            for index, text in enumerate(parts)
        ),
        filters=filters,
    )
```

- [x] **Step 4: Add backwards-compatible optional events to KIS request**

In `SearchRequest`:

```python
events: list[NonEmptyString] | None = Field(default=None, min_length=1, max_length=10)
```

Do not remove `query`, `filters`, `top_k`, or `search_id`.

- [x] **Step 5: Run planner and public-contract tests**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/temporal/test_planner.py tests/unit/orchestration/test_public_contracts.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit query planning**

```bash
git add src/hcmai/temporal/planner.py src/hcmai/common/schemas/search.py tests/unit/temporal/test_planner.py
git commit -m "feat: add deterministic alignment query planner"
```

---

### Task 4: Generalize Visual Video Scoring and Preserve Search Filters

**Files:**
- Modify: `src/hcmai/retrieval/retriever/video_scores.py`
- Modify: `src/hcmai/retrieval/retriever/pipeline.py`
- Modify: `tests/unit/retriever/test_score_videos.py`

**Interfaces:**
- Produces: `RetrievalService.score_event_videos(events, filters, ...) -> list[VideoEventScores]`.
- Keeps `score_visual_videos()` temporarily as a compatibility wrapper until Task 8.

- [x] **Step 1: Add a failing test that filters shortlist frames by video and time**

Build a small fake index exposing the same methods used by `score_videos`:

```python
# tests/unit/retriever/test_score_videos.py
import numpy as np

from hcmai.common.schemas import SearchFilters
from hcmai.retrieval.retriever.video_scores import score_videos


class FakeIndex:
    video_ids = np.array(["V01", "V01", "V02", "V02"], dtype=object)
    frame_ids = np.array(["a", "b", "c", "d"], dtype=object)
    frame_idx = np.array([0, 1, 0, 1])
    timestamps = np.array([0, 1000, 0, 1000])

    def search_filtered(self, query_vectors, top_k, filters):
        assert filters.video_ids == ["V02"]
        return np.array([[0.9, 0.8]]), np.array([[2, 3]])

    def filtered_positions(self, filters):
        return np.array([3], dtype=np.int64)

    def video_positions(self, video_id):
        return np.array([2, 3], dtype=np.int64)

    def score_subset(self, query_vectors, positions, chunk_size):
        return np.full((len(query_vectors), len(positions)), 0.75, dtype=np.float32)


def test_score_videos_respects_video_and_time_filters():
    results = score_videos(
        FakeIndex(),
        np.array([[1.0, 0.0]], dtype=np.float32),
        top_k=10,
        max_videos=10,
        filters=SearchFilters(video_ids=["V02"], start_time_ms=1000),
    )

    assert [result.video_id for result in results] == ["V02"]
    assert results[0].frame_ids.tolist() == ["d"]
```

- [x] **Step 2: Update the existing fake index before running the full unit file**

`tests/unit/retriever/test_score_videos.py` already exercises unfiltered
shortlisting. Extend its `_FakeIndex` with `search_filtered()` delegating to
`search()` when `filters is None`, and `filtered_positions()` returning
`None`. This preserves the existing assertions while making the fake match the
real `DenseIndex` protocol.

- [x] **Step 3: Run the filter test and verify signature failure**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/retriever/test_score_videos.py -v
```

Expected: FAIL because `score_videos` does not accept `filters` and uses unfiltered `search()`.

- [x] **Step 4: Add `filters` to `score_videos` and use existing DenseIndex filter primitives**

Change the shortlist search to:

```python
_, positions = index.search_filtered(query_vectors, top_k, filters)
```

Before rescoring each shortlisted video, intersect its canonical positions with the allowed positions:

```python
allowed = index.filtered_positions(filters)

windows = []
kept_video_ids = []
for video_id in shortlist:
    window = index.video_positions(video_id)
    if allowed is not None:
        window = window[np.isin(window, allowed)]
    if len(window):
        kept_video_ids.append(video_id)
        windows.append(window)
```

Build results from `kept_video_ids`, not the unfiltered shortlist.

- [x] **Step 5: Add `RetrievalService.score_event_videos`**

```python
def score_event_videos(
    self,
    events: Sequence[str],
    filters: SearchFilters | None = None,
    top_k: int = 500,
    max_videos: int = 200,
    rrf_k: int = 60,
    chunk_size: int = 65_536,
) -> list[VideoEventScores]:
    if not events:
        raise ValueError("events must not be empty")
    visual = self._retriever_for("visual")
    batch = visual.encode(list(events))
    return score_videos(
        visual.index,
        batch.vectors,
        top_k,
        max_videos,
        rrf_k,
        chunk_size,
        filters=filters,
    )
```

Keep `score_visual_videos()` as a wrapper calling `score_event_videos(events, None, ...)` until all callers migrate.

- [x] **Step 6: Run retrieval tests**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/retriever/test_score_videos.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit filter-aware event scoring**

```bash
git add src/hcmai/retrieval/retriever/video_scores.py src/hcmai/retrieval/retriever/pipeline.py tests/unit/retriever/test_score_videos.py
git commit -m "refactor: expose filter aware event video scores"
```

---

### Task 5: Consolidate the Pure DP Into `temporal/dp.py`

**Files:**
- Create: `src/hcmai/temporal/dp.py`
- Modify: `tests/unit/temporal/test_monotonic_dp.py`

**Interfaces:**
- Consumes: `VideoEventScores`.
- Produces: `DPPath`, `align_video()`, `rank_paths()` with the same numerical behavior as the existing TRAKE implementation.

- [x] **Step 1: Change characterization test imports to the new module**

```python
from hcmai.temporal.dp import align_video
```

- [x] **Step 2: Run the tests to verify the new module is missing**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/temporal/test_monotonic_dp.py -v
```

Expected: FAIL on import.

- [x] **Step 3: Move the pure algorithm without behavior changes**

Copy the algorithmic contents of `temporal/aligners/monotonic_dp.py` into `temporal/dp.py` and rename only the internal dataclass:

```python
@dataclass(frozen=True, slots=True)
class DPPath:
    video_id: str
    score: float
    frame_idx: tuple[int, ...]
    frame_ids: tuple[str, ...]
```

Keep the current recurrence, `cluster_starts()`, gap penalty, event power, ranking, and diversification semantics unchanged in this task.

- [x] **Step 4: Run characterization tests**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/temporal/test_monotonic_dp.py -v
```

Expected: PASS with exactly the same expected frame paths.

- [ ] **Step 5: Commit the pure DP module**

```bash
git add src/hcmai/temporal/dp.py tests/unit/temporal/test_monotonic_dp.py
git commit -m "refactor: isolate monotonic alignment algorithm"
```

---

### Task 6: Build the Stateless `TemporalAlignmentService`

**Files:**
- Create: `src/hcmai/temporal/service.py`
- Modify: `src/hcmai/temporal/__init__.py`
- Create: `tests/unit/temporal/test_service.py`

**Interfaces:**
- Consumes: `AlignmentPlan`, `max_paths: int`.
- Produces: `AlignmentResult(paths: tuple[AlignmentPath, ...], candidate_video_count: int)`.
- Dependencies: `DataService`, `RetrievalService`, `AlignmentConfig`.

- [x] **Step 1: Write a failing service test with fake retrieval and data dependencies**

```python
# tests/unit/temporal/test_service.py
from types import SimpleNamespace

import numpy as np

from hcmai.common.schemas import AlignmentEvent, AlignmentPlan, FrameRecord
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.service import TemporalAlignmentService


class FakeRetrieval:
    def score_event_videos(self, events, filters=None, **kwargs):
        assert events == ["first", "second"]
        return [VideoEventScores(
            video_id="V01",
            frame_ids=np.array(["f0", "f1"], dtype=object),
            frame_idx=np.array([0, 1]),
            timestamps_ms=np.array([0, 1000]),
            scores=np.array([[0.9, 0.1], [0.1, 0.8]]),
        )]


class FakeData:
    frames = {
        "f0": FrameRecord(frame_id="f0", video_id="V01", frame_idx=0, timestamp_ms=0, image_path="f0.jpg", width=640, height=360),
        "f1": FrameRecord(frame_id="f1", video_id="V01", frame_idx=1, timestamp_ms=1000, image_path="f1.jpg", width=640, height=360),
    }

    def get_frame(self, frame_id):
        return self.frames[frame_id]


def test_alignment_service_materializes_canonical_path():
    config = SimpleNamespace(
        top_k=500,
        max_videos=200,
        rrf_k=60,
        lambda_gap=0.0,
        event_power=1.0,
        chunk_size=65536,
        cluster_delta=0.0,
    )
    service = TemporalAlignmentService(FakeData(), FakeRetrieval(), config)
    plan = AlignmentPlan(events=(
        AlignmentEvent(event_id="e0", text="first", order=0),
        AlignmentEvent(event_id="e1", text="second", order=1),
    ))

    result = service.align(plan, max_paths=5)

    assert len(result.paths) == 1
    assert [frame.frame_id for frame in result.paths[0].frames] == ["f0", "f1"]
    assert result.paths[0].event_ids == ("e0", "e1")
```

Add two negative cases to this same unit file before implementation:

1. a score matrix whose row count differs from the `AlignmentPlan` event
   count; and
2. a score-matrix column whose `frame_id`, `video_id`, `frame_idx`, or
   `timestamp_ms` disagrees with `FakeData`.

Both must raise before a public `AlignmentPath` is returned. This carries the
current `MonotonicOrderedPathAligner` identity check into the replacement
facade instead of trusting index metadata at the task boundary.

- [x] **Step 2: Run service test and verify import failure**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/temporal/test_service.py -v
```

Expected: FAIL because `temporal.service` does not exist.

- [x] **Step 3: Implement the service as the single temporal facade**

```python
# src/hcmai/temporal/service.py
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

from hcmai.common.config import AlignmentConfig
from hcmai.common.schemas import AlignmentPath, AlignmentPlan
from hcmai.data.pipeline import DataService
from hcmai.retrieval.retriever.pipeline import RetrievalService

from .dp import rank_paths


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    plan: AlignmentPlan
    paths: tuple[AlignmentPath, ...]
    candidate_video_count: int


class TemporalAlignmentService:
    def __init__(self, data: DataService, retrieval: RetrievalService, config: AlignmentConfig):
        self.data = data
        self.retrieval = retrieval
        self.config = config

    def align(self, plan: AlignmentPlan, *, max_paths: int) -> AlignmentResult:
        if max_paths <= 0:
            raise ValueError("max_paths must be greater than zero")

        scores = self.retrieval.score_event_videos(
            [event.text for event in plan.events],
            filters=plan.filters,
            top_k=self.config.top_k,
            max_videos=self.config.max_videos,
            rrf_k=self.config.rrf_k,
            chunk_size=self.config.chunk_size,
        )
        for video in scores:
            self._validate_video_scores(plan, video)
        rows = rank_paths(
            scores,
            lambda_gap=self.config.lambda_gap,
            max_rows=max_paths,
            event_power=self.config.event_power,
            cluster_delta=self.config.cluster_delta,
        )
        event_ids = tuple(event.event_id for event in plan.events)
        paths = []
        for row in rows:
            frames = tuple(self.data.get_frame(frame_id) for frame_id in row.frame_ids)
            digest = sha1(f"{row.video_id}\0{'|'.join(row.frame_ids)}".encode()).hexdigest()[:16]
            paths.append(AlignmentPath(
                path_id=f"path-{digest}",
                video_id=row.video_id,
                frames=frames,
                event_ids=event_ids,
                score=row.score,
            ))
        return AlignmentResult(plan=plan, paths=tuple(paths), candidate_video_count=len(scores))

    def _validate_video_scores(self, plan, video) -> None:
        """Reject matrix columns that conflict with canonical frame metadata."""
        frame_count = len(video.frame_ids)
        if video.scores.shape != (len(plan.events), frame_count):
            raise ValueError("dense score matrix shape does not match alignment plan")
        if not (
            len(video.frame_idx) == frame_count
            and len(video.timestamps_ms) == frame_count
        ):
            raise ValueError("dense score metadata arrays must have equal lengths")
        for position, frame_id in enumerate(video.frame_ids):
            frame = self.data.get_frame(str(frame_id))
            if frame.video_id != video.video_id:
                raise ValueError("dense score frame has mixed canonical video identity")
            if frame.frame_idx != int(video.frame_idx[position]):
                raise ValueError("dense score frame_idx conflicts with canonical data")
            if frame.timestamp_ms != round(float(video.timestamps_ms[position])):
                raise ValueError("dense score timestamp conflicts with canonical data")
```

- [x] **Step 4: Export only the new service/planner from `temporal.__init__` during migration**

Add:

```python
from .planner import build_alignment_plan
from .service import AlignmentResult, TemporalAlignmentService
```

Keep old exports temporarily until Task 8 removes old callers.

- [x] **Step 5: Run service and DP tests**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/temporal/test_service.py tests/unit/temporal/test_monotonic_dp.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit the stateless service**

```bash
git add src/hcmai/temporal/service.py src/hcmai/temporal/__init__.py tests/unit/temporal/test_service.py
git commit -m "feat: add stateless temporal alignment service"
```

---

### Task 7: Migrate TRAKE First Because It Already Matches DP Semantics

**Files:**
- Modify: `src/hcmai/orchestration/workflows/trake.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Create: `tests/unit/orchestration/test_trake_pipeline.py`
- Modify: `tests/integration/test_trake_api.py`

**Interfaces:**
- Consumes: `TemporalAlignmentService`, `TRAKERequest.events`.
- Produces: unchanged `TRAKEResponse` public structure.

- [x] **Step 1: Write a fake-alignment TRAKE workflow test**

```python
# tests/unit/orchestration/test_trake_pipeline.py
from types import SimpleNamespace

from hcmai.common.schemas import AlignmentPath, FrameRecord, TRAKERequest
from hcmai.orchestration.workflows.trake import TRAKEPipeline


class FakeAlignment:
    def align(self, plan, *, max_paths):
        frames = (
            FrameRecord(frame_id="f0", video_id="V01", frame_idx=10, timestamp_ms=1000, image_path="f0.jpg", width=640, height=360),
            FrameRecord(frame_id="f1", video_id="V01", frame_idx=20, timestamp_ms=2000, image_path="f1.jpg", width=640, height=360),
        )
        return SimpleNamespace(paths=(AlignmentPath(
            path_id="path-1",
            video_id="V01",
            frames=frames,
            event_ids=("e0", "e1"),
            score=1.5,
        ),))


def test_trake_pipeline_uses_shared_alignment_service():
    pipeline = TRAKEPipeline(FakeAlignment())
    response = pipeline.execute(TRAKERequest(
        query="first then second",
        events=["first", "second"],
        top_k=5,
    ))

    assert response.total_results == 1
    assert response.submissions[0].frame_ids == ["f0", "f1"]
```

- [x] **Step 2: Run test and verify it fails against old constructor/flow**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/orchestration/test_trake_pipeline.py -v
```

Expected: FAIL because TRAKE still expects `TemporalEvidenceCore` and `ordered_plan()`.

- [x] **Step 3: Replace TRAKE temporal-core usage with planner + alignment service**

In `TRAKEPipeline.execute()`:

```python
plan = build_alignment_plan(request.query, request.events)
aligned = self.alignment.align(plan, max_paths=request.top_k)
rows = aligned.paths
```

Keep the current submission materialization logic.

- [x] **Step 4: Change constructor to one dependency**

```python
def __init__(self, alignment: TemporalAlignmentService | None) -> None:
    self.alignment = alignment
```

Use `TaskPipelineDependencyError("Alignment service not loaded")` when absent.

- [x] **Step 5: Update the default registry to construct one `TemporalAlignmentService` and inject it into TRAKE**

Do not remove KIS's old temporal core in this task; run both services side-by-side temporarily.

- [x] **Step 6: Update the existing route-level fake to the generic protocol**

`tests/integration/test_trake_api.py` currently provides
`score_visual_videos()`. Change the fake to `score_event_videos(events,
filters=None, **kwargs)` and assert TRAKE passes `filters=None`. Keep the
end-to-end assertion that every submission contains one canonical frame per
event and has nondecreasing timestamps.

- [x] **Step 7: Run TRAKE tests plus DP/service tests**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/orchestration/test_trake_pipeline.py tests/unit/temporal -v
```

Expected: PASS.

- [ ] **Step 8: Commit TRAKE migration**

```bash
git add src/hcmai/orchestration/workflows/trake.py src/hcmai/orchestration/pipeline.py tests/unit/orchestration/test_trake_pipeline.py tests/integration/test_trake_api.py
git commit -m "refactor: run trake through shared alignment service"
```

---

### Task 8: Migrate KIS to Stateless DP Alignment

**Files:**
- Modify: `src/hcmai/orchestration/workflows/kis.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Create: `tests/unit/orchestration/test_kis_pipeline.py`
- Modify: `tests/integration/test_kis_golden_path.py`
- Modify or replace: `tests/unit/orchestration/test_kis_reranking.py`
- Modify: `tests/unit/orchestration/test_request_scoped_latency.py`

**Interfaces:**
- Consumes: `SearchRequest.query`, optional `SearchRequest.events`, filters, `TemporalAlignmentService`.
- Produces: existing `SearchResponse`; each result's `frame_ids` contains the entire aligned path and `scores.final` equals path score.

- [x] **Step 1: Write a failing KIS path-projection test**

```python
# tests/unit/orchestration/test_kis_pipeline.py
from types import SimpleNamespace

from hcmai.common.schemas import AlignmentPath, FrameRecord, SearchRequest
from hcmai.orchestration.workflows.kis import KISPipeline


class FakeAlignment:
    def align(self, plan, *, max_paths):
        frames = tuple(
            FrameRecord(
                frame_id=f"f{index}",
                video_id="V01",
                frame_idx=index,
                timestamp_ms=index * 1000,
                image_path=f"f{index}.jpg",
                width=640,
                height=360,
            )
            for index in range(3)
        )
        return SimpleNamespace(paths=(AlignmentPath(
            path_id="path-1",
            video_id="V01",
            frames=frames,
            event_ids=("e0", "e1", "e2"),
            score=2.4,
        ),))


class FakeData:
    def get_frame(self, frame_id):
        index = int(frame_id[1:])
        return FrameRecord(
            frame_id=frame_id,
            video_id="V01",
            frame_idx=index,
            timestamp_ms=index * 1000,
            image_path=f"{frame_id}.jpg",
            width=640,
            height=360,
        )

    def get_evidence(self, frame_id, source):
        return None


def test_kis_returns_middle_frame_and_preserves_alignment_path():
    pipeline = KISPipeline(FakeData(), FakeAlignment())
    response = pipeline.execute(SearchRequest(
        query="first. second. third.",
        events=["first", "second", "third"],
        top_k=5,
    ))

    assert response.total_results == 1
    assert response.results[0].frame_id == "f1"
    assert response.results[0].frame_ids == ["f0", "f1", "f2"]
    assert response.results[0].scores.final == 2.4
```

- [x] **Step 2: Run KIS test and verify failure against old constructor/scene flow**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/orchestration/test_kis_pipeline.py -v
```

Expected: FAIL.

- [x] **Step 3: Reject ambiguous `min_score` filters before alignment**

Add this request validation before building the plan:

```python
if request.filters is not None and request.filters.min_score is not None:
    raise TaskPipelineRequestError(
        "min_score is not supported for multi-event alignment; use video/time filters"
    )
```

- [x] **Step 4: Replace KIS `localize()`/scene logic with one alignment call**

The core execute flow becomes:

```python
plan = build_alignment_plan(request.query, request.events, request.filters)
aligned = self.alignment.align(plan, max_paths=request.top_k)
candidates = [_path_to_candidate(path) for path in aligned.paths]
```

Remove calls to:

```text
TemporalEvidenceCore.localize
_representative_candidates(scene)
KISPipeline._rerank
```

- [x] **Step 5: Add a simple deterministic path-to-frame projection**

```python
def _path_to_candidate(path: AlignmentPath) -> RetrievalCandidate:
    position = len(path.frames) // 2
    frame = path.frames[position]
    return RetrievalCandidate(
        frame_id=frame.frame_id,
        final_score=path.score,
        metadata={
            "path_id": path.path_id,
            "event_ids": list(path.event_ids),
            "frame_ids": [item.frame_id for item in path.frames],
        },
    )
```

Do not synthesize visual/context source scores from the path in this baseline.

- [x] **Step 6: Keep `search_id` compatible without server state**

At response materialization time:

```python
response_search_id = request.search_id or f"search-{uuid4().hex}"
response_request = request.model_copy(update={"search_id": response_search_id})
```

No state store lookup or versioning is allowed.

- [x] **Step 7: Simplify the KIS constructor**

Target constructor:

```python
def __init__(self, data: DataService | None, alignment: TemporalAlignmentService | None):
    self.data = data
    self.alignment = alignment
    self.materializer = SearchMaterializer(data) if data is not None else None
```

Remove direct KIS ownership of `RetrievalService`, `SearchConfig`, `TemporalEvidenceCore`, and `RerankingService`.

- [x] **Step 8: Update default registry so KIS and TRAKE receive the same alignment-service instance**

```python
alignment = (
    TemporalAlignmentService(self.data, self.retrieval, self.config.alignment)
    if self.data is not None and self.retrieval is not None
    else None
)

pipelines = [
    KISPipeline(self.data, alignment),
    TRAKEPipeline(alignment),
]
```

- [x] **Step 9: Simplify KIS telemetry around one alignment stage**

Keep parse and materialization timers, but replace progressive budget diagnostics with one timer around `alignment.align()`. Record its duration in `latency_ms.temporal_refinement` and use backend `monotonic_dp`. Remove references to `candidate_pool_size`, `top_m_evidence`, `scene_top_p_global`, progressive diff modes, and rerank timing from the KIS workflow.

- [x] **Step 10: Run KIS, TRAKE, planner, service, and DP tests together**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest \
  tests/unit/orchestration/test_kis_pipeline.py \
  tests/unit/orchestration/test_trake_pipeline.py \
  tests/unit/temporal \
  -v
```

Expected: PASS.

- [x] **Step 11: Replace, rather than preserve, obsolete KIS behavior tests**

Rewrite `tests/integration/test_kis_golden_path.py` around deterministic
alignment-path projection and canonical IDs. Replace the former
reranker-specific assertions in `tests/unit/orchestration/test_kis_reranking.py`
with a regression that the default KIS score is the DP path score and that no
reranker stage is present. Update request-scoped latency assertions to use the
alignment stage; do not retain an assertion for fusion/rerank timing that the
new KIS flow does not emit.

- [ ] **Step 12: Commit KIS migration**

```bash
git add src/hcmai/orchestration/workflows/kis.py src/hcmai/orchestration/pipeline.py tests/unit/orchestration/test_kis_pipeline.py tests/integration/test_kis_golden_path.py tests/unit/orchestration/test_kis_reranking.py tests/unit/orchestration/test_request_scoped_latency.py
git commit -m "refactor: run kis through shared monotonic alignment"
```

---

### Task 9: Remove Default Single-Frame Reranker Wiring

**Files:**
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/common/config.py`
- Create: `tests/unit/orchestration/test_registry.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: default `SearchService` with only data, retrieval, config, and optional LLM used by retrieval infrastructure; no default reranking dependency.
- Keeps: `src/hcmai/retrieval/reranking/` package untouched for explicit experiments.

- [x] **Step 1: Write a registry test that asserts both task heads share one alignment service and no reranker is required**

Use fake data/retrieval dependencies and inspect the constructed pipelines. The assertion must verify:

```python
assert kis.alignment is trake.alignment
```

and KIS has no `reranking` attribute.

- [x] **Step 2: Run the registry test and verify failure**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/orchestration/test_registry.py -v
```

Expected: FAIL against current constructor/wiring.

- [x] **Step 3: Stop creating `RerankingService` in `load_search_service()`**

Delete the block guarded by:

```python
if llm is not None and data is not None and settings.search.rerank_count > 0:
```

Construct `SearchService` without a `reranking` argument.

- [x] **Step 4: Remove reranking from `SearchService.__init__` and instance state**

Delete imports and constructor fields only when no caller remains.

- [x] **Step 5: Remove search-level reranker settings**

Delete from `SearchConfig`:

```text
rerank_count
reranker
```

Delete `RerankerPolicyConfig` if `rg` confirms it has no remaining references.

- [x] **Step 6: Run orchestration tests**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/orchestration tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit default reranker detachment**

```bash
git add src/hcmai/orchestration/setup.py src/hcmai/orchestration/pipeline.py src/hcmai/common/config.py tests/unit/orchestration/test_registry.py tests/test_config.py
git commit -m "refactor: detach single frame reranker from default search"
```

---

### Task 10: Delete Progressive State, Scene Alignment, Backfill, and Legacy Temporal Modes

**Files:**
- Delete: `src/hcmai/temporal/core.py`
- Delete: `src/hcmai/temporal/ports.py`
- Delete: `src/hcmai/temporal/query.py`
- Delete: `src/hcmai/temporal/settings.py`
- Delete: `src/hcmai/temporal/providers/dense.py`
- Delete: `src/hcmai/temporal/providers/sparse.py`
- Delete: `src/hcmai/temporal/aligners/monotonic.py`
- Delete: `src/hcmai/temporal/aligners/monotonic_dp.py`
- Delete: `src/hcmai/temporal/aligners/scene.py`
- Delete: `src/hcmai/temporal/state/evidence.py`
- Delete: `src/hcmai/temporal/state/state.py`
- Delete: `src/hcmai/temporal/utils/relations.py`
- Delete: `src/hcmai/temporal/utils/scoring.py`
- Modify: `src/hcmai/temporal/__init__.py`
- Modify: `src/hcmai/common/schemas/__init__.py`
- Modify: `src/hcmai/common/schemas/temporal.py` or delete it if no nonlegacy symbols remain
- Modify: `src/hcmai/common/config.py`
- Modify: `configs/baseline.yaml`
- Delete or rewrite: legacy-only temporal tests under `tests/unit/temporal/`
- Modify: `tests/integration/test_progressive_temporal_core.py`

**Interfaces:**
- Consumes: migrated KIS/TRAKE code from Tasks 7-9.
- Produces: no runtime concept of scene candidates, progressive state, query snapshots, binary evaluation states, backfill, or task-specific alignment mode.

- [x] **Step 1: Prove no migrated runtime caller imports legacy symbols**

Run:

```bash
rg -n "TemporalEvidenceCore|ProgressiveEvidenceState|ProgressiveSearchState|ProgressiveSceneAligner|SceneCandidate|TemporalAlignmentMode|ProgressiveLocalizationResult|parse_temporal_constraints|score_scene" src/hcmai/orchestration src/hcmai/temporal/service.py src/hcmai/temporal/planner.py
```

Expected: no matches in migrated runtime files.

- [x] **Step 2: Delete the legacy files listed above**

Do not delete `temporal/dp.py`, `temporal/planner.py`, or `temporal/service.py`.

- [x] **Step 3: Replace `temporal/__init__.py` with minimal exports**

```python
"""Stateless ordered event-to-frame alignment."""

from .planner import build_alignment_plan
from .service import AlignmentResult, TemporalAlignmentService

__all__ = [
    "AlignmentResult",
    "TemporalAlignmentService",
    "build_alignment_plan",
]
```

- [x] **Step 4: Remove legacy temporal schema exports**

Remove exports for:

```text
QueryUnit
FrameEvidence
SceneCandidate
TemporalAlignmentMode
TemporalConstraint
TemporalQueryPlan
TemporalRelation
OrderedPathCandidate
```

After `rg` shows no callers, delete `common/schemas/temporal.py` entirely; the replacement types live in `common/schemas/alignment.py`.

- [x] **Step 5: Remove progressive configuration**

Delete `ProgressiveSearchConfig` and these `SearchConfig` fields:

```text
candidate_count
temporal_window_ms
progressive
```

The final search config keeps:

```python
fusion: FusionConfig
cache: RetrievalCacheConfig
alignment: AlignmentConfig
```

plus any unrelated settings still used elsewhere.

Remove the matching `candidate_count`, `rerank_count`, `temporal_window_ms`,
`reranker`, and `progressive` keys from `configs/baseline.yaml`, leaving its
explicit `search.alignment` block. Configure Pydantic models with
`extra="forbid"` at this boundary, or add an equivalent config-load test, so a
deleted experiment setting cannot be silently accepted and ignored.

- [x] **Step 6: Remove now-empty `temporal/providers`, `temporal/state`, `temporal/utils`, and `temporal/aligners` packages if no files remain**

Do not keep empty package directories merely for compatibility; all internal callers have already migrated.

- [x] **Step 7: Audit and remove or rewrite every legacy-only test**

Delete tests whose sole subject is a removed contract, including the current
progressive state/diff/relation/scene tests:

```text
tests/unit/temporal/test_config_identity.py
tests/unit/temporal/test_core_regressions.py
tests/unit/temporal/test_plan04_convergence.py
tests/unit/temporal/test_query_evidence.py
tests/unit/temporal/test_scoring_relations.py
tests/unit/temporal/test_state.py
tests/integration/test_progressive_temporal_core.py
```

Before deleting a test, transfer any still-relevant invariant—especially
canonical identity, chronological output, request validation, and no hidden
fallback—to the new planner/service/workflow tests. Use `rg` to locate any
additional test import of a deleted module; the list is an inventory as of
2026-08-30, not an assumption that it is exhaustive.

- [x] **Step 8: Verify legacy symbols are gone from Python runtime code**

Run:

```bash
rg -n "UNKNOWN|EVALUATED_NO_MATCH|MATCHED|backfill_max|scene_top_|scene_max_|candidate_match_weight|ProgressiveSearch|ProgressiveScene|TemporalEvidenceCore" src/hcmai --glob '*.py'
```

Expected: no matches related to the deleted temporal implementation.

- [x] **Step 9: Run all Python tests**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests -v
```

Expected: PASS.

- [ ] **Step 10: Commit the deletion separately for easy review/revert**

```bash
git add src/hcmai/temporal src/hcmai/common/config.py src/hcmai/common/schemas configs/baseline.yaml tests/unit/temporal tests/integration/test_progressive_temporal_core.py
git commit -m "refactor: remove progressive scene temporal pipeline"
```

---

### Task 11: Remove the Old TRAKE-Named Retrieval API and Settings

**Files:**
- Modify: `src/hcmai/retrieval/retriever/pipeline.py`
- Modify: `src/hcmai/retrieval/retriever/video_scores.py`
- Modify: `src/hcmai/common/config.py`
- Modify: `tests/unit/retriever/test_score_videos.py`

**Interfaces:**
- Produces: only generic `score_event_videos`; no runtime `TRAKESettings` or `score_visual_videos` naming remains.

- [x] **Step 1: Find compatibility callers**

Run:

```bash
rg -n "score_visual_videos|TRAKESettings" src/hcmai tests
```

Expected after prior tasks: only compatibility definition(s), no task workflow callers.

- [x] **Step 2: Delete `RetrievalService.score_visual_videos` compatibility wrapper**

Keep `score_event_videos` as the only public video-scoring API.

- [x] **Step 3: Confirm `temporal/settings.py` and `TRAKESettings` are already gone**

Run:

```bash
rg -n "TRAKESettings" src/hcmai
```

Expected: no matches.

- [x] **Step 4: Run retrieval and service tests**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests/unit/retriever tests/unit/temporal -v
```

Expected: PASS.

- [ ] **Step 5: Commit generic naming cleanup**

```bash
git add src/hcmai/retrieval/retriever/pipeline.py src/hcmai/retrieval/retriever/video_scores.py src/hcmai/common/config.py tests/unit/retriever/test_score_videos.py
git commit -m "refactor: make event video scoring task agnostic"
```

---

### Task 12: Rewrite Temporal Documentation Around the Research Baseline

**Files:**
- Modify: `README.md`
- Rewrite: `src/hcmai/temporal/README.md`
- Modify: `src/hcmai/README.md`
- Modify: `src/hcmai/common/schemas/README.md`
- Create: `docs/research/alignment-baseline.md`

**Interfaces:**
- Produces: one documentation path explaining the baseline, its assumptions, and where future research modules attach.

- [x] **Step 1: Replace the temporal README structure**

The new README must contain these exact conceptual sections:

```text
1. Problem definition: ordered event-to-frame alignment
2. Query planning
3. Candidate video scoring
4. Event x frame score matrix
5. Monotonic DP
6. KIS projection
7. TRAKE projection
8. Known limitations
9. Research extension points
```

Do not document deleted states, scene budgets, rescued-video backfill, or relation scoring.

- [x] **Step 2: Add a research-baseline note with the baseline equation and explicit non-capabilities**

Document:

```text
Baseline score = event/frame visual similarity + temporal gap penalty.

Not modeled yet:
- entity identity continuity,
- object state transitions,
- multimodal dense alignment,
- multi-frame VLM verification,
- incremental DP.
```

This file becomes the reference for ablation experiment naming.

- [x] **Step 3: Update root/schema docs so `AlignmentPlan` and `AlignmentPath` replace scene/progressive terminology**

- [x] **Step 4: Search documentation for stale legacy descriptions**

Run:

```bash
rg -n "UNKNOWN|EVALUATED_NO_MATCH|rescued|backfill|ProgressiveScene|scene_top_|progressive scene" src/hcmai docs
```

Expected: matches only in the historical design/plan documents under `docs/superpowers`, not current runtime documentation.

- [ ] **Step 5: Commit documentation cleanup**

```bash
git add README.md src/hcmai/temporal/README.md src/hcmai/README.md src/hcmai/common/schemas/README.md docs/research/alignment-baseline.md
git commit -m "docs: document unified temporal alignment baseline"
```

---

### Task 13: Final Verification and Research-Readiness Gate

**Files:**
- No feature files created.
- May modify only tests/docs if verification exposes a concrete defect.

**Interfaces:**
- Produces: evidence that the refactor is import-clean, test-clean, and free of legacy runtime semantics.

- [x] **Step 1: Compile all Python modules**

Run:

```bash
PYTHONPATH=src aic/bin/python -m compileall -q src/hcmai
```

Expected: exit code 0.

- [x] **Step 2: Run the complete Python test suite**

Run:

```bash
PYTHONPATH=src aic/bin/python -m pytest tests -v
```

Expected: PASS.

- [x] **Step 3: Confirm KIS and TRAKE both depend on the same service type**

Run:

```bash
rg -n "TemporalAlignmentService" src/hcmai/orchestration/workflows src/hcmai/orchestration/pipeline.py
```

Expected: KIS, TRAKE, and registry references; no `TemporalEvidenceCore`.

- [x] **Step 4: Confirm the legacy temporal implementation is absent**

Run:

```bash
find src/hcmai/temporal -type f | sort
```

Expected runtime source set is limited to:

```text
src/hcmai/temporal/__init__.py
src/hcmai/temporal/dp.py
src/hcmai/temporal/planner.py
src/hcmai/temporal/service.py
src/hcmai/temporal/README.md
```

`__pycache__` files are not source and should not be committed.

- [x] **Step 5: Confirm deleted config concepts cannot be referenced**

Run:

```bash
rg -n "candidate_count|temporal_window_ms|progressive_state|backfill_max|scene_max_|scene_top_|candidate_evaluation_weight" src/hcmai --glob '*.py'
```

Expected: no matches.

- [x] **Step 6: Confirm no default reranker overwrites KIS alignment score**

Run:

```bash
rg -n "reranker_score|_rerank|RerankingService" src/hcmai/orchestration
```

Expected: no matches in default orchestration.

- [ ] **Step 7: Run two manual smoke cases against a loaded local service**

KIS case:

```text
Query events:
1. chef holds a skewered ingredient
2. chef rolls it in chopped green and red mixture
3. ingredient is moved onto white flour
4. chef rotates it through the flour
5. ingredient is fully coated and placed aside

Expected invariant:
SearchResult.frame_ids is chronological and has one frame per planned event.
```

TRAKE case:

```text
Events: ["person enters room", "person sits down"]
Expected invariant:
TRAKESubmission.timestamps_ms[0] <= TRAKESubmission.timestamps_ms[1].
```

- [ ] **Step 8: Record baseline metrics before adding any research hypothesis**

For the development query set, record at minimum:

```text
Recall@K / task metric
mean candidate videos
mean aligned path span
mean events per query
p50 / p95 total latency
```

Store results outside runtime code in the project's existing experiment/results location. If no experiment directory exists, create `artifacts/evaluation/alignment_baseline/` rather than adding a new Python subsystem.

- [ ] **Step 9: Commit verification-only fixes if any were required**

If no fixes were needed, do not create an empty commit. If fixes were needed:

```bash
git status --short
# Then stage only the concrete test/source/documentation files fixed here.
git commit -m "fix: close unified alignment verification gaps"
```

---

## Post-Refactor Research Sequence

Do **not** mix these into the cleanup branch. Run them as separate experiments/plans in this order so each result is attributable:

1. **Multimodal score matrix:** replace visual-only event/frame scores with visual + context + ASR evidence while keeping DP unchanged.
2. **Entity continuity transition:** add a transition term that rewards the same chef/object across adjacent aligned events.
3. **Object state transition:** score expected changes such as uncoated -> flour-coated without requiring appearance identity.
4. **Top-B paths + multi-frame VLM verification:** verify complete paths instead of one representative image.
5. **Incremental DP:** cache prior event layers only if profiling shows stateless recomputation is a latency bottleneck.
6. **VQA head:** consume the same `AlignmentPath` and answer from aligned frames/clip; do not add a second retrieval path.

Each experiment must compare against the exact baseline produced by this plan.
