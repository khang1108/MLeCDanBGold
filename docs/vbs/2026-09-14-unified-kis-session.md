# Unified KIS Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one revisioned KIS interaction flow where KIS-T behaves as a normal one-shot search and KIS-C appends clues to the same logical search intent without introducing server-side mutable sessions.

**Architecture:** The frontend owns the ordered KIS input list and sends the full list on every revision to `POST /api/v1/kis/search`. The backend deterministically rebuilds a `KISIntent`, then delegates the resolved ordered events to the existing `KISPipeline`/`TemporalSearchService`; `/api/v1/search` remains unchanged as the stateless baseline. A dedicated frontend KIS session helper and panel keep live KIS state separate from Query History and from future EventTrail interaction state.

**Tech Stack:** Python 3 + FastAPI + Pydantic, existing HCMAI temporal retrieval pipeline, React 19 + react-scripts/Jest + Testing Library.

**Spec:** `docs/superpowers/specs/2026-09-14-unified-kis-session-design.md`

## Global Constraints

- KIS-T usually starts with one sufficiently complete textual query.
- KIS-C starts with an incomplete clue and receives additional clues over time.
- Both may be refined repeatedly.
- A new clue/refinement updates the current intent; it does not create a new logical KIS session.
- Existing `/api/v1/search` remains a stateless retrieval/debug endpoint.
- EventTrail actions such as anchor/reject/verify are explicitly out of scope for this phase.
- Do not add server-side mutable KIS session storage, TTL handling, Redis/SQLite state, or worker synchronization in this phase.
- Do not model KIS-C in the frontend as `old_query + new_query`; the backend owns deterministic intent construction from ordered inputs.
- The deterministic baseline must not claim to perfectly resolve pronouns, contradictions, entity identity, or event rewrites.
- Image search remains separate; KIS-V unification is not part of this phase.
- Existing Query History remains replay/persistence state, not the live KIS session domain model.
- A failed revision must not commit the draft input locally.
- `New Search` must abort the active request and reset the KIS drafting session to revision 0.
- Existing `/api/v1/search` request/response compatibility must be preserved.

---

## File Structure

### Backend: create

- `src/hcmai/api/contracts/kis.py` — revisioned KIS HTTP DTOs (`KISInput`, `KISIntent`, request/response).
- `src/hcmai/orchestration/workflows/kis_intent.py` — deterministic ordered-input → structured KIS intent builder.
- `src/hcmai/api/routers/kis.py` — `POST /api/v1/kis/search` HTTP boundary only.
- `src/hcmai/api/dres_logging.py` — shared DRES result-log side effect used by stateless and revisioned KIS routers.
- `tests/api/test_kis_contracts.py` — request/revision/source validation.
- `tests/orchestration/workflows/test_kis_intent.py` — deterministic intent construction.
- `tests/orchestration/workflows/test_kis_pipeline.py` — precomputed-event KIS reuse.
- `tests/api/test_kis_router.py` — endpoint delegation, conflict, and DRES query-text behavior.

### Backend: modify

- `src/hcmai/api/contracts/__init__.py:1-80` — export KIS session contracts.
- `src/hcmai/orchestration/workflows/kis.py:21-121` — split query planning from shared event execution while preserving empty query short-circuit.
- `src/hcmai/orchestration/pipeline.py:74-150,310-315` — instantiate the intent builder and expose `search_kis_revision`.
- `src/hcmai/api/routers/search.py:50-65,125-140,200-270` — consume shared DRES logging helper without changing `/api/v1/search` behavior.
- `src/hcmai/api/routers/__init__.py:1-30` — export `create_kis_router`.
- `src/hcmai/app.py:20-38,204-215` — register the new router.

### Frontend: create

- `frontend/src/api/kis.js` — revisioned KIS client and response validation.
- `frontend/src/api/kis.test.js` — request body/response-contract tests.
- `frontend/src/features/kis/kisSession.js` — pure live-session state transitions.
- `frontend/src/features/kis/kisSession.test.js` — append/commit/failure/reset semantics.
- `frontend/src/features/kis/components/KisPanel.jsx` — text input + compact clue history + current event interpretation.
- `frontend/src/features/kis/components/KisPanel.test.jsx` — rendering and keyboard submission tests.
- `frontend/src/features/kis/index.js` — small feature exports.

### Frontend: modify

- `frontend/src/styles/workspace.css` — style `.kis-panel`, `.kis-clue-history`, `.kis-current-intent`, and `.kis-intent-event`.
- `frontend/src/features/search/components/SearchWorkspace.jsx:54-107,203-228,248-438,548-710` — consume live KIS session state and the revisioned API for text KIS.
- `frontend/src/features/search/components/SearchWorkspace.test.jsx:1-end` — switch text-search mocks/assertions to the KIS API and add multi-revision behavior.
- `frontend/src/features/workspace/queryHistory.js:81-139` — retain KIS revision/input metadata in replay snapshots.
- `frontend/src/features/workspace/queryHistory.test.js` — snapshot metadata regression coverage.

---

### Task 1: Define the revisioned KIS contract and deterministic intent builder

**Files:**
- Create: `src/hcmai/api/contracts/kis.py`
- Modify: `src/hcmai/api/contracts/__init__.py:1-80`
- Create: `src/hcmai/orchestration/workflows/kis_intent.py`
- Create: `tests/api/test_kis_contracts.py`
- Create: `tests/orchestration/workflows/test_kis_intent.py`

**Interfaces:**
- Consumes: `hcmai.temporal.planner.plan_query_events(query: str) -> tuple[str, ...]`; `DEFAULT_MAX_TEMPORAL_EVENT_COUNT`.
- Produces:
  - `KISInput(text: str)`
  - `KISIntent(revision: int, inputs: list[str], query_text: str, events: list[str])`
  - `KISRevisionSearchRequest(inputs: list[KISInput], expected_revision: int, use_dense: bool, use_bm25: bool, top_k: int)`
  - `KISRevisionSearchResponse(revision, inputs, intent, query, events, dense_events, bm25_caption_events, use_dense, use_bm25, results, latency)`
  - `KISIntentBuilder.build(inputs: Sequence[str]) -> KISIntent`

- [x] **Step 1: Write contract tests for first revision, append revision, stale revision, and source validation**

```python
# tests/api/test_kis_contracts.py
import unittest

from pydantic import ValidationError

from hcmai.api.contracts.kis import KISRevisionSearchRequest


class KISRevisionSearchRequestTest(unittest.TestCase):
    def test_first_input_expects_revision_zero(self) -> None:
        request = KISRevisionSearchRequest(
            inputs=[{"text": "A woman is standing in a kitchen."}],
            expected_revision=0,
        )
        self.assertEqual(request.expected_revision, 0)
        self.assertEqual(len(request.inputs), 1)

    def test_second_input_expects_previous_revision_one(self) -> None:
        request = KISRevisionSearchRequest(
            inputs=[
                {"text": "A woman is standing in a kitchen."},
                {"text": "She is talking to a man."},
            ],
            expected_revision=1,
        )
        self.assertEqual(len(request.inputs), 2)

    def test_exposes_previous_revision_for_router_conflict_check(self) -> None:
        request = KISRevisionSearchRequest(
            inputs=[
                {"text": "A woman is standing in a kitchen."},
                {"text": "She is talking to a man."},
            ],
            expected_revision=0,
        )
        self.assertEqual(request.previous_revision, 1)
        self.assertTrue(request.has_revision_conflict)

    def test_requires_at_least_one_retrieval_source(self) -> None:
        with self.assertRaisesRegex(ValidationError, "at least one"):
            KISRevisionSearchRequest(
                inputs=[{"text": "A woman is standing in a kitchen."}],
                expected_revision=0,
                use_dense=False,
                use_bm25=False,
            )


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the contract test and verify it fails because the KIS contract module does not exist**

Run:

```bash
./aic/bin/pytest tests/api/test_kis_contracts.py -v
```

Expected: `ModuleNotFoundError: No module named 'hcmai.api.contracts.kis'` or test collection failure.

- [x] **Step 3: Implement the KIS request/response contracts with explicit revision-conflict metadata**

```python
# src/hcmai/api/contracts/kis.py
from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from hcmai.api.contracts.latency import SearchLatency
from hcmai.api.contracts.search import SearchResult

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class KISInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: NonBlank


class KISIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    inputs: list[NonBlank] = Field(min_length=1)
    query_text: NonBlank
    events: list[NonBlank] = Field(min_length=1)


class KISRevisionSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: list[KISInput] = Field(min_length=1)
    expected_revision: int = Field(ge=0)
    use_dense: bool = True
    use_bm25: bool = True
    top_k: int = Field(default=20, ge=1)

    @property
    def previous_revision(self) -> int:
        return len(self.inputs) - 1

    @property
    def has_revision_conflict(self) -> bool:
        return self.expected_revision != self.previous_revision

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        if not self.use_dense and not self.use_bm25:
            raise ValueError("at least one of use_dense or use_bm25 must be true")
        return self


class KISRevisionSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    inputs: list[KISInput] = Field(min_length=1)
    intent: KISIntent
    query: str
    events: list[str]
    dense_events: list[str] | None = None
    bm25_caption_events: list[str] | None = None
    use_dense: bool
    use_bm25: bool
    results: list[SearchResult] = Field(default_factory=list)
    latency: SearchLatency
```

Also export the four classes from `src/hcmai/api/contracts/__init__.py`; do not move or rename the existing stateless `SearchRequest`/`SearchResponse` classes.

- [x] **Step 4: Run contract tests and verify they pass**

Run:

```bash
./aic/bin/pytest tests/api/test_kis_contracts.py -v
```

Expected: 4 tests pass.

- [x] **Step 5: Write intent-builder tests that lock deterministic phase-1 semantics**

```python
# tests/orchestration/workflows/test_kis_intent.py
import unittest

from hcmai.orchestration.workflows.kis_intent import KISIntentBuilder


class KISIntentBuilderTest(unittest.TestCase):
    def test_builds_revision_from_ordered_inputs(self) -> None:
        intent = KISIntentBuilder(max_temporal_event_count=8).build([
            "  A woman is standing in a kitchen.  ",
            "She is talking to a man.",
        ])
        self.assertEqual(intent.revision, 2)
        self.assertEqual(
            intent.inputs,
            ["A woman is standing in a kitchen.", "She is talking to a man."],
        )
        self.assertEqual(
            intent.query_text,
            "A woman is standing in a kitchen. She is talking to a man.",
        )
        self.assertEqual(
            intent.events,
            ["A woman is standing in a kitchen", "She is talking to a man"],
        )

    def test_clues_without_terminal_punctuation_still_split_into_distinct_events(self) -> None:
        intent = KISIntentBuilder(max_temporal_event_count=8).build([
            "A woman is standing in a kitchen",
            "She is talking to a man",
        ])
        self.assertEqual(intent.revision, 2)
        self.assertEqual(len(intent.events), 2)
        self.assertEqual(intent.events[0], "A woman is standing in a kitchen")
        self.assertEqual(intent.events[1], "She is talking to a man")

    def test_preserves_input_order(self) -> None:
        intent = KISIntentBuilder(max_temporal_event_count=8).build([
            "First event happens.",
            "Then the second event happens.",
            "Finally the third event happens.",
        ])
        self.assertEqual(intent.revision, 3)
        self.assertEqual(intent.events[0], "First event happens")
        self.assertEqual(intent.events[-1], "Finally the third event happens")

    def test_rejects_intent_that_exceeds_temporal_event_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 2 temporal events"):
            KISIntentBuilder(max_temporal_event_count=2).build([
                "One happens. Two happens. Three happens."
            ])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 6: Run the intent-builder test and verify it fails because the builder is missing**

Run:

```bash
./aic/bin/pytest tests/orchestration/workflows/test_kis_intent.py -v
```

Expected: import failure for `KISIntentBuilder`.

- [x] **Step 7: Implement `KISIntentBuilder` as the isolated deterministic baseline**

```python
# src/hcmai/orchestration/workflows/kis_intent.py
from __future__ import annotations

from collections.abc import Sequence

from hcmai.api.contracts.kis import KISIntent
from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.temporal import plan_query_events


class KISIntentBuilder:
    def __init__(
        self,
        max_temporal_event_count: int = DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    ) -> None:
        self.max_temporal_event_count = max_temporal_event_count

    def build(self, inputs: Sequence[str]) -> KISIntent:
        normalized = [" ".join(text.split()) for text in inputs]
        if not normalized or any(not text for text in normalized):
            raise ValueError("KIS intent inputs must contain non-empty text")

        # Join with newlines so plan_query_events recognizes distinct clues
        # even when users do not provide trailing punctuation.
        planned_text = "\n".join(normalized)
        events = list(plan_query_events(planned_text))
        if len(events) > self.max_temporal_event_count:
            raise ValueError(
                f"requests may contain at most {self.max_temporal_event_count} temporal events"
            )

        query_text = " ".join(normalized)
        return KISIntent(
            revision=len(normalized),
            inputs=normalized,
            query_text=query_text,
            events=events,
        )
```

The joined text is deliberately a deterministic baseline. Do not add pronoun resolution, contradiction resolution, or LLM calls in this task.

- [x] **Step 8: Run both Task 1 suites**

Run:

```bash
./aic/bin/pytest tests/api/test_kis_contracts.py tests/orchestration/workflows/test_kis_intent.py -v
```

Expected: all tests pass.

- [x] **Step 9: Commit Task 1**

```bash
git add \
  src/hcmai/api/contracts/kis.py \
  src/hcmai/api/contracts/__init__.py \
  src/hcmai/orchestration/workflows/kis_intent.py \
  tests/api/test_kis_contracts.py \
  tests/orchestration/workflows/test_kis_intent.py
git commit -m "feat: add revisioned KIS intent contracts"
```

---

### Task 2: Reuse the existing KIS temporal pipeline from precomputed events

**Files:**
- Modify: `src/hcmai/orchestration/workflows/kis.py:21-121`
- Modify: `src/hcmai/orchestration/pipeline.py:74-150,310-315`
- Create: `tests/orchestration/workflows/test_kis_pipeline.py`

**Interfaces:**
- Consumes: `KISIntentBuilder.build(...)`, existing `TemporalSearchService.search(...)`, existing `SearchResponse`/`SearchResult`.
- Produces:
  - `KISPipeline.execute_events(*, query: str, events: Sequence[str], retrieval_events: Sequence[str] | None, use_dense: bool, use_bm25: bool, top_k: int, query_ms: float = 0.0) -> SearchResponse`
  - `SearchService.search_kis_revision(request: KISRevisionSearchRequest) -> KISRevisionSearchResponse`
- Preserves: `SearchService.search_kis(request: SearchRequest) -> SearchResponse` unchanged for `/api/v1/search` callers.

- [ ] **Step 1: Write a pipeline regression test proving stateless and precomputed-event paths hit the same temporal service**

```python
# tests/orchestration/workflows/test_kis_pipeline.py
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from hcmai.api.contracts import SearchRequest
from hcmai.orchestration.workflows.kis import KISPipeline


class KISPipelineReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = Mock()
        self.temporal.search.return_value = SimpleNamespace(
            paths=(),
            retrieval_ms=2.0,
            alignment_ms=3.0,
        )
        self.pipeline = KISPipeline(corpus=Mock(), temporal=self.temporal)

    def test_execute_events_uses_precomputed_events_without_replanning(self) -> None:
        response = self.pipeline.execute_events(
            query="A woman enters. Then she sits.",
            events=("A woman enters", "Then she sits"),
            retrieval_events=None,
            use_dense=True,
            use_bm25=True,
            top_k=20,
            query_ms=1.0,
        )
        self.assertEqual(response.events, ["A woman enters", "Then she sits"])
        self.temporal.search.assert_called_once_with(
            ("A woman enters", "Then she sits"),
            retrieval_events=("A woman enters", "Then she sits"),
            caption_events=("A woman enters", "Then she sits"),
            use_dense=True,
            use_bm25=True,
            top_k=20,
        )

    def test_stateless_execute_remains_available(self) -> None:
        response = self.pipeline.execute(SearchRequest(query="A woman enters."))
        self.assertEqual(response.query, "A woman enters.")
        self.assertEqual(response.events, ["A woman enters."])

    def test_stateless_execute_with_empty_query_returns_empty_response(self) -> None:
        response = self.pipeline.execute(SearchRequest(query="   "))
        self.assertEqual(response.events, [])
        self.assertEqual(response.results, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the pipeline test and verify it fails because `execute_events` does not exist**

Run:

```bash
./aic/bin/pytest tests/orchestration/workflows/test_kis_pipeline.py -v
```

Expected: `AttributeError: 'KISPipeline' object has no attribute 'execute_events'`.

- [ ] **Step 3: Refactor `KISPipeline.execute` into planning + shared execution**

Implement the shape below in `src/hcmai/orchestration/workflows/kis.py`:

```python
from collections.abc import Sequence


def execute(self, request: SearchRequest) -> SearchResponse:
    query_started = perf_counter()
    if not request.query.strip():
        query_ms = (perf_counter() - query_started) * 1_000
        return SearchResponse(
            query=request.query,
            events=[],
            dense_events=[] if request.use_dense else None,
            bm25_caption_events=None,
            use_dense=request.use_dense,
            use_bm25=request.use_bm25,
            results=[],
            latency=SearchLatency(
                query_ms=query_ms,
                retrieval_ms=0,
                alignment_ms=0,
                materialization_ms=0,
                total_ms=query_ms,
            ),
        )

    events = plan_query_events(request.query)
    query_ms = (perf_counter() - query_started) * 1_000
    return self.execute_events(
        query=request.query,
        events=events,
        retrieval_events=request.retrieval_events,
        use_dense=request.use_dense,
        use_bm25=request.use_bm25,
        top_k=request.top_k,
        query_ms=query_ms,
    )


def execute_events(
    self,
    *,
    query: str,
    events: Sequence[str],
    retrieval_events: Sequence[str] | None,
    use_dense: bool,
    use_bm25: bool,
    top_k: int,
    query_ms: float = 0.0,
) -> SearchResponse:
    original_events = tuple(events)
    if len(original_events) > self.max_temporal_event_count:
        raise ValueError(
            f"requests may contain at most {self.max_temporal_event_count} temporal events"
        )
    resolved_retrieval_events = (
        tuple(retrieval_events) if retrieval_events is not None else original_events
    )
    if len(resolved_retrieval_events) != len(original_events):
        raise InvalidQueryInputError("retrieval_events must match the original event count")
    caption_events = original_events if use_bm25 else None

    if self.corpus is None or self.materializer is None:
        raise RuntimeError("canonical frame data is not loaded")
    if self.temporal is None:
        raise RuntimeError("temporal search service is not loaded")

    search = self.temporal.search(
        original_events,
        retrieval_events=resolved_retrieval_events,
        caption_events=caption_events,
        use_dense=use_dense,
        use_bm25=use_bm25,
        top_k=top_k,
    )
    materialization_started = perf_counter()
    results = [self.materializer.build_kis_result(path) for path in search.paths]
    materialization_ms = (perf_counter() - materialization_started) * 1_000
    total_ms = query_ms + search.retrieval_ms + search.alignment_ms + materialization_ms

    return SearchResponse(
        query=query,
        events=list(original_events),
        dense_events=list(resolved_retrieval_events) if use_dense else None,
        bm25_caption_events=list(caption_events) if caption_events is not None else None,
        use_dense=use_dense,
        use_bm25=use_bm25,
        results=results,
        latency=SearchLatency(
            query_ms=query_ms,
            retrieval_ms=search.retrieval_ms,
            alignment_ms=search.alignment_ms,
            materialization_ms=materialization_ms,
            total_ms=total_ms,
        ),
    )
```

`execute` preserves the short-circuit for empty queries and contains query planning plus delegation to `execute_events`; retrieval and materialization live exactly once in `execute_events`.

- [ ] **Step 4: Run the pipeline test and verify it passes**

Run:

```bash
./aic/bin/pytest tests/orchestration/workflows/test_kis_pipeline.py -v
```

Expected: both tests pass and the old `execute(SearchRequest)` path still works.

- [ ] **Step 5: Add `KISIntentBuilder` to `SearchService` and write the revisioned orchestration method**

Modify `SearchService.__init__` and add this method near `search_kis`:

```python
from hcmai.api.contracts.kis import (
    KISRevisionSearchRequest,
    KISRevisionSearchResponse,
)
from hcmai.orchestration.workflows.kis_intent import KISIntentBuilder

# __init__
self.kis_intent = KISIntentBuilder(self.config.max_temporal_event_count)


def search_kis_revision(
    self,
    request: KISRevisionSearchRequest,
) -> KISRevisionSearchResponse:
    self._ensure_search_ready()
    started = perf_counter()
    intent = self.kis_intent.build([item.text for item in request.inputs])
    intent_ms = (perf_counter() - started) * 1_000

    search = self.kis.execute_events(
        query=intent.query_text,
        events=intent.events,
        retrieval_events=None,
        use_dense=request.use_dense,
        use_bm25=request.use_bm25,
        top_k=request.top_k,
        query_ms=intent_ms,
    )
    return KISRevisionSearchResponse(
        revision=intent.revision,
        inputs=request.inputs,
        intent=intent,
        query=search.query,
        events=search.events,
        dense_events=search.dense_events,
        bm25_caption_events=search.bm25_caption_events,
        use_dense=search.use_dense,
        use_bm25=search.use_bm25,
        results=search.results,
        latency=search.latency,
    )
```

Do not modify `search_kis`; stateless callers must continue to hit the same public method and response type.

- [ ] **Step 6: Extend the pipeline test with a concrete `SearchService.search_kis_revision` delegation test**

Append this test to `tests/orchestration/workflows/test_kis_pipeline.py`:

```python
from hcmai.api.contracts import SearchLatency, SearchResponse
from hcmai.api.contracts.kis import KISIntent, KISRevisionSearchRequest
from hcmai.orchestration.pipeline import SearchService


def test_search_service_revision_uses_builder_output_directly(self) -> None:
    service = object.__new__(SearchService)
    service._ensure_search_ready = Mock()
    service.kis_intent = Mock()
    service.kis_intent.build.return_value = KISIntent(
        revision=2,
        inputs=["A woman enters.", "Then she sits."],
        query_text="A woman enters. Then she sits.",
        events=["A woman enters", "Then she sits"],
    )
    service.kis = Mock()
    service.kis.execute_events.return_value = SearchResponse(
        query="A woman enters. Then she sits.",
        events=["A woman enters", "Then she sits"],
        dense_events=["A woman enters", "Then she sits"],
        bm25_caption_events=["A woman enters", "Then she sits"],
        use_dense=True,
        use_bm25=True,
        results=[],
        latency=SearchLatency(
            query_ms=1,
            retrieval_ms=2,
            alignment_ms=3,
            materialization_ms=0,
            total_ms=6,
        ),
    )

    request = KISRevisionSearchRequest(
        inputs=[{"text": "A woman enters."}, {"text": "Then she sits."}],
        expected_revision=1,
    )
    response = service.search_kis_revision(request)

    self.assertEqual(response.revision, 2)
    self.assertEqual(response.intent.events, ["A woman enters", "Then she sits"])
    service.kis.execute_events.assert_called_once()
    call = service.kis.execute_events.call_args.kwargs
    self.assertEqual(call["events"], ["A woman enters", "Then she sits"])
    self.assertEqual(call["query"], "A woman enters. Then she sits.")
```

This test proves `search_kis_revision` does not invoke a second planner or a duplicate retrieval pipeline.

- [ ] **Step 7: Run Task 1 + Task 2 backend tests**

Run:

```bash
./aic/bin/pytest \
  tests/api/test_kis_contracts.py \
  tests/orchestration/workflows/test_kis_intent.py \
  tests/orchestration/workflows/test_kis_pipeline.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add \
  src/hcmai/orchestration/workflows/kis.py \
  src/hcmai/orchestration/pipeline.py \
  tests/orchestration/workflows/test_kis_pipeline.py
git commit -m "refactor: reuse KIS retrieval from resolved intent events"
```

---

### Task 3: Expose `POST /api/v1/kis/search` and preserve DRES logging semantics

**Files:**
- Create: `src/hcmai/api/dres_logging.py`
- Create: `src/hcmai/api/routers/kis.py`
- Modify: `src/hcmai/api/routers/search.py:50-65,125-140,200-270`
- Modify: `src/hcmai/api/routers/__init__.py:1-30`
- Modify: `src/hcmai/app.py:20-38,204-215`
- Create: `tests/api/test_kis_router.py`

**Interfaces:**
- Consumes: `SearchService.search_kis_revision(request)` from Task 2.
- Produces: `POST /api/v1/kis/search -> KISRevisionSearchResponse`.
- Preserves: `POST /api/v1/search` and image/filter routes exactly.
- DRES log value for revisioned KIS: resolved `response.intent.query_text`, not just the latest clue.

- [ ] **Step 1: Write endpoint tests for successful revisioned search and validation conflict**

```python
# tests/api/test_kis_router.py
import unittest
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hcmai.api.contracts import SearchLatency
from hcmai.api.contracts.kis import KISIntent, KISRevisionSearchResponse
from hcmai.api.routers.kis import create_kis_router


class KISRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Mock()
        self.service.search_kis_revision.return_value = KISRevisionSearchResponse(
            revision=2,
            inputs=[
                {"text": "A woman is in a kitchen."},
                {"text": "She talks to a man."},
            ],
            intent=KISIntent(
                revision=2,
                inputs=["A woman is in a kitchen.", "She talks to a man."],
                query_text="A woman is in a kitchen. She talks to a man.",
                events=["A woman is in a kitchen", "She talks to a man"],
            ),
            query="A woman is in a kitchen. She talks to a man.",
            events=["A woman is in a kitchen", "She talks to a man"],
            dense_events=["A woman is in a kitchen", "She talks to a man"],
            bm25_caption_events=["A woman is in a kitchen", "She talks to a man"],
            use_dense=True,
            use_bm25=True,
            results=[],
            latency=SearchLatency(
                query_ms=1,
                retrieval_ms=2,
                alignment_ms=3,
                materialization_ms=0,
                total_ms=6,
            ),
        )
        app = FastAPI()
        app.include_router(create_kis_router({
            "service": self.service,
            "vbs_service": None,
        }))
        self.client = TestClient(app)

    def test_revisioned_kis_search_delegates_to_service(self) -> None:
        response = self.client.post("/api/v1/kis/search", json={
            "inputs": [
                {"text": "A woman is in a kitchen."},
                {"text": "She talks to a man."},
            ],
            "expected_revision": 1,
            "top_k": 20,
            "use_dense": True,
            "use_bm25": True,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["revision"], 2)
        self.assertEqual(response.json()["intent"]["events"][1], "She talks to a man")
        self.service.search_kis_revision.assert_called_once()

    def test_revision_mismatch_returns_conflict(self) -> None:
        response = self.client.post("/api/v1/kis/search", json={
            "inputs": [
                {"text": "A woman is in a kitchen."},
                {"text": "She talks to a man."},
            ],
            "expected_revision": 0,
        })
        self.assertEqual(response.status_code, 409)

    def test_invalid_input_or_excessive_events_returns_422(self) -> None:
        self.service.search_kis_revision.side_effect = ValueError("requests may contain at most 8 temporal events")
        response = self.client.post("/api/v1/kis/search", json={
            "inputs": [
                {"text": "A woman is in a kitchen."},
                {"text": "She talks to a man."},
            ],
            "expected_revision": 1,
        })
        self.assertEqual(response.status_code, 422)
        self.assertIn("temporal events", response.json()["detail"])
```

- [ ] **Step 2: Run the router test and verify it fails because the route is unregistered**

Run:

```bash
./aic/bin/pytest tests/api/test_kis_router.py -v
```

Expected: import/module failure for `hcmai.api.routers.kis` before the router is created.

- [ ] **Step 3: Move the reusable DRES log helper out of `routers/search.py`**

Create `src/hcmai/api/dres_logging.py` with the current behavior copied exactly into a shared function:

```python
from __future__ import annotations

import time
from typing import Any

from fastapi import Response

from hcmai.common.utils.logging import get_logger
from hcmai.vbs.models import ApiClientAnswer, QueryEvent, QueryResultLog, RankedAnswer

logger = get_logger(__name__)


async def record_dres_result_log(
    service_container: dict[str, Any],
    response: Response,
    *,
    user_id: str | None,
    category: str,
    event_value: str,
    results: list[Any],
    rank_offset: int = 0,
) -> None:
    log_status = "skipped"
    try:
        vbs_service = service_container.get("vbs_service")
        if (
            user_id is not None
            and user_id.strip()
            and vbs_service is not None
            and vbs_service.session_status(user_id).get("connected")
        ):
            evaluation_id = await vbs_service.resolve_evaluation(user_id)
            timestamp = int(time.time() * 1000)
            ranked = [
                RankedAnswer(
                    rank=rank_offset + index + 1,
                    answer=ApiClientAnswer(
                        media_item_name=vbs_service.media_item_name(result.video_id),
                        start=result.timestamp_ms,
                        end=result.timestamp_ms,
                    ),
                )
                for index, result in enumerate(results)
            ]
            payload = QueryResultLog(
                timestamp=timestamp,
                sort_type="list",
                result_set_availability="",
                results=ranked,
                events=[QueryEvent(
                    timestamp=timestamp,
                    category=category,
                    event_type="SEARCH",
                    value=event_value,
                )],
            )
            await vbs_service.log_results(user_id, evaluation_id, payload)
            log_status = "sent"
    except Exception as error:
        logger.warning(
            "DRES result logging failed status=failed error_type=%s",
            type(error).__name__,
        )
        log_status = "failed"
    response.headers["X-DRES-Log-Status"] = log_status
```

Delete `_record_dres_result_log` and `_now_ms` from `routers/search.py` after all three existing calls use the shared helper. Then update `src/hcmai/api/routers/search.py` calls from:

```python
await _record_dres_result_log(...)
```

to:

```python
from hcmai.api.dres_logging import record_dres_result_log

await record_dres_result_log(...)
```

Do not change the existing category/value behavior for text, image, or filter routes.

- [ ] **Step 4: Implement the dedicated KIS router with explicit HTTP 409 and 422 mapping**

```python
# src/hcmai/api/routers/kis.py
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Response, status
from fastapi.concurrency import run_in_threadpool
from hcmai.api.contracts.kis import KISRevisionSearchRequest, KISRevisionSearchResponse
from hcmai.api.dres_logging import record_dres_result_log
from hcmai.orchestration.utils.errors import InvalidQueryInputError
from hcmai.orchestration.pipeline import SearchServiceUnavailableError


def create_kis_router(service_container: dict[str, Any]) -> APIRouter:
    router = APIRouter()

    @router.post("/api/v1/kis/search", response_model=KISRevisionSearchResponse)
    async def search_kis_revision(
        request: KISRevisionSearchRequest,
        response: Response,
        user_id: Annotated[str | None, Header(alias="X-VBS-User-ID")] = None,
    ) -> KISRevisionSearchResponse:
        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        if request.has_revision_conflict:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"expected_revision {request.expected_revision} does not match "
                    f"previous revision {request.previous_revision}"
                ),
            )
        try:
            result = await run_in_threadpool(service.search_kis_revision, request)
            await record_dres_result_log(
                service_container,
                response,
                user_id=user_id,
                category="TEXT",
                event_value=result.intent.query_text,
                results=result.results,
            )
            return result
        except (ValueError, InvalidQueryInputError) as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error

    return router
```

The explicit `request.has_revision_conflict` branch is the only 409 path. Malformed bodies, empty inputs, excessive events, and invalid retrieval-source combinations return proper HTTP 422 responses.

- [ ] **Step 5: Export and register `create_kis_router`**

In `src/hcmai/api/routers/__init__.py` add:

```python
from hcmai.api.routers.kis import create_kis_router
```

and include it in `__all__`.

In `src/hcmai/app.py`, import it and register it immediately after `create_search_router`:

```python
app.include_router(create_search_router(service_container))
app.include_router(create_kis_router(service_container))
```

- [ ] **Step 6: Add a focused unit test for the resolved DRES log value**

Patch `hcmai.api.routers.kis.record_dres_result_log` in `tests/api/test_kis_router.py` and assert the route passes the complete resolved intent text:

```python
from unittest.mock import AsyncMock, patch


def test_dres_log_uses_resolved_intent_text(self) -> None:
    with patch(
        "hcmai.api.routers.kis.record_dres_result_log",
        new_callable=AsyncMock,
    ) as record_log:
        response = self.client.post("/api/v1/kis/search", json={
            "inputs": [
                {"text": "A woman is in a kitchen."},
                {"text": "She talks to a man."},
            ],
            "expected_revision": 1,
        })
    self.assertEqual(response.status_code, 200)
    self.assertEqual(
        record_log.call_args.kwargs["event_value"],
        "A woman is in a kitchen. She talks to a man.",
    )
```

This test locks the current-resolved-query logging behavior without requiring a live DRES server.

- [ ] **Step 7: Run all backend tests**

Run:

```bash
./aic/bin/pytest tests/api/test_kis_*.py tests/orchestration/workflows/test_kis_*.py -v
```

Expected: all Task 1–3 tests pass; existing app imports still succeed.

- [ ] **Step 8: Smoke-check the stateless route remains present**

Run:

```bash
./aic/bin/python - <<'PY'
from hcmai.app import create_app
paths = {route.path for route in create_app(search_service=object()).routes}
assert "/api/v1/search" in paths
assert "/api/v1/kis/search" in paths
print("routes ok")
PY
```

Expected: `routes ok`.

- [ ] **Step 9: Commit Task 3**

```bash
git add \
  src/hcmai/api/dres_logging.py \
  src/hcmai/api/routers/kis.py \
  src/hcmai/api/routers/search.py \
  src/hcmai/api/routers/__init__.py \
  src/hcmai/app.py \
  tests/api/test_kis_router.py
git commit -m "feat: expose revisioned KIS search API"
```

---

### Task 4: Add the frontend KIS API client and pure session state model

**Files:**
- Create: `frontend/src/api/kis.js`
- Create: `frontend/src/api/kis.test.js`
- Create: `frontend/src/features/kis/kisSession.js`
- Create: `frontend/src/features/kis/kisSession.test.js`
- Create: `frontend/src/features/kis/index.js`

**Interfaces:**
- Consumes: `POST /api/v1/kis/search` from Task 3; `normalizeSearchLatency` from `frontend/src/api/search.js`.
- Produces:
  - `searchKisSession({ inputs, expectedRevision, topK, useDense, useBm25, signal, userId })`
  - `createKisSession()`
  - `withKisDraft(session, draft)`
  - `buildKisRevisionRequest(session)`
  - `commitKisRevision(session, response)`
  - `resetKisSession({ draft = '' } = {})`

- [ ] **Step 1: Write the frontend API test for exact request body and response normalization**

```javascript
// frontend/src/api/kis.test.js
import { searchKisSession } from './kis';

beforeEach(() => {
  global.fetch = jest.fn();
});

test('posts the complete ordered input list and previous revision', async () => {
  global.fetch.mockResolvedValue({
    ok: true,
    status: 200,
    json: jest.fn().mockResolvedValue({
      revision: 2,
      inputs: [{ text: 'woman in kitchen' }, { text: 'talking to a man' }],
      intent: {
        revision: 2,
        inputs: ['woman in kitchen', 'talking to a man'],
        query_text: 'woman in kitchen talking to a man',
        events: ['woman in kitchen talking to a man'],
      },
      query: 'woman in kitchen talking to a man',
      events: ['woman in kitchen talking to a man'],
      dense_events: ['woman in kitchen talking to a man'],
      bm25_caption_events: ['woman in kitchen talking to a man'],
      use_dense: true,
      use_bm25: true,
      results: [],
      latency: {
        query_ms: 1.234,
        retrieval_ms: 2,
        alignment_ms: 3,
        materialization_ms: 0,
        total_ms: 6.234,
      },
    }),
  });

  const response = await searchKisSession({
    inputs: [{ text: 'woman in kitchen' }, { text: 'talking to a man' }],
    expectedRevision: 1,
    topK: 20,
    useDense: true,
    useBm25: true,
    userId: 'team-a',
  });

  expect(JSON.parse(global.fetch.mock.calls[0][1].body)).toEqual({
    inputs: [{ text: 'woman in kitchen' }, { text: 'talking to a man' }],
    expected_revision: 1,
    top_k: 20,
    use_dense: true,
    use_bm25: true,
  });
  expect(global.fetch.mock.calls[0][1].headers['X-VBS-User-ID']).toBe('team-a');
  expect(response.revision).toBe(2);
  expect(response.latency.query_ms).toBe(1.23);
});
```

- [ ] **Step 2: Run the API test and verify it fails because `api/kis.js` is missing**

Run:

```bash
cd frontend && CI=true npm test -- --runInBand src/api/kis.test.js
```

Expected: module-not-found failure.

- [ ] **Step 3: Implement `searchKisSession` with strict response checks**

```javascript
// frontend/src/api/kis.js
import { requestJson } from './client';
import { normalizeSearchLatency } from './search';

export const searchKisSession = async ({
  inputs,
  expectedRevision,
  topK,
  useDense = true,
  useBm25 = true,
  signal,
  userId,
}) => {
  if (!Array.isArray(inputs) || inputs.length === 0) {
    throw new Error('KIS inputs must contain at least one clue');
  }
  const payload = await requestJson('/api/v1/kis/search', {
    method: 'POST',
    body: {
      inputs,
      expected_revision: expectedRevision,
      top_k: topK,
      use_dense: useDense,
      use_bm25: useBm25,
    },
    signal,
    headers: userId?.trim() ? { 'X-VBS-User-ID': userId.trim() } : {},
  });

  if (
    !Number.isSafeInteger(payload?.revision)
    || !Array.isArray(payload?.inputs)
    || !Array.isArray(payload?.intent?.events)
    || !Array.isArray(payload?.results)
    || typeof payload?.latency?.total_ms !== 'number'
  ) {
    throw new Error('KIS server returned an invalid response contract');
  }
  return { ...payload, latency: normalizeSearchLatency(payload.latency) };
};
```

- [ ] **Step 4: Run `kis.test.js` and verify it passes**

Run:

```bash
cd frontend && CI=true npm test -- --runInBand src/api/kis.test.js
```

Expected: pass.

- [ ] **Step 5: Write pure-state tests for draft, append, failed-request safety, and reset**

```javascript
// frontend/src/features/kis/kisSession.test.js
import {
  buildKisRevisionRequest,
  commitKisRevision,
  createKisSession,
  resetKisSession,
  withKisDraft,
} from './kisSession';

test('first draft builds revision-zero append request', () => {
  const session = withKisDraft(createKisSession(), ' woman in kitchen ');
  expect(buildKisRevisionRequest(session)).toEqual({
    inputs: [{ text: 'woman in kitchen' }],
    expectedRevision: 0,
  });
});

test('commit appends only after a successful backend response', () => {
  const session = withKisDraft(createKisSession(), 'woman in kitchen');
  const committed = commitKisRevision(session, {
    revision: 1,
    inputs: [{ text: 'woman in kitchen' }],
    intent: {
      revision: 1,
      inputs: ['woman in kitchen'],
      query_text: 'woman in kitchen',
      events: ['woman in kitchen'],
    },
  });
  expect(committed.revision).toBe(1);
  expect(committed.inputs).toEqual([{ text: 'woman in kitchen' }]);
  expect(committed.draft).toBe('');
});

test('building a request never mutates current committed inputs', () => {
  const session = {
    ...createKisSession(),
    revision: 1,
    inputs: [{ text: 'woman in kitchen' }],
    draft: 'talking to a man',
  };
  buildKisRevisionRequest(session);
  expect(session.inputs).toEqual([{ text: 'woman in kitchen' }]);
  expect(session.revision).toBe(1);
});

test('builds re-search request when draft is empty but committed inputs exist', () => {
  const session = {
    ...createKisSession(),
    revision: 2,
    inputs: [{ text: 'clue 1' }, { text: 'clue 2' }],
    draft: '',
  };
  expect(buildKisRevisionRequest(session)).toEqual({
    inputs: [{ text: 'clue 1' }, { text: 'clue 2' }],
    expectedRevision: 1,
  });
});

test('commit supports idempotent re-search response with same revision', () => {
  const session = {
    ...createKisSession(),
    revision: 1,
    inputs: [{ text: 'clue 1' }],
    draft: '',
  };
  const committed = commitKisRevision(session, {
    revision: 1,
    inputs: [{ text: 'clue 1' }],
    intent: { revision: 1, inputs: ['clue 1'], query_text: 'clue 1', events: ['clue 1'] },
  });
  expect(committed.revision).toBe(1);
});

test('reset clears all live KIS state', () => {
  expect(resetKisSession()).toEqual({
    draft: '',
    inputs: [],
    revision: 0,
    intent: null,
  });
});
```

- [ ] **Step 6: Run state tests and verify they fail because the helper is missing**

Run:

```bash
cd frontend && CI=true npm test -- --runInBand src/features/kis/kisSession.test.js
```

Expected: module-not-found failure.

- [ ] **Step 7: Implement the pure session helper without network/history concerns**

```javascript
// frontend/src/features/kis/kisSession.js
export const createKisSession = ({ draft = '', inputs = [], revision = 0, intent = null } = {}) => ({
  draft,
  inputs,
  revision,
  intent,
});

export const resetKisSession = (options = {}) => createKisSession(options);

export const withKisDraft = (session, draft) => ({ ...session, draft });

export const buildKisRevisionRequest = (session) => {
  const text = session.draft.trim();
  if (text) {
    return {
      inputs: [...session.inputs, { text }],
      expectedRevision: session.revision,
    };
  }
  if (session.inputs.length > 0) {
    return {
      inputs: session.inputs,
      expectedRevision: session.revision - 1,
    };
  }
  return null;
};

export const commitKisRevision = (session, response) => {
  if (response.revision !== session.revision + 1 && response.revision !== session.revision) {
    throw new Error('KIS response revision is invalid for the current session');
  }
  return {
    draft: '',
    inputs: response.inputs.map((item) => ({ text: item.text })),
    revision: response.revision,
    intent: response.intent,
  };
};
```

Export these helpers from `frontend/src/features/kis/index.js`.

- [ ] **Step 8: Run the API + state tests together**

Run:

```bash
cd frontend && CI=true npm test -- --runInBand \
  src/api/kis.test.js \
  src/features/kis/kisSession.test.js
```

Expected: all pass.

- [ ] **Step 9: Commit Task 4**

```bash
git add frontend/src/api/kis.js frontend/src/api/kis.test.js frontend/src/features/kis
git commit -m "feat: add frontend KIS revision session model"
```

---

### Task 5: Build the unified KIS Panel and wire text KIS to revisioned search

**Files:**
- Create: `frontend/src/features/kis/components/KisPanel.jsx`
- Create: `frontend/src/features/kis/components/KisPanel.test.jsx`
- Modify: `frontend/src/features/kis/index.js`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx:54-107,203-228,248-438,548-710`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`

**Interfaces:**
- Consumes: Task 4 session helpers and `searchKisSession`.
- Produces: one visible KIS panel where the same input means “Search” at revision 0 and “add another clue/refinement” after revision 1.
- Preserves: image search, filter controls, result rendering, exploration snapshot, Query History activity state, answer workspace.

- [ ] **Step 1: Write `KisPanel` rendering tests**

```javascript
// frontend/src/features/kis/components/KisPanel.test.jsx
import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import KisPanel from './KisPanel';

test('shows one input for both initial search and later clue refinement', () => {
  const onDraftChange = jest.fn();
  const onSubmit = jest.fn();
  render(
    <KisPanel
      draft=""
      inputs={[{ text: 'A woman is in a kitchen.' }]}
      revision={1}
      intent={{ events: ['A woman is in a kitchen'] }}
      isSearching={false}
      onDraftChange={onDraftChange}
      onSubmit={onSubmit}
    />,
  );

  expect(screen.getByPlaceholderText('Search or add another clue…')).toBeTruthy();
  expect(screen.getByText('A woman is in a kitchen.')).toBeTruthy();
  expect(screen.getByText('E1')).toBeTruthy();
  expect(screen.getByText('A woman is in a kitchen')).toBeTruthy();
});

test('Enter submits and Shift+Enter remains available for multiline text', () => {
  const onSubmit = jest.fn();
  render(
    <KisPanel
      draft="next clue"
      inputs={[]}
      revision={0}
      intent={null}
      isSearching={false}
      onDraftChange={() => {}}
      onSubmit={onSubmit}
    />,
  );
  const input = screen.getByPlaceholderText('Search or add another clue…');
  expect(fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })).toBe(true);
  expect(onSubmit).not.toHaveBeenCalled();
  expect(fireEvent.keyDown(input, { key: 'Enter', shiftKey: false })).toBe(false);
  expect(onSubmit).toHaveBeenCalledTimes(1);
});
```

- [ ] **Step 2: Run the panel tests and verify they fail because `KisPanel` is missing**

Run:

```bash
cd frontend && CI=true npm test -- --runInBand src/features/kis/components/KisPanel.test.jsx
```

Expected: module-not-found failure.

- [ ] **Step 3: Implement the presentation-only `KisPanel`**

```jsx
// frontend/src/features/kis/components/KisPanel.jsx
import React from 'react';

const KisPanel = ({
  draft,
  inputs,
  revision,
  intent,
  isSearching,
  onDraftChange,
  onSubmit,
  inputRef,
  onFocus,
  onBlur,
  isDragOver = false,
}) => (
  <section className="kis-panel" aria-label="KIS search session">
    <textarea
      ref={inputRef}
      id="event-query"
      className={`input-text query-input-field ${isDragOver ? 'drag-over' : ''}`}
      rows={1}
      value={draft}
      onChange={(event) => onDraftChange(event.target.value)}
      placeholder="Search or add another clue…"
      onFocus={onFocus}
      onBlur={onBlur}
      disabled={isSearching}
      onKeyDown={(event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          onSubmit(event);
        }
      }}
    />

    {inputs.length > 0 && (
      <ol className="kis-clue-history" aria-label="KIS clue history">
        {inputs.map((input, index) => (
          <li key={`${index}-${input.text}`}>
            <span>#{index + 1}</span>
            <span>{input.text}</span>
          </li>
        ))}
      </ol>
    )}

    {intent?.events?.length > 0 && (
      <div className="kis-current-intent" aria-label={`Current KIS intent revision ${revision}`}>
        {intent.events.map((eventText, index) => (
          <div className="kis-intent-event" key={`${index}-${eventText}`}>
            <strong>{`E${index + 1}`}</strong>
            <span>{eventText}</span>
          </div>
        ))}
      </div>
    )}
  </section>
);

export default KisPanel;
```

Keep the component stateless; it must not import API/history modules.

- [ ] **Step 3b: Add styling for `KisPanel` and clue badges in `workspace.css`**

Append to `frontend/src/styles/workspace.css`:

```css
/* KIS Revisioned Session & Panel */
.kis-panel {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
}

.kis-clue-history {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.kis-clue-history li {
  display: flex;
  align-items: baseline;
  gap: 8px;
  font-size: 0.85rem;
  color: var(--text-secondary, #94a3b8);
  background: var(--surface-secondary, rgba(255, 255, 255, 0.03));
  padding: 4px 8px;
  border-radius: 4px;
  border-left: 2px solid var(--accent-primary, #3b82f6);
}

.kis-clue-history li span:first-child {
  font-weight: 600;
  font-size: 0.75rem;
  color: var(--accent-primary, #3b82f6);
}

.kis-current-intent {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 4px;
}

.kis-intent-event {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--surface-elevated, rgba(59, 130, 246, 0.1));
  border: 1px solid var(--border-subtle, rgba(59, 130, 246, 0.25));
  border-radius: 12px;
  padding: 2px 8px;
  font-size: 0.8rem;
  color: var(--text-primary, #f1f5f9);
}

.kis-intent-event strong {
  font-size: 0.7rem;
  color: var(--accent-primary, #38bdf8);
}
```

- [ ] **Step 4: Run the panel tests and verify they pass**

Run:

```bash
cd frontend && CI=true npm test -- --runInBand src/features/kis/components/KisPanel.test.jsx
```

Expected: pass.

- [ ] **Step 5: Update `SearchWorkspace.test.jsx` mocks and add the KIS-C two-revision acceptance test before changing production code**

Replace the text KIS mock import:

```javascript
import { searchKisSession } from '../../../api/kis';
import { searchFramesByImage } from '../../../api/search';

jest.mock('../../../api/kis');
jest.mock('../../../api/search', () => ({
  ...jest.requireActual('../../../api/search'),
  searchFramesByImage: jest.fn(),
}));
```

Add:

```javascript
test('keeps committed KIS clues and sends the full list on revision two', async () => {
  searchKisSession
    .mockResolvedValueOnce({
      revision: 1,
      inputs: [{ text: 'A woman is in a kitchen.' }],
      intent: {
        revision: 1,
        inputs: ['A woman is in a kitchen.'],
        query_text: 'A woman is in a kitchen.',
        events: ['A woman is in a kitchen'],
      },
      query: 'A woman is in a kitchen.',
      events: ['A woman is in a kitchen'],
      dense_events: ['A woman is in a kitchen'],
      bm25_caption_events: ['A woman is in a kitchen'],
      use_dense: true,
      use_bm25: true,
      results: [],
      warnings: [],
      latency: SEARCH_LATENCY,
    })
    .mockResolvedValueOnce({
      revision: 2,
      inputs: [
        { text: 'A woman is in a kitchen.' },
        { text: 'She is talking to a man.' },
      ],
      intent: {
        revision: 2,
        inputs: ['A woman is in a kitchen.', 'She is talking to a man.'],
        query_text: 'A woman is in a kitchen. She is talking to a man.',
        events: ['A woman is in a kitchen', 'She is talking to a man'],
      },
      query: 'A woman is in a kitchen. She is talking to a man.',
      events: ['A woman is in a kitchen', 'She is talking to a man'],
      dense_events: ['A woman is in a kitchen', 'She is talking to a man'],
      bm25_caption_events: ['A woman is in a kitchen', 'She is talking to a man'],
      use_dense: true,
      use_bm25: true,
      results: [],
      warnings: [],
      latency: SEARCH_LATENCY,
    });

  renderSearch({ topK: 20, setTopK: jest.fn() });
  submit('A woman is in a kitchen.');
  expect(await screen.findByText('A woman is in a kitchen.')).toBeTruthy();

  submit('She is talking to a man.');
  await waitFor(() => expect(searchKisSession).toHaveBeenLastCalledWith(
    expect.objectContaining({
      inputs: [
        { text: 'A woman is in a kitchen.' },
        { text: 'She is talking to a man.' },
      ],
      expectedRevision: 1,
    }),
  ));
  expect(screen.getByText('She is talking to a man.')).toBeTruthy();
  expect(screen.getByText('E2')).toBeTruthy();
});
```

Also add a failed-second-revision test asserting the first committed clue remains, the second draft remains editable, and revision 1 is not mutated.

- [ ] **Step 6: Run `SearchWorkspace.test.jsx` and verify the new tests fail against current stateless behavior**

Run:

```bash
cd frontend && CI=true npm test -- --runInBand src/features/search/components/SearchWorkspace.test.jsx
```

Expected: failures around `searchKisSession`/clue history while image/filter tests remain structurally valid.

- [ ] **Step 7: Replace only the text-KIS state in `SearchWorkspace` with the dedicated KIS session**

At the top of `SearchWorkspace.jsx`:

```javascript
import { searchKisSession } from '../../../api/kis';
import KisPanel from '../../kis/components/KisPanel';
import {
  buildKisRevisionRequest,
  commitKisRevision,
  createKisSession,
  resetKisSession,
  withKisDraft,
} from '../../kis/kisSession';
```

Delete the now-unused `parseRetrievalDescription` helper and its dedicated unit test; the revisioned request builder replaces that stateless parsing boundary.

Replace:

```javascript
const [eventDescription, setEventDescription] = useState('');
```

with:

```javascript
const [kisSession, setKisSession] = useState(() => createKisSession());
```

Use `kisSession.draft` anywhere the old query textarea value was used for text KIS, including the textarea auto-height `useLayoutEffect` dependency. `onQueryChange` should continue to receive the current draft text only:

```javascript
useEffect(() => {
  onQueryChange?.(kisSession.draft);
}, [kisSession.draft, onQueryChange]);
```

Update `submit`'s `useCallback` dependency array to include `kisSession` in place of `eventDescription`. Do not merge `activeQuerySession` into this object; it remains Query History/view-state only.

- [ ] **Step 8: Change the text submit path to append-on-success semantics**

Replace the stateless text call inside `submit` with:

```javascript
const pending = buildKisRevisionRequest(kisSession);
if (!pending) return;

const response = await searchKisSession({
  inputs: pending.inputs,
  expectedRevision: pending.expectedRevision,
  topK,
  useDense,
  useBm25,
  signal: controller.signal,
  userId: capturedUserId,
});
if (controller.signal.aborted) return;

setKisSession((current) => commitKisRevision(current, response));
```

Then keep the existing result rendering/exploration snapshot logic, but derive `query` and `events` from the revision response:

```javascript
const explorationSnapshot = {
  ...response,
  query: response.intent.query_text,
  events: response.intent.events,
};
```

Do not clear committed clues before the request. On request failure, set `error` but leave `kisSession` unchanged so the draft can be retried.

- [ ] **Step 9: Render `KisPanel` in place of the text textarea while leaving image upload behavior separate**

Inside the existing query input wrapper, keep the image preview branch unchanged. Replace the text `<textarea>` branch with:

```jsx
<KisPanel
  draft={kisSession.draft}
  inputs={kisSession.inputs}
  revision={kisSession.revision}
  intent={kisSession.intent}
  isSearching={isSearching}
  onDraftChange={(draft) => setKisSession((current) => withKisDraft(current, draft))}
  onSubmit={submit}
  inputRef={setQueryTextareaRef}
  onFocus={onFocusQueryInput}
  onBlur={onBlurQueryInput}
  isDragOver={isImageDragOver}
/>
```

Update the submit button condition to allow re-searching committed clues or submitting a new draft:

```javascript
disabled={isSearching || (!kisSession.draft.trim() && kisSession.inputs.length === 0 && !selectedImageFile)}
```

Keep the visible button label `Search`; the input placeholder/history communicates that subsequent submissions are clue refinements.

- [ ] **Step 10: Make `New Search` abort and reset the KIS session, and restore clue visibility on replay**

In `handleNewSearch`, replace `setEventDescription('')` with:

```javascript
setKisSession(resetKisSession());
```

For history replay, hydrate the session with saved clues and revision metadata for inspection:

```javascript
const replaySnapshot = item.result_snapshot;
setKisSession(createKisSession({
  draft: item.query_text || '',
  inputs: replaySnapshot?.kis_inputs || (item.query_text ? [{ text: item.query_text }] : []),
  revision: replaySnapshot?.kis_revision || (item.query_text ? 1 : 0),
  intent: replaySnapshot?.events ? { events: replaySnapshot.events } : null,
}));
```

This shows the replayed query text and restores previous clue badges in `KisPanel`. If the user submits from replay with a new or edited draft, it advances the session smoothly.

- [ ] **Step 11: Run the focused KIS frontend suites**

Run:

```bash
cd frontend && CI=true npm test -- --runInBand \
  src/features/kis/components/KisPanel.test.jsx \
  src/features/kis/kisSession.test.js \
  src/features/search/components/SearchWorkspace.test.jsx
```

Expected: all pass, including existing image/filter/exploration tests.

- [ ] **Step 12: Commit Task 5**

```bash
git add \
  frontend/src/styles/workspace.css \
  frontend/src/features/kis \
  frontend/src/features/search/components/SearchWorkspace.jsx \
  frontend/src/features/search/components/SearchWorkspace.test.jsx
git commit -m "feat: unify textual KIS search and clue refinement"
```

---

### Task 6: Persist KIS revision metadata in Query History and run end-to-end regressions

**Files:**
- Modify: `frontend/src/features/workspace/queryHistory.js:81-139`
- Modify: `frontend/src/features/workspace/queryHistory.test.js`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx:350-430`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx:298-360` and related history tests

**Interfaces:**
- Consumes: successful `KISRevisionSearchResponse` from Task 5.
- Produces: each stored search revision retains `kis_revision`, `kis_inputs`, and the existing resolved `events`; replay remains result-snapshot replay, not live-session rehydration.

- [ ] **Step 1: Add snapshot-helper tests for KIS revision metadata**

```javascript
// add to frontend/src/features/workspace/queryHistory.test.js
import { buildKisSnapshot } from './queryHistory';

test('retains KIS revision and ordered committed inputs in the replay snapshot', () => {
  const snapshot = buildKisSnapshot([], {
    events: ['woman in kitchen', 'talking to man'],
    latency: {
      query_ms: 1,
      retrieval_ms: 2,
      alignment_ms: 3,
      materialization_ms: 0,
      total_ms: 6,
    },
    warnings: [],
    kisRevision: 2,
    kisInputs: [
      { text: 'A woman is in a kitchen.' },
      { text: 'She is talking to a man.' },
    ],
  });

  expect(snapshot.kis_revision).toBe(2);
  expect(snapshot.kis_inputs).toEqual([
    { text: 'A woman is in a kitchen.' },
    { text: 'She is talking to a man.' },
  ]);
  expect(snapshot.events).toEqual(['woman in kitchen', 'talking to man']);
});
```

- [ ] **Step 2: Run the snapshot test and verify it fails because metadata is currently discarded**

Run:

```bash
cd frontend && CI=true npm test -- --runInBand src/features/workspace/queryHistory.test.js
```

Expected: `kis_revision`/`kis_inputs` assertions fail.

- [ ] **Step 3: Extend `normalizeSnapshotOptions` and `buildKisSnapshot`**

Add exact validation rather than copying arbitrary live state:

```javascript
const normalizeKisInputs = (inputs, field) => {
  if (!Array.isArray(inputs)) throw new Error(`${field} must be an array`);
  return inputs.map((input, index) => ({
    text: requireText(input?.text, `${field}[${index}].text`).trim(),
  }));
};
```

Extend `normalizeSnapshotOptions` with backward-compatible optional KIS metadata:

```javascript
const hasKisRevision = options.kisRevision !== undefined;
const hasKisInputs = options.kisInputs !== undefined;
if (hasKisRevision !== hasKisInputs) {
  throw new Error(`${field} must provide kisRevision and kisInputs together`);
}

let kisRevision = null;
let kisInputs = null;
if (hasKisRevision) {
  if (!Number.isSafeInteger(options.kisRevision) || options.kisRevision < 1) {
    throw new Error(`${field}.kisRevision must be a positive integer`);
  }
  kisInputs = normalizeKisInputs(options.kisInputs, `${field}.kisInputs`);
  if (options.kisRevision !== kisInputs.length) {
    throw new Error(`${field}.kisRevision must equal kisInputs.length`);
  }
  kisRevision = options.kisRevision;
}

return {
  events: normalizeEvents(options.events, `${field}.events`),
  latency: normalizeLatency(options.latency, `${field}.latency`),
  warnings: options.warnings === undefined
    ? []
    : normalizeEvents(options.warnings, `${field}.warnings`),
  kisRevision,
  kisInputs,
};
```

Then construct the snapshot base conditionally so old stateless/replay-helper callers keep their existing shape:

```javascript
const snapshot = {
  events,
  latency,
  warnings,
};
if (kisRevision !== null) {
  snapshot.kis_revision = kisRevision;
  snapshot.kis_inputs = kisInputs;
}

return {
  ...snapshot,
  results: results.map((result, index) => {
    const frameId = requireText(result?.frame_id, `results[${index}].frame_id`);
    const frameIds = normalizeFrameIds(
      result.frame_ids || [frameId],
      `results[${index}].frame_ids`,
    );
    const timestampsMs = normalizeNonNegativeIntegers(
      result.timestamps_ms || [result.timestamp_ms],
      `results[${index}].timestamps_ms`,
      frameIds.length,
    );
    const metadata = normalizeMetadata(result.metadata, `results[${index}].metadata`);
    return {
      ...result,
      frame_id: frameId,
      video_id: requireText(result.video_id, `results[${index}].video_id`),
      frame_idx: normalizeNonNegativeIntegers(
        [result.frame_idx],
        `results[${index}].frame_idx`,
        1,
      )[0],
      timestamp_ms: normalizeNonNegativeIntegers(
        [result.timestamp_ms],
        `results[${index}].timestamp_ms`,
        1,
      )[0],
      score: resolveScore(result, `results[${index}].score`),
      frame_ids: frameIds,
      timestamps_ms: timestampsMs,
      caption: normalizeCaption(result, `results[${index}].caption`),
      metadata,
    };
  }),
};
```

Old `buildKisSnapshot` calls that omit both new options keep the existing snapshot shape, and old replay snapshots remain supported because `getSnapshotKind` and `ReplayResults` do not require the metadata.

- [ ] **Step 4: Update the successful-search history write to use resolved intent and committed session metadata**

In `SearchWorkspace.jsx`, build the snapshot once from the successful response:

```javascript
const historySnapshot = buildKisSnapshot(response.results || [], {
  events: response.intent.events,
  latency: response.latency,
  warnings: response.warnings || [],
  kisRevision: response.revision,
  kisInputs: response.inputs,
});
```

Persist:

```javascript
await createQueryHistory({
  queryId,
  userId: historyIdentity,
  queryText: response.intent.query_text,
  resultSnapshot: historySnapshot,
  signal: controller.signal,
});
```

Each successful revision gets its own history record ID as today; this is historical replay, not the live KIS session identifier.

- [ ] **Step 5: Update `SearchWorkspace` history tests to lock revision 1 and revision 2 snapshots**

For revision 2, assert:

```javascript
expect(createQueryHistory).toHaveBeenLastCalledWith(expect.objectContaining({
  queryText: 'A woman is in a kitchen. She is talking to a man.',
  resultSnapshot: expect.objectContaining({
    kis_revision: 2,
    kis_inputs: [
      { text: 'A woman is in a kitchen.' },
      { text: 'She is talking to a man.' },
    ],
    events: ['A woman is in a kitchen', 'She is talking to a man'],
  }),
}));
```

Also retain the existing assertion that history persistence failure does not remove live retrieval results or create a fake `activeQuerySession`.

- [ ] **Step 6: Run the complete frontend suite**

Run:

```bash
cd frontend && CI=true npm test -- --runInBand
```

Expected: all frontend tests pass.

- [ ] **Step 7: Run the production frontend build**

Run:

```bash
cd frontend && npm run build
```

Expected: build succeeds with no missing imports or compile errors.

- [ ] **Step 8: Run the complete backend suite**

Run from repository root:

```bash
./aic/bin/pytest tests/ -v
```

Expected: all backend tests pass.

- [ ] **Step 9: Perform one API-level acceptance smoke test for revision 1 → revision 2**

With the backend running locally (`./aic/bin/python -m uvicorn hcmai.app:app`), execute:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/kis/search \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [{"text": "A woman is standing in a kitchen."}],
    "expected_revision": 0,
    "top_k": 5,
    "use_dense": true,
    "use_bm25": true
  }'
```

Verify the response has `revision: 1`, one `inputs` entry, a non-empty `intent.events`, and `results` in the existing KIS result shape.

Then execute:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/kis/search \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"text": "A woman is standing in a kitchen."},
      {"text": "She is talking to a man."}
    ],
    "expected_revision": 1,
    "top_k": 5,
    "use_dense": true,
    "use_bm25": true
  }'
```

Verify `revision: 2`, both ordered inputs are echoed, and the resolved intent/event list reflects the combined session.

- [ ] **Step 10: Verify stale revision is HTTP 409 and stateless search still works**

Run:

```bash
curl -i -X POST http://127.0.0.1:8000/api/v1/kis/search \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"text": "first clue"},
      {"text": "second clue"}
    ],
    "expected_revision": 0
  }'
```

Expected: `HTTP/1.1 409 Conflict`.

Then run:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "a red vehicle passes", "top_k": 5}'
```

Expected: the original `SearchResponse` shape is unchanged.

- [ ] **Step 11: Commit Task 6**

```bash
git add \
  frontend/src/features/workspace/queryHistory.js \
  frontend/src/features/workspace/queryHistory.test.js \
  frontend/src/features/search/components/SearchWorkspace.jsx \
  frontend/src/features/search/components/SearchWorkspace.test.jsx
git commit -m "feat: persist revisioned KIS search history"
```

---

## Final Verification Gate

Before declaring this phase complete, verify every acceptance criterion from the design spec:

1. **One-input KIS-T equivalence** — compare `/api/v1/search` and `/api/v1/kis/search` on the same text and confirm they resolve the same `events` and use the same `KISPipeline.execute_events` path.
2. **KIS-C revisions** — submit three clues and confirm the UI shows three committed entries with revisions 1, 2, and 3 across successful requests.
3. **Shared temporal retrieval** — no second KIS scorer/alignment implementation exists; revisioned KIS delegates to `KISPipeline` + `TemporalSearchService`.
4. **New Search reset** — active request is aborted, draft/history/current intent are cleared, next submit sends `expected_revision: 0`.
5. **History separation** — replay does not populate live committed KIS inputs; submitting from replay starts a new revision-1 session.
6. **Stateless compatibility** — existing `/api/v1/search`, image search, filters, exploration snapshots, and result materialization tests remain green.
7. **Failure semantics** — a failed second revision leaves revision 1 committed and preserves the unsent second clue in the draft.
8. **No out-of-scope code** — no EventTrail actions, anchor/reject/verify commands, entity consistency, VLM verification, server session persistence, AVS-specific logic, or VQA reasoning are introduced.

Run the final commands:

```bash
./aic/bin/pytest tests/ -v
cd frontend
CI=true npm test -- --runInBand
npm run build
```

All three commands must succeed before moving to interactive temporal evidence work.
