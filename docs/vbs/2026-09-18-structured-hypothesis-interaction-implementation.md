# Structured Hypothesis Interaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace opaque KIS query rewriting and cell-by-cell EventTrail decline with source-grounded Query Hypothesis editing and event-conditioned Result Hypothesis exploration, while preserving fast-path search/submit behavior.

**Architecture:** Evolve `KISIntent` into the canonical query-hypothesis state by adding source provenance, server-owned language metadata, and explicit revisioned structural mutations behind a bounded query-hypothesis session service. Evolve EventTrail in place into a Result Hypothesis Explorer by adding event-conditioned DP alternatives, bounded temporal modes, preview-only alternative views, and Keep/Use/Reject constraint actions. Search snapshots remain bound to the exact query revision that produced them; chat becomes proposal-only for canonical semantic changes.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic v2, NumPy, React 19, react-scripts/Jest, existing HCMAI retrieval/EventTrail infrastructure.

**Spec:** `docs/superpowers/specs/2026-09-18-structured-hypothesis-interaction-design.md`

## Global Constraints

- Target source is `src_v2.0`; AVS behavior must remain unchanged.
- Transition-aware/camera-motion scoring is out of scope.
- Do not add a new online VLM reranker or learned temporal encoder.
- Initial LLM resolution may choose event count/boundaries but must not author free-form English canonical semantics.
- No punctuation/keyword-based rule splitter is allowed as a semantic fallback.
- `KISIntent.query_text` must remain the canonical original query, not a join of event rewrites.
- Query and result revisions are monotonic; undo creates a new revision rather than decrementing counters.
- Preview is non-mutating on both query and result sides.
- A committed temporal branch must become an explicit constraint followed by re-decoding; never assign a client-selected path directly.
- Result alternatives are focused-event complete-path hypotheses and are bounded; do not store a persistent branch tree.
- Advanced hypothesis interaction remains optional; Search and Submit fast paths stay available.
- The extracted source snapshot has no `.git`. Every commit step below is safe to run in either a git checkout or this snapshot via the provided conditional shell command.

---

## File Structure Map

### Query hypothesis core

- Modify `src/hcmai/kis/models.py` — provenance/origin/language domain fields and validators.
- Modify `src/hcmai/kis/resolution/initial.py` — source-grounded initial segmentation and safe one-event fallback.
- Modify `src/hcmai/kis/resolution/prompts.py` — segmentation-only prompt and difficult few-shot examples.
- Create `src/hcmai/kis/hypothesis/__init__.py` — public query-hypothesis API.
- Create `src/hcmai/kis/hypothesis/grounding.py` — source normalization, sequential fragment alignment, language classification.
- Create `src/hcmai/kis/hypothesis/models.py` — query session/checkpoint/action domain models.
- Create `src/hcmai/kis/hypothesis/mutations.py` — deterministic split/merge/reorder/edit/add/restore functions.
- Create `src/hcmai/kis/hypothesis/store.py` — bounded TTL store with per-session locking.
- Create `src/hcmai/kis/hypothesis/service.py` — open/get/preview/commit/undo lifecycle.
- Create `src/hcmai/api/contracts/query_hypothesis.py` — request/response/action contracts.
- Create `src/hcmai/api/routers/query_hypothesis.py` — HTTP endpoints.
- Modify `src/hcmai/api/routers/__init__.py`, `src/hcmai/app.py` — router registration.

### Retrieval/search integration

- Modify `src/hcmai/retrieval/plan.py` — explicit translated dense view while preserving canonical/BM25 text.
- Modify `src/hcmai/orchestration/setup/__init__.py` — construct shared `EventTranslator`.
- Modify `src/hcmai/orchestration/pipeline.py` — own query-hypothesis service, build revision-bound search from canonical session state, pass translation projection.
- Modify `src/hcmai/api/contracts/kis.py` — bind searches to query-hypothesis session/revision without trusting a client-authored replacement intent.

### Chat authority migration

- Modify `src/hcmai/kis/feedback/models.py` — remove direct restructure/edit execution semantics from feedback action union and add proposal payloads.
- Modify `src/hcmai/kis/feedback/prompts.py` — instruct chat to propose canonical edits, not commit them.
- Modify `src/hcmai/kis/feedback/resolver.py` — validate proposals without canonical mutation authority.
- Modify `src/hcmai/kis/feedback/service.py` — keep clarify/retrieval refinement; return semantic proposals instead of mutating `KISIntent`.
- Modify `src/hcmai/api/contracts/feedback.py` — expose proposal fields to frontend.

### Result hypothesis core

- Modify `src/hcmai/temporal/dp.py` — event-conditioned forward/backward hypothesis decoder.
- Modify `src/hcmai/orchestration/workflows/search/temporal.py` — materialize event-conditioned paths through existing corpus authority.
- Modify `src/hcmai/event_trail/config.py` — bounded alternative/mode configuration.
- Modify `src/hcmai/event_trail/decoding/decoder.py` — build focused-event modes under active constraints.
- Modify `src/hcmai/event_trail/models.py` — alternatives/modes/session cache and Keep/Use/Reject domain actions.
- Modify `src/hcmai/event_trail/actions/transitions.py` — positive anchor and mode-exclusion transitions.
- Modify `src/hcmai/event_trail/actions/handler.py` — dispatch new actions and invalidate alternatives after commits.
- Modify `src/hcmai/event_trail/actions/diff.py` — project alternatives/preview path data.
- Modify `src/hcmai/event_trail/service.py` — focus/alternatives/preview semantics and revision guards.
- Modify `src/hcmai/api/contracts/event_trail.py` — Hypothesis Explorer contracts.
- Modify `src/hcmai/api/routers/event_trail.py` — non-mutating alternative endpoint and new action errors.

### Frontend query hypothesis

- Create `frontend/src/api/queryHypothesis.js` — query-hypothesis transport.
- Create `frontend/src/features/kis/queryHypothesisSession.js` — client session reducer/state helpers.
- Create `frontend/src/features/kis/components/QueryHypothesisEditor.jsx` — event topology editor container.
- Create `frontend/src/features/kis/components/QueryHypothesisPreview.jsx` — split/merge/reorder/edit/add proposal preview.
- Modify `frontend/src/features/kis/components/EventCard.jsx`, `EventList.jsx`, `KisPanel.jsx` — direct structural affordances/provenance display.
- Modify `frontend/src/features/search/components/SearchWorkspace.jsx` — resolve/edit/search lifecycle and stale-result marker.
- Modify `frontend/src/api/kis.js`, `frontend/src/features/kis/session.js` — revision-bound search-only execution from active query hypothesis.

### Frontend Result Hypothesis Explorer

- Modify `frontend/src/api/eventTrail.js` — alternatives and Keep/Use/Reject transport.
- Modify `frontend/src/features/event-trail/hooks/useEventTrail.js` — focused-event alternative state, preview, stale conflicts.
- Create `frontend/src/features/event-trail/components/HypothesisAlternatives.jsx` — bounded alternative strip.
- Create `frontend/src/features/event-trail/components/HypothesisPathPreview.jsx` — complete-path counterfactual diff.
- Modify `frontend/src/features/event-trail/components/EventTrailPanel.jsx`, `EvidenceInspector.jsx`, `EventRail.jsx` — product-facing Hypothesis Explorer migration.
- Modify event-trail component/hook tests.

### Tests

- Create `tests/kis/test_query_grounding.py`
- Create `tests/kis/test_initial_resolution.py`
- Create `tests/kis/test_query_hypothesis_mutations.py`
- Create `tests/kis/test_query_hypothesis_service.py`
- Create `tests/kis/test_retrieval_projection.py`
- Create `tests/kis/test_feedback_proposals.py`
- Create `tests/temporal/test_event_conditioned_dp.py`
- Create `tests/event_trail/test_hypothesis_modes.py`
- Create `tests/event_trail/test_hypothesis_actions.py`
- Create `tests/api/test_query_hypothesis_api.py`
- Create `tests/api/test_hypothesis_explorer_api.py`

---

### Task 1: Add source provenance and source-grounded initial resolution

**Files:**
- Modify: `src/hcmai/kis/models.py:20-170`
- Create: `src/hcmai/kis/hypothesis/__init__.py`
- Create: `src/hcmai/kis/hypothesis/grounding.py`
- Modify: `src/hcmai/kis/resolution/initial.py:1-95`
- Modify: `src/hcmai/kis/resolution/prompts.py:1-55`
- Test: `tests/kis/test_query_grounding.py`
- Test: `tests/kis/test_initial_resolution.py`

**Interfaces:**
- Produces: `SourceProvenance(source_text: str, start_char: int, end_char: int)`.
- Produces: `KISEvent.origin: Literal['source','user_override','user_added']` and `KISEvent.source_provenance`.
- Produces: `KISIntent.language: Literal['vi','en','mixed']`.
- Produces: `align_source_fragments(query: str, fragments: Sequence[str]) -> tuple[SourceProvenance, ...]`.
- Produces: `infer_query_language(query: str) -> Literal['vi','en','mixed']`.
- Keeps: `KISIntentResolver.resolve_initial(text, revision)` as the entry point used by orchestration.

- [ ] **Step 1: Write provenance/grounding tests**

```python
# tests/kis/test_query_grounding.py
from hcmai.kis.hypothesis.grounding import align_source_fragments, infer_query_language


def test_align_source_fragments_is_left_to_right_for_repeated_text():
    query = "cốc rồi cốc rồi bàn"
    spans = align_source_fragments(query, ["cốc", "cốc", "bàn"])
    assert [(s.start_char, s.end_char) for s in spans] == [(0, 3), (8, 11), (16, 19)]


def test_align_source_fragments_rejects_paraphrase():
    try:
        align_source_fragments("người đàn ông vào phòng", ["a man enters a room"])
    except ValueError as exc:
        assert "not grounded" in str(exc)
    else:
        raise AssertionError("paraphrase must not be accepted as source grounding")


def test_infer_query_language_is_server_owned_and_bounded():
    assert infer_query_language("A person walks into a room") == "en"
    assert infer_query_language("Người đàn ông bước vào phòng") == "vi"
    assert infer_query_language("Người đàn ông picks up a cup") in {"vi", "mixed"}
```

- [ ] **Step 2: Run grounding tests and verify failure**

Run:
```bash
cd /mnt/data/src_v2_0_extracted
PYTHONPATH=src python -m pytest tests/kis/test_query_grounding.py -q
```
Expected: FAIL because `hcmai.kis.hypothesis.grounding` does not exist.

- [ ] **Step 3: Add provenance/origin/language domain fields and grounding helpers**

```python
# src/hcmai/kis/models.py
class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_text: NonBlank
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("source provenance end_char must exceed start_char")
        return self


class KISEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: EventId
    text: NonBlank | None = None
    source_provenance: SourceProvenance | None = None
    origin: Literal["source", "user_override", "user_added"] = "source"
    images: list[KISImageRef] = Field(default_factory=list)
    bindings: list[KISEntityBinding] = Field(default_factory=list)


class KISIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    query_text: NonBlank | None
    language: Literal["vi", "en", "mixed"] = "en"
    entities: list[KISEntity] = Field(default_factory=list)
    events: list[KISEvent] = Field(min_length=1)
    temporal_edges: list[KISTemporalEdge] = Field(default_factory=list)
```

```python
# src/hcmai/kis/hypothesis/grounding.py
from __future__ import annotations
from collections.abc import Sequence
import unicodedata
from hcmai.kis.models import SourceProvenance

_VI_MARKERS = set("ăâđêôơưĂÂĐÊÔƠƯ")


def infer_query_language(query: str) -> str:
    normalized = unicodedata.normalize("NFC", query)
    has_vi = any(ch in _VI_MARKERS or ("\u0300" <= ch <= "\u036f") for ch in unicodedata.normalize("NFD", normalized))
    has_ascii_word = any(part.isascii() and part.isalpha() for part in normalized.split())
    if has_vi and has_ascii_word:
        return "mixed"
    if has_vi:
        return "vi"
    return "en"


def align_source_fragments(query: str, fragments: Sequence[str]) -> tuple[SourceProvenance, ...]:
    cursor = 0
    spans: list[SourceProvenance] = []
    for raw in fragments:
        fragment = " ".join(raw.split())
        start = query.find(fragment, cursor)
        if start < 0:
            raise ValueError(f"fragment is not grounded in source query: {fragment!r}")
        end = start + len(fragment)
        spans.append(SourceProvenance(source_text=fragment, start_char=start, end_char=end))
        cursor = end
    return tuple(spans)
```

- [ ] **Step 4: Change the initial structured contract from English event text to source fragments**

```python
# src/hcmai/kis/models.py
class KISInitialResolutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_text: InitialEventText = Field(
        description="Verbatim source-language fragment for one retrievable chronological moment."
    )
```

```python
# src/hcmai/kis/resolution/prompts.py
KIS_INITIAL_RESOLVER_SYSTEM_PROMPT = """You segment a video-search description into chronologically ordered retrievable moments.

Rules:
- Copy every returned event directly from the user's source text; do not translate or paraphrase.
- Create the smallest chronological units that could reasonably occur at different timestamps.
- Keep attributes that are simultaneous in one visual moment together.
- A continuous camera movement may contain multiple events when it reveals distinct retrievable moments.
- Preserve uncertainty and do not invent details.
- Return only source fragments in chronological order.

Examples:
Input: Người đàn ông bước vào phòng, sau đó lấy chiếc cốc, rồi ngồi xuống bàn.
Events:
1. Người đàn ông bước vào phòng
2. sau đó lấy chiếc cốc
3. rồi ngồi xuống bàn

Input: Cận cảnh một chiếc cốc đỏ trên bàn gỗ.
Events:
1. Cận cảnh một chiếc cốc đỏ trên bàn gỗ.

Input: Máy quay cho thấy hải sản, chuyển sang các nguyên liệu nhiều màu sắc, cuối cùng là toàn cảnh tất cả nguyên liệu.
Events:
1. Máy quay cho thấy hải sản
2. chuyển sang các nguyên liệu nhiều màu sắc
3. cuối cùng là toàn cảnh tất cả nguyên liệu
"""
```

- [ ] **Step 5: Write resolver tests for grounded success and safe fallback**

```python
# tests/kis/test_initial_resolution.py
from hcmai.kis.resolution.initial import KISIntentResolver
from hcmai.kis.models import KISInitialResolution, KISInitialResolutionEvent


class StubLLM:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
    def generate_structured(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.output


def test_initial_resolution_preserves_original_query_and_provenance():
    query = "Người đàn ông bước vào phòng, sau đó lấy chiếc cốc"
    llm = StubLLM(KISInitialResolution(events=[
        KISInitialResolutionEvent(source_text="Người đàn ông bước vào phòng"),
        KISInitialResolutionEvent(source_text="sau đó lấy chiếc cốc"),
    ]))
    intent = KISIntentResolver(llm).resolve_initial(query, revision=1)
    assert intent.query_text == query
    assert [e.text for e in intent.events] == [
        "Người đàn ông bước vào phòng",
        "sau đó lấy chiếc cốc",
    ]
    assert all(e.origin == "source" for e in intent.events)
    assert intent.events[0].source_provenance.start_char == 0


def test_invalid_grounding_falls_back_to_one_untouched_event():
    query = "Người đàn ông bước vào phòng"
    llm = StubLLM(KISInitialResolution(events=[
        KISInitialResolutionEvent(source_text="A man enters a room")
    ]))
    intent = KISIntentResolver(llm).resolve_initial(query, revision=1)
    assert [e.text for e in intent.events] == [query]
    assert intent.events[0].source_provenance.source_text == query
```

- [ ] **Step 6: Implement safe initial resolution**

```python
# src/hcmai/kis/resolution/initial.py
from hcmai.kis.hypothesis.grounding import align_source_fragments, infer_query_language
from hcmai.kis.models import SourceProvenance


def _fallback_intent(query: str, revision: int) -> KISIntent:
    return KISIntent(
        revision=revision,
        query_text=query,
        language=infer_query_language(query),
        entities=[],
        events=[KISEvent(
            id="E1",
            text=query,
            origin="source",
            source_provenance=SourceProvenance(source_text=query, start_char=0, end_char=len(query)),
            bindings=[],
        )],
        temporal_edges=[],
    )

# In _resolve(): use canonical_query = " ".join(normalized); request fragments; align sequentially.
# On structured-output/provider/grounding failure, return _fallback_intent(canonical_query, revision).
```

Catch the repository's bounded inference failures explicitly (`InferenceResponseError`, `InferenceUnavailableError`, `ValueError`) around generation/alignment; do not catch process-level exceptions such as `KeyboardInterrupt` or `SystemExit`.

- [ ] **Step 7: Run tests**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_query_grounding.py tests/kis/test_initial_resolution.py -q
```
Expected: PASS.

- [ ] **Step 8: Commit/checkpoint**

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src/hcmai/kis tests/kis
  git commit -m "feat: ground KIS query hypotheses in source text"
else
  echo "No git checkout; Task 1 checkpoint = passing tests"
fi
```

---

### Task 2: Separate canonical semantics from literal retrieval projection

**Files:**
- Modify: `src/hcmai/retrieval/plan.py:112-154`
- Modify: `src/hcmai/orchestration/pipeline.py:108-145, 637-702`
- Modify: `src/hcmai/orchestration/setup/__init__.py:60-118, 166-216`
- Test: `tests/kis/test_retrieval_projection.py`

**Interfaces:**
- Consumes: `KISIntent.language`, canonical `KISEvent.text`, existing `EventTranslator.translate(events, language)`.
- Produces: `SearchService.event_translator: EventTranslator | None`.
- Produces: `build_retrieval_plan(..., dense_text_by_event: dict[str, str] | None = None)`.
- Invariant: `canonical_text=event.text`, `bm25_text=event.text`, translated text only feeds `dense_text` unless a retrieval override explicitly replaces it.

- [ ] **Step 1: Write projection tests**

```python
# tests/kis/test_retrieval_projection.py
from hcmai.kis.models import KISEvent, KISIntent
from hcmai.retrieval.plan import build_retrieval_plan


def make_intent():
    return KISIntent(
        revision=1,
        query_text="người đàn ông ngồi xuống",
        language="vi",
        entities=[],
        events=[KISEvent(id="E1", text="người đàn ông ngồi xuống")],
        temporal_edges=[],
    )


def test_dense_translation_does_not_replace_canonical_or_bm25_text():
    plan = build_retrieval_plan(
        make_intent(),
        dense_text_by_event={"E1": "a man sits down"},
        use_dense=True,
        use_bm25=True,
    )
    row = plan.events[0]
    assert row.canonical_text == "người đàn ông ngồi xuống"
    assert row.dense_text == "a man sits down"
    assert row.bm25_text == "người đàn ông ngồi xuống"
```

- [ ] **Step 2: Run test and verify failure**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_retrieval_projection.py -q
```
Expected: FAIL because `dense_text_by_event` is not accepted.

- [ ] **Step 3: Extend retrieval-plan builder without changing canonical text**

```python
# src/hcmai/retrieval/plan.py
def build_retrieval_plan(
    intent: Any,
    overrides: dict[str, Any] | None = None,
    *,
    dense_text_by_event: dict[str, str] | None = None,
    use_dense: bool = True,
    use_bm25: bool = True,
) -> KISRetrievalPlan:
    rows: list[KISRetrievalEvent] = []
    for event in intent.events:
        override = (overrides or {}).get(event.id)
        override_dense = (
            getattr(override, "dense_text", None)
            if override is not None and not isinstance(override, dict)
            else (override or {}).get("dense_text") if isinstance(override, dict) else None
        )
        override_bm25 = (
            getattr(override, "bm25_text", None)
            if override is not None and not isinstance(override, dict)
            else (override or {}).get("bm25_text") if isinstance(override, dict) else None
        )

        dense_text: str | None = None
        if use_dense:
            if override_dense is not None:
                dense_text = override_dense
            elif dense_text_by_event is not None and event.id in dense_text_by_event:
                dense_text = dense_text_by_event[event.id]
            else:
                dense_text = event.text

        bm25_text: str | None = None
        if use_bm25:
            bm25_text = override_bm25 if override_bm25 is not None else event.text

        rows.append(
            KISRetrievalEvent(
                event_id=event.id,
                canonical_text=event.text,
                dense_text=dense_text,
                bm25_text=bm25_text,
                image_refs=tuple(event.images),
            )
        )
    return KISRetrievalPlan(events=tuple(rows))
```

- [ ] **Step 4: Wire `EventTranslator` in setup and SearchService**

```python
# src/hcmai/orchestration/setup/__init__.py
from hcmai.retrieval.translation import EventTranslator


def _load_event_translator(settings, llm):
    return EventTranslator(llm, settings.event_translation) if llm is not None else None

# load_search_service():
event_translator = _load_event_translator(settings, llm_client)

# Add this keyword to the existing SearchService(...) constructor call:
event_translator=event_translator,
```

```python
# src/hcmai/orchestration/pipeline.py
# __init__ argument and assignment:
event_translator: EventTranslator | None = None
self.event_translator = event_translator


def _dense_projection(self, intent: KISIntent) -> dict[str, str] | None:
    text_events = [event for event in intent.events if event.text is not None]
    if not text_events or intent.language == "en" or self.event_translator is None:
        return None
    translated = self.event_translator.translate(
        [event.text for event in text_events], intent.language
    )
    return {event.id: text for event, text in zip(text_events, translated, strict=True)}
```

Use `_dense_projection(intent)` in `search_kis()` before `build_retrieval_plan(...)`.

- [ ] **Step 5: Run projection tests and focused resolver tests**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_retrieval_projection.py tests/kis/test_initial_resolution.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit/checkpoint**

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src/hcmai/retrieval/plan.py src/hcmai/orchestration tests/kis/test_retrieval_projection.py
  git commit -m "feat: separate KIS canonical and retrieval text"
else
  echo "No git checkout; Task 2 checkpoint = passing tests"
fi
```

---

### Task 3: Implement deterministic Query Hypothesis mutations and monotonic undo

**Files:**
- Create: `src/hcmai/kis/hypothesis/models.py`
- Create: `src/hcmai/kis/hypothesis/mutations.py`
- Test: `tests/kis/test_query_hypothesis_mutations.py`

**Interfaces:**
- Produces action types `SplitEvent`, `MergeEvents`, `ReorderEvents`, `EditEvent`, `AddEvent`.
- Produces `apply_query_action(intent: KISIntent, action: QueryHypothesisAction) -> KISIntent`.
- Produces `restore_query_state(current: KISIntent, previous: KISIntent) -> KISIntent` with `revision=current.revision+1`.
- Structural operations always re-canonicalize IDs to `E1..En` and rebuild adjacent temporal edges.

- [ ] **Step 1: Write mutation tests**

```python
# tests/kis/test_query_hypothesis_mutations.py
from hcmai.kis.hypothesis.models import SplitEvent, MergeEvents, ReorderEvents, EditEvent, AddEvent
from hcmai.kis.hypothesis.mutations import apply_query_action, restore_query_state


def test_split_recanonicalizes_ids_and_bumps_revision(base_intent):
    out = apply_query_action(base_intent, SplitEvent(event_id="E1", split_at=10, image_assignments={}))
    assert out.revision == base_intent.revision + 1
    assert [e.id for e in out.events] == ["E1", "E2", "E3"]
    assert [(x.source, x.target) for x in out.temporal_edges] == [("E1", "E2"), ("E2", "E3")]


def test_merge_requires_adjacent_events(base_intent):
    try:
        apply_query_action(base_intent, MergeEvents(left_event_id="E1", right_event_id="E3"))
    except ValueError as exc:
        assert "adjacent" in str(exc)
    else:
        raise AssertionError("non-adjacent merge must fail")


def test_edit_preserves_provenance_and_marks_override(base_intent):
    out = apply_query_action(base_intent, EditEvent(event_id="E1", text="edited semantics"))
    assert out.events[0].origin == "user_override"
    assert out.events[0].source_provenance == base_intent.events[0].source_provenance


def test_undo_restore_is_monotonic(base_intent):
    edited = apply_query_action(base_intent, EditEvent(event_id="E1", text="edited"))
    restored = restore_query_state(edited, base_intent)
    assert restored.revision == edited.revision + 1
    assert restored.events[0].text == base_intent.events[0].text
```

Add a local `base_intent` fixture in this test module so the repository does not require a global `conftest.py`.

- [ ] **Step 2: Run tests and verify failure**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_query_hypothesis_mutations.py -q
```
Expected: FAIL because the mutation modules do not exist.

- [ ] **Step 3: Add explicit action models**

```python
# src/hcmai/kis/hypothesis/models.py
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True, slots=True)
class SplitEvent:
    event_id: str
    split_at: int
    image_assignments: dict[str, tuple[Literal["left", "right"], ...]]

@dataclass(frozen=True, slots=True)
class MergeEvents:
    left_event_id: str
    right_event_id: str

@dataclass(frozen=True, slots=True)
class ReorderEvents:
    event_ids: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class EditEvent:
    event_id: str
    text: str

@dataclass(frozen=True, slots=True)
class AddEvent:
    position: int
    text: str | None
    images: tuple = ()

QueryHypothesisAction = SplitEvent | MergeEvents | ReorderEvents | EditEvent | AddEvent
```

- [ ] **Step 4: Implement canonicalization and actions**

```python
# src/hcmai/kis/hypothesis/mutations.py
from hcmai.kis.models import KISEvent, KISIntent, KISTemporalEdge


def _canonicalize(intent: KISIntent, events: list[KISEvent]) -> KISIntent:
    canonical = [event.model_copy(update={"id": f"E{i+1}"}) for i, event in enumerate(events)]
    edges = [KISTemporalEdge(source=f"E{i}", target=f"E{i+1}") for i in range(1, len(canonical))]
    return intent.model_copy(update={
        "revision": intent.revision + 1,
        "events": canonical,
        "temporal_edges": edges,
    })
```

Implement each action as a pure function over `KISIntent`; `query_text` and `language` remain unchanged. Split validates `0 < split_at < len(event.text)`, trims each side, and requires every existing image asset to have an explicit child assignment. Merge unions images by `asset_id`. Reorder requires exactly the existing event-ID set. Add marks the new event `origin='user_added'` and `source_provenance=None`.

- [ ] **Step 5: Run mutation tests**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_query_hypothesis_mutations.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit/checkpoint**

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src/hcmai/kis/hypothesis tests/kis/test_query_hypothesis_mutations.py
  git commit -m "feat: add deterministic query hypothesis mutations"
else
  echo "No git checkout; Task 3 checkpoint = passing tests"
fi
```

---

### Task 4: Add bounded Query Hypothesis sessions, preview/commit/undo API, and revision conflicts

**Files:**
- Create: `src/hcmai/kis/hypothesis/store.py`
- Create: `src/hcmai/kis/hypothesis/service.py`
- Create: `src/hcmai/api/contracts/query_hypothesis.py`
- Create: `src/hcmai/api/routers/query_hypothesis.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/app.py:20-31, 187-197`
- Modify: `src/hcmai/orchestration/pipeline.py:108-233`
- Test: `tests/kis/test_query_hypothesis_service.py`
- Test: `tests/api/test_query_hypothesis_api.py`

**Interfaces:**
- Produces `QueryHypothesisSession(session_id, intent, history, created_at, updated_at)`.
- Produces `QueryHypothesisService.open(text, image_refs)`, `.get(session_id)`, `.preview(session_id, expected_revision, action)`, `.commit(...)`, `.undo(...)`.
- API routes under `/api/v1/kis/hypotheses`.
- Conflict code: `QUERY_REVISION_CONFLICT` -> HTTP 409.
- Preview returns an intent projection but never changes the stored session.

- [ ] **Step 1: Write service conflict/preview/undo tests**

```python
# tests/kis/test_query_hypothesis_service.py
def test_preview_does_not_bump_revision(service, opened_session):
    before = service.get(opened_session.session_id)
    preview = service.preview(
        opened_session.session_id,
        expected_revision=before.intent.revision,
        action=EditEvent(event_id="E1", text="preview text"),
    )
    after = service.get(opened_session.session_id)
    assert preview.intent.revision == before.intent.revision + 1
    assert after.intent == before.intent


def test_commit_rejects_stale_revision(service, opened_session):
    service.commit(opened_session.session_id, 1, EditEvent(event_id="E1", text="first"))
    try:
        service.commit(opened_session.session_id, 1, EditEvent(event_id="E1", text="stale"))
    except QueryHypothesisError as exc:
        assert exc.code == "QUERY_REVISION_CONFLICT"
    else:
        raise AssertionError("stale query commit must fail")


def test_undo_creates_new_monotonic_revision(service, opened_session):
    committed = service.commit(opened_session.session_id, 1, EditEvent(event_id="E1", text="changed"))
    restored = service.undo(opened_session.session_id, committed.intent.revision)
    assert restored.intent.revision == committed.intent.revision + 1
```

Use a local fake resolver returning a deterministic grounded `KISIntent` so these tests do not call an LLM.

- [ ] **Step 2: Run service test and verify failure**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_query_hypothesis_service.py -q
```
Expected: FAIL because service/store do not exist.

- [ ] **Step 3: Define session/view/error types and implement the bounded store**

```python
# src/hcmai/kis/hypothesis/models.py (append to the action models from Task 3)
from dataclasses import dataclass


class QueryHypothesisError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class QueryHypothesisCheckpoint:
    intent: KISIntent


@dataclass(slots=True)
class QueryHypothesisSession:
    session_id: str
    intent: KISIntent
    history: tuple[QueryHypothesisCheckpoint, ...]
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class QueryHypothesisView:
    session_id: str
    intent: KISIntent
    query_revision: int
    can_undo: bool

    @classmethod
    def from_session(cls, session: QueryHypothesisSession) -> "QueryHypothesisView":
        return cls(
            session_id=session.session_id,
            intent=session.intent,
            query_revision=session.intent.revision,
            can_undo=bool(session.history),
        )


@dataclass(frozen=True, slots=True)
class QueryHypothesisPreview:
    base_revision: int
    intent: KISIntent
```

```python
# src/hcmai/kis/hypothesis/store.py
@dataclass(slots=True)
class QueryHypothesisSlot:
    session: QueryHypothesisSession
    lock: threading.RLock
    in_use: int
    inserted_at: float

class QueryHypothesisStore:
    def __init__(self, ttl_seconds: int = 1800, max_entries: int = 100, clock=None) -> None:
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("ttl_seconds and max_entries must be positive")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock if clock is not None else time.monotonic
        self._map_lock = threading.Lock()
        self._entries: OrderedDict[str, QueryHypothesisSlot] = OrderedDict()
        self._tombstones: OrderedDict[str, None] = OrderedDict()

    def put(self, session: QueryHypothesisSession) -> None:
        now = self._clock()
        with self._map_lock:
            self._tombstones.pop(session.session_id, None)
            if session.session_id in self._entries:
                slot = self._entries[session.session_id]
                slot.session = session
                self._entries.move_to_end(session.session_id)
                return
            if len(self._entries) >= self._max_entries:
                evict_key = next((key for key, slot in self._entries.items() if slot.in_use == 0), None)
                if evict_key is None:
                    raise QueryHypothesisError("QUERY_HYPOTHESIS_UNAVAILABLE", "Query hypothesis capacity is busy")
                self._entries.pop(evict_key)
            self._entries[session.session_id] = QueryHypothesisSlot(
                session=session, lock=threading.RLock(), in_use=0, inserted_at=now
            )

    def get(self, session_id: str) -> QueryHypothesisSession:
        with self.locked(session_id) as slot:
            return slot.session

    @contextmanager
    def locked(self, session_id: str) -> Iterator[QueryHypothesisSlot]:
        now = self._clock()
        with self._map_lock:
            slot = self._entries.get(session_id)
            if slot is None:
                code = "QUERY_HYPOTHESIS_EXPIRED" if session_id in self._tombstones else "QUERY_HYPOTHESIS_NOT_FOUND"
                raise QueryHypothesisError(code, f"Query hypothesis {session_id} is unavailable")
            if now >= slot.inserted_at + self._ttl_seconds:
                if slot.in_use == 0:
                    self._entries.pop(session_id, None)
                    self._tombstones[session_id] = None
                raise QueryHypothesisError("QUERY_HYPOTHESIS_EXPIRED", f"Query hypothesis {session_id} has expired")
            self._entries.move_to_end(session_id)
            slot.in_use += 1
        try:
            with slot.lock:
                yield slot
        finally:
            with self._map_lock:
                slot.in_use -= 1

    def remove(self, session_id: str) -> None:
        with self._map_lock:
            self._entries.pop(session_id, None)
            self._tombstones.pop(session_id, None)
```

Use error codes `QUERY_HYPOTHESIS_NOT_FOUND`, `QUERY_HYPOTHESIS_EXPIRED`, and `QUERY_HYPOTHESIS_UNAVAILABLE`.

- [ ] **Step 4: Implement service lifecycle**

```python
# src/hcmai/kis/hypothesis/service.py
class QueryHypothesisService:
    def __init__(self, resolver, store: QueryHypothesisStore, clock=time.time) -> None:
        self._resolver = resolver
        self._store = store
        self._clock = clock

    def _require_revision(self, session: QueryHypothesisSession, expected_revision: int) -> None:
        if session.intent.revision != expected_revision:
            raise QueryHypothesisError(
                "QUERY_REVISION_CONFLICT",
                f"expected query revision {expected_revision}, current is {session.intent.revision}",
            )

    def open(self, text: str, image_refs: tuple[KISImageRef, ...] = ()) -> QueryHypothesisView:
        intent = self._resolver.resolve_initial(text, revision=1)
        if image_refs:
            first = intent.events[0].model_copy(update={"images": list(image_refs)})
            intent = intent.model_copy(update={"events": [first, *intent.events[1:]]})
        now = self._clock()
        session = QueryHypothesisSession(
            session_id=f"qh_{uuid4().hex}",
            intent=intent,
            history=(),
            created_at=now,
            updated_at=now,
        )
        self._store.put(session)
        return QueryHypothesisView.from_session(session)

    def get(self, session_id: str) -> QueryHypothesisView:
        return QueryHypothesisView.from_session(self._store.get(session_id))

    def preview(self, session_id: str, expected_revision: int, action: QueryHypothesisAction) -> QueryHypothesisPreview:
        with self._store.locked(session_id) as slot:
            self._require_revision(slot.session, expected_revision)
            proposed = apply_query_action(slot.session.intent, action)
            return QueryHypothesisPreview(base_revision=expected_revision, intent=proposed)

    def commit(self, session_id: str, expected_revision: int, action: QueryHypothesisAction) -> QueryHypothesisView:
        with self._store.locked(session_id) as slot:
            self._require_revision(slot.session, expected_revision)
            previous = slot.session.intent
            committed = apply_query_action(previous, action)
            slot.session.history = (*slot.session.history[-19:], QueryHypothesisCheckpoint(intent=previous))
            slot.session.intent = committed
            slot.session.updated_at = self._clock()
            return QueryHypothesisView.from_session(slot.session)

    def undo(self, session_id: str, expected_revision: int) -> QueryHypothesisView:
        with self._store.locked(session_id) as slot:
            self._require_revision(slot.session, expected_revision)
            if not slot.session.history:
                raise QueryHypothesisError("QUERY_CANNOT_UNDO", "No query hypothesis checkpoint is available")
            checkpoint = slot.session.history[-1]
            slot.session.history = slot.session.history[:-1]
            slot.session.intent = restore_query_state(slot.session.intent, checkpoint.intent)
            slot.session.updated_at = self._clock()
            return QueryHypothesisView.from_session(slot.session)
```

- [ ] **Step 5: Define HTTP contracts and action discriminators**

```python
# src/hcmai/api/contracts/query_hypothesis.py
class QueryHypothesisOpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: NonBlank
    image_refs: list[KISImageRef] = Field(default_factory=list)

class EditAction(BaseModel):
    type: Literal["edit"] = "edit"
    event_id: EventId
    text: NonBlank

class SplitAction(BaseModel):
    type: Literal["split"] = "split"
    event_id: EventId
    split_at: int = Field(gt=0)
    image_assignments: dict[str, list[Literal["left", "right"]]] = Field(default_factory=dict)

class MergeAction(BaseModel):
    type: Literal["merge"] = "merge"
    left_event_id: EventId
    right_event_id: EventId

class ReorderAction(BaseModel):
    type: Literal["reorder"] = "reorder"
    event_ids: list[EventId] = Field(min_length=1)

class AddAction(BaseModel):
    type: Literal["add"] = "add"
    position: int = Field(ge=0)
    text: NonBlank | None = None
    images: list[KISImageRef] = Field(default_factory=list)

QueryHypothesisActionRequest = Annotated[
    EditAction | SplitAction | MergeAction | ReorderAction | AddAction,
    Field(discriminator="type"),
]

def to_domain_action(action: QueryHypothesisActionRequest) -> QueryHypothesisAction:
    if isinstance(action, EditAction):
        return EditEvent(event_id=action.event_id, text=action.text)
    if isinstance(action, SplitAction):
        assignments = {
            asset_id: tuple(sides)
            for asset_id, sides in action.image_assignments.items()
        }
        return SplitEvent(event_id=action.event_id, split_at=action.split_at, image_assignments=assignments)
    if isinstance(action, MergeAction):
        return MergeEvents(left_event_id=action.left_event_id, right_event_id=action.right_event_id)
    if isinstance(action, ReorderAction):
        return ReorderEvents(event_ids=tuple(action.event_ids))
    return AddEvent(position=action.position, text=action.text, images=tuple(action.images))
```

Response includes `session_id`, `intent`, `query_revision`, `can_undo`, and for preview `base_revision`.

- [ ] **Step 6: Add router and map errors**

Routes:
```text
POST /api/v1/kis/hypotheses/open
GET  /api/v1/kis/hypotheses/{session_id}
POST /api/v1/kis/hypotheses/{session_id}/preview
POST /api/v1/kis/hypotheses/{session_id}/commit
POST /api/v1/kis/hypotheses/{session_id}/undo
```

Map `QUERY_REVISION_CONFLICT` to 409, expiry to 410, not-found to 404, invalid structural operations to 422.

- [ ] **Step 7: Register the service in `SearchService` and router in app**

```python
# SearchService.__init__
self.query_hypothesis_store = QueryHypothesisStore(
    ttl_seconds=self.event_trail_settings.session_ttl_seconds,
    max_entries=self.event_trail_settings.max_sessions,
)
self.query_hypotheses = (
    QueryHypothesisService(self.query_hypothesis_store, self.intent_resolver, self._canonical_image_refs)
    if self.intent_resolver is not None else None
)
```

Register `create_query_hypothesis_router(service_container)` in `create_app()`.

- [ ] **Step 8: Write API conflict test and run**

```python
# tests/api/test_query_hypothesis_api.py
def test_stale_commit_returns_409(client, seeded_query_hypothesis):
    session_id = seeded_query_hypothesis["session_id"]
    payload = {"expected_query_revision": 0, "action": {"type": "edit", "event_id": "E1", "text": "x"}}
    response = client.post(f"/api/v1/kis/hypotheses/{session_id}/commit", json=payload)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "QUERY_REVISION_CONFLICT"
```

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_query_hypothesis_service.py tests/api/test_query_hypothesis_api.py -q
```
Expected: PASS.

- [ ] **Step 9: Commit/checkpoint**

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src/hcmai/kis/hypothesis src/hcmai/api src/hcmai/app.py src/hcmai/orchestration/pipeline.py tests
  git commit -m "feat: add revisioned query hypothesis sessions"
else
  echo "No git checkout; Task 4 checkpoint = passing tests"
fi
```

---

### Task 5: Bind KIS search to the exact Query Hypothesis revision

**Files:**
- Modify: `src/hcmai/api/contracts/kis.py:99-146`
- Modify: `src/hcmai/orchestration/pipeline.py:320-702`
- Modify: `src/hcmai/api/routers/kis.py:31-83`
- Test: `tests/kis/test_query_hypothesis_service.py`
- Test: `tests/api/test_query_hypothesis_api.py`

**Interfaces:**
- KIS search gains `query_hypothesis_session_id: str | None`.
- When that ID is supplied, backend loads canonical intent from the query-hypothesis session and requires `expected_revision == session.intent.revision`.
- Search response gains `query_hypothesis_session_id` and retains `intent.revision` as the query revision used by the snapshot.
- Existing stateless operation route may remain for image-only/compatibility during the migration task, but the new frontend KIS flow must use `search_only` against the server-owned session intent.

- [ ] **Step 1: Add search-binding test**

```python
def test_search_uses_server_hypothesis_not_client_replacement(search_service, query_session):
    request = KISSearchRequest(
        query_hypothesis_session_id=query_session.session_id,
        base_intent=None,
        expected_revision=query_session.intent.revision,
        operation={"kind": "search_only"},
        use_dense=True,
        use_bm25=False,
        top_k=5,
    )
    # Stub KIS pipeline captures the intent passed to execute().
    response = search_service.search_kis(request)
    assert response.intent == query_session.intent
```

- [ ] **Step 2: Run test and verify request contract fails first**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_query_hypothesis_service.py -q
```
Expected: FAIL until the KIS contract accepts the session ID.

- [ ] **Step 3: Extend KIS contracts**

```python
# Add this field to the existing KISSearchRequest without removing its existing fields/validators:
query_hypothesis_session_id: str | None = None

# Keep the existing base_intent, expected_revision, operation, query/image/filter fields unchanged.
# Add this field to the existing KISSearchResponse:
query_hypothesis_session_id: str | None = None
```

Validation: `search_only` may use a server query session with `base_intent=None`; non-session legacy operations keep their existing validation until Task 10 removes direct topology authority from the frontend path.

- [ ] **Step 4: Resolve canonical intent from the session before search**

```python
# SearchService.search_kis
if request.query_hypothesis_session_id:
    if self.query_hypotheses is None:
        raise SearchServiceUnavailableError("Query Hypothesis service is unavailable")
    view = self.query_hypotheses.get(request.query_hypothesis_session_id)
    if view.intent.revision != request.expected_revision:
        raise RevisionConflictError(
            f"Expected revision {request.expected_revision} does not match query hypothesis {view.intent.revision}"
        )
    if request.operation.kind != "search_only":
        raise InvalidQueryInputError("server-owned query hypotheses may only execute search_only through KIS search")
    intent = view.intent
    summary = KISOperationSummary(kind="search_only", affected_event_ids=[])
    intent_ms = 0.0
else:
    intent, summary, intent_ms = self._resolve_operation(request)
```

Return `query_hypothesis_session_id=request.query_hypothesis_session_id` in the response.

- [ ] **Step 5: Verify evidence snapshot still binds the exact intent revision**

Add an assertion in the test after search snapshot creation:
```python
snapshot = search_service.event_trail_snapshots.get(response.evidence_snapshot_id)
assert snapshot.kis_revision == query_session.intent.revision
```

- [ ] **Step 6: Run focused backend tests**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_query_hypothesis_service.py tests/api/test_query_hypothesis_api.py -q
```
Expected: PASS.

- [ ] **Step 7: Commit/checkpoint**

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src/hcmai/api/contracts/kis.py src/hcmai/orchestration/pipeline.py src/hcmai/api/routers/kis.py tests
  git commit -m "feat: bind KIS search to query hypothesis revisions"
else
  echo "No git checkout; Task 5 checkpoint = passing tests"
fi
```

---

### Task 6: De-authorize generic chat and return semantic proposals instead of committed topology changes

**Files:**
- Modify: `src/hcmai/kis/feedback/models.py:16-134`
- Modify: `src/hcmai/kis/feedback/prompts.py`
- Modify: `src/hcmai/kis/feedback/resolver.py:22-108`
- Modify: `src/hcmai/kis/feedback/service.py:150-330`
- Modify: `src/hcmai/api/contracts/feedback.py`
- Test: `tests/kis/test_feedback_proposals.py`

**Interfaces:**
- Removes direct execution of `RestructureAction` and `EditIntentAction` against canonical intent.
- Produces `QueryEditProposalAction` with a query-hypothesis action payload suitable for preview/apply.
- Keeps `RefineRetrievalAction` and `ClarifyAction` as supporting feedback actions.
- Feedback response exposes `query_proposal` and does not bump `KISIntent.revision` merely because a proposal was generated.

- [ ] **Step 1: Write chat-authority tests**

```python
# tests/kis/test_feedback_proposals.py
def test_chat_split_request_returns_proposal_without_mutating_intent(feedback_service, session):
    before = session.state.intent
    response = feedback_service.turn(session.session_id, split_request)
    assert response.intent == before
    assert response.query_proposal is not None
    assert response.query_proposal.action.type == "split"


def test_refine_retrieval_still_changes_only_retrieval_override(feedback_service, session):
    response = feedback_service.turn(session.session_id, refine_request)
    assert response.intent == session.state.intent
    assert "E1" in response.retrieval_overrides
```

- [ ] **Step 2: Run test and verify current chat mutates intent**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_feedback_proposals.py -q
```
Expected: FAIL under current `edit_intent`/`restructure` execution paths.

- [ ] **Step 3: Replace canonical-mutation actions with proposal action**

```python
# src/hcmai/kis/feedback/models.py
class QueryEditProposalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["query_edit_proposal"] = "query_edit_proposal"
    action: dict[str, Any]
    explanation: str

FeedbackAction = Annotated[
    QueryEditProposalAction | RefineRetrievalAction | ClarifyAction | RepairEventAction,
    Field(discriminator="type"),
]
```

Remove `RestructureAction`, `EditIntentAction`, `AnchorAction`, and `RejectCandidateAction` from the generic feedback action union; result-side anchors/rejections belong to Hypothesis Explorer.

- [ ] **Step 4: Update feedback prompt and resolver validation**

Prompt requirement:
```text
When the user asks to split, merge, reorder, add, or semantically edit canonical events,
return query_edit_proposal. Never directly apply topology changes. Retrieval-only wording
may use refine_retrieval. Result-path anchors/rejections are handled by Hypothesis Explorer.
```

Validate proposal event references against `context.intent.events`, but do not execute it.

- [ ] **Step 5: Change FeedbackService proposal branch**

For `query_edit_proposal`, append chat turns and return a response with:
```python
status="proposal"
intent=session.state.intent
query_proposal=action
changed_event_ids=[]
can_undo=session.state.can_undo
```

Do not call `_execute_search()`, do not change intent revision, and do not invalidate EventTrail from the backend merely because a proposal exists.

- [ ] **Step 6: Run chat proposal tests**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_feedback_proposals.py -q
```
Expected: PASS.

- [ ] **Step 7: Commit/checkpoint**

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src/hcmai/kis/feedback src/hcmai/api/contracts/feedback.py tests/kis/test_feedback_proposals.py
  git commit -m "refactor: make KIS chat propose canonical edits"
else
  echo "No git checkout; Task 6 checkpoint = passing tests"
fi
```

---

### Task 7: Implement event-conditioned complete-path decoding in pure DP

**Files:**
- Modify: `src/hcmai/temporal/dp.py:19-197`
- Modify: `src/hcmai/orchestration/workflows/search/temporal.py:35-52, 345-432`
- Test: `tests/temporal/test_event_conditioned_dp.py`

**Interfaces:**
- Produces `ConditionedDPPath(path: DPPath, focus_frame_position: int, focus_timestamp_ms: int, score: float)`.
- Produces `align_video_conditioned(video, focus_event_index, *, allowed, lambda_gap, event_power, cluster_delta, max_paths, min_separation_ms) -> list[ConditionedDPPath]`.
- Produces `TemporalSearchService.decode_event_alternatives(...) -> tuple[EventConditionedPath, ...]`, where each row carries a materialized `AlignedPath` plus the focused frame position/timestamp.
- Must match brute-force conditioned optimum on small deterministic matrices.

- [ ] **Step 1: Write brute-force correctness tests**

```python
# tests/temporal/test_event_conditioned_dp.py
import itertools
import numpy as np
from hcmai.temporal.dp import align_video_conditioned


def brute_force(video, focus_event_index, focus_position, allowed):
    best = None
    n_events, n_frames = video.scores.shape
    for positions in itertools.combinations(range(n_frames), n_events):
        if positions[focus_event_index] != focus_position:
            continue
        if any(not allowed[e, p] for e, p in enumerate(positions)):
            continue
        score = sum(float(video.scores[e, p]) for e, p in enumerate(positions))
        if best is None or score > best[0]:
            best = (score, positions)
    return best


def test_conditioned_paths_match_bruteforce_for_focus_event(video_scores_fixture):
    video = video_scores_fixture
    allowed = np.ones_like(video.scores, dtype=bool)
    paths = align_video_conditioned(
        video, 1, allowed=allowed, lambda_gap=0.0, event_power=1.0,
        cluster_delta=0.0, max_paths=4, min_separation_ms=0,
    )
    for item in paths:
        expected = brute_force(video, 1, item.focus_frame_position, allowed)
        assert expected is not None
        assert abs(item.score - expected[0]) < 1e-9
        assert tuple(item.path.frame_idx) == tuple(int(video.frame_idx[p]) for p in expected[1])
```

Include a second test where the highest unary score at the focus event does **not** produce the highest complete path, proving that alternatives are complete-hypothesis conditioned rather than frame-score ranked.

- [ ] **Step 2: Run DP test and verify failure**

Run:
```bash
PYTHONPATH=src python -m pytest tests/temporal/test_event_conditioned_dp.py -q
```
Expected: FAIL because `align_video_conditioned` does not exist.

- [ ] **Step 3: Implement forward/backward conditioned DP**

Add an internal preprocessing helper shared with `align_video()` so score power, cluster starts, `allowed`, and weighted timestamps use identical semantics.

Core recurrence:
```python
# prefix[e, t] = best score of E0..Ee with Ee fixed at t
# suffix[e, t] = best score of Ee..E_last with Ee fixed at t
conditioned_score[t] = prefix[focus_event_index, t] + suffix[focus_event_index, t] - scores[focus_event_index, t]
```

Store predecessor arrays for prefix and successor arrays for suffix so each selected focus position reconstructs one complete strict-order path. Apply `min_separation_ms` to the **focused event timestamp**, not the final event timestamp.

- [ ] **Step 4: Expose materialized conditioned paths through TemporalSearchService**

```python
@dataclass(frozen=True, slots=True)
class EventConditionedPath:
    path: AlignedPath
    focus_frame_position: int
    focus_timestamp_ms: int
    score: float


def decode_event_alternatives(
    self,
    video: VideoEventScores,
    *,
    allowed: np.ndarray,
    focus_event_index: int,
    max_paths: int,
    min_separation_ms: int,
    decoder_config: DecoderConfigSnapshot | None = None,
) -> tuple[EventConditionedPath, ...]:
    config = self.snapshot_decoder_config() if decoder_config is None else decoder_config
    conditioned = align_video_conditioned(
        video,
        focus_event_index,
        allowed=allowed,
        lambda_gap=config.lambda_gap,
        event_power=config.event_power,
        cluster_delta=config.cluster_delta,
        max_paths=max_paths,
        min_separation_ms=min_separation_ms,
    )
    return tuple(
        EventConditionedPath(
            path=self._materialize_aligned_path(item.path, video),
            focus_frame_position=item.focus_frame_position,
            focus_timestamp_ms=item.focus_timestamp_ms,
            score=item.score,
        )
        for item in conditioned
    )
```

- [ ] **Step 5: Run DP tests**

Run:
```bash
PYTHONPATH=src python -m pytest tests/temporal/test_event_conditioned_dp.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit/checkpoint**

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src/hcmai/temporal/dp.py src/hcmai/orchestration/workflows/search/temporal.py tests/temporal
  git commit -m "feat: decode event-conditioned temporal hypotheses"
else
  echo "No git checkout; Task 7 checkpoint = passing tests"
fi
```

---

### Task 8: Build bounded temporal modes and cache focused-event alternatives in EventTrail sessions

**Files:**
- Modify: `src/hcmai/event_trail/config.py:9-46`
- Modify: `src/hcmai/event_trail/models.py:61-210`
- Modify: `src/hcmai/event_trail/decoding/decoder.py:29-166`
- Modify: `src/hcmai/event_trail/actions/diff.py`
- Test: `tests/event_trail/test_hypothesis_modes.py`

**Interfaces:**
- Produces `TemporalMode(mode_id, event_id, representative_frame_id, representative_timestamp_ms, interval, score, path)`.
- Adds `focused_event_id: str | None` and `alternatives: tuple[TemporalMode, ...]` to `EventTrailSession`.
- Produces `TemporalConstraintDecoder.alternatives(video, constraints, decoder_config, event_idx, event_id, settings) -> tuple[TemporalMode, ...]`.
- Current active occurrence must be represented as one mode when feasible.
- Mode interval radius is `min(mode_max_radius_ms, nearest_competing_peak_distance // 2)`; with no competitor use `mode_max_radius_ms`.

- [ ] **Step 1: Write mode grouping tests**

```python
# tests/event_trail/test_hypothesis_modes.py
def test_mode_regions_are_bounded_by_nearest_competitor_and_max_radius(mode_builder):
    modes = mode_builder(peaks=[(18_000, 1.0), (41_000, 0.9), (70_000, 0.8)], max_radius_ms=8_000)
    assert modes[0].interval == (10_000, 26_000)
    assert modes[1].interval == (33_000, 49_000)
    assert modes[2].interval == (62_000, 78_000)


def test_current_occurrence_is_not_exposed_as_multiple_nearby_modes(decoder_fixture):
    modes = decoder_fixture.alternatives(event_idx=1)
    timestamps = [mode.representative_timestamp_ms for mode in modes]
    assert all(abs(a - b) >= decoder_fixture.settings.mode_min_separation_ms for i, a in enumerate(timestamps) for b in timestamps[i+1:])
```

- [ ] **Step 2: Run test and verify failure**

Run:
```bash
PYTHONPATH=src python -m pytest tests/event_trail/test_hypothesis_modes.py -q
```
Expected: FAIL because temporal modes/alternative decoder do not exist.

- [ ] **Step 3: Add EventTrail alternative configuration**

```python
@dataclass(frozen=True, slots=True)
class EventTrailSettings:
    snapshot_ttl_seconds: int = 900
    session_ttl_seconds: int = 1800
    max_snapshots: int = 64
    max_sessions: int = 128
    alternative_count: int = 4
    mode_min_separation_ms: int = 5_000
    mode_max_radius_ms: int = 8_000
```

Load optional env vars `HCMAI_EVENT_TRAIL_ALTERNATIVE_COUNT`, `HCMAI_EVENT_TRAIL_MODE_MIN_SEPARATION_MS`, and `HCMAI_EVENT_TRAIL_MODE_MAX_RADIUS_MS` through the existing positive-integer parser.

- [ ] **Step 4: Add domain mode/session fields**

```python
@dataclass(frozen=True, slots=True)
class TemporalMode:
    mode_id: str
    event_id: str
    representative_frame_id: str
    representative_frame_idx: int
    representative_timestamp_ms: int
    interval: Interval
    score: float
    path: AlignedPath
    is_current: bool = False

# Add these fields to the existing EventTrailSession dataclass; keep its current
# snapshot, constraints, current_path, last_valid_path, history, and revision fields unchanged.
focused_event_id: str | None = None
alternatives: tuple[TemporalMode, ...] = ()
```

- [ ] **Step 5: Implement mode derivation from conditioned paths**

`TemporalConstraintDecoder.alternatives()` first builds the same admissibility mask as `decode()`, then calls `temporal.decode_event_alternatives(...)`. Assign opaque IDs containing no client-meaningful timestamp, for example `alt_{uuid4().hex}`. Derive non-overlapping bounded intervals from neighboring representative timestamps. Mark the mode containing the active path's focused-event timestamp as `is_current=True`.

- [ ] **Step 6: Project mode/path data in `TrailView`**

Extend `TrailView` with `focused_event_id` and `alternatives`. Reuse `EventCandidate` tuples to represent each alternative's complete path; do not duplicate canonical frame identity logic.

- [ ] **Step 7: Run mode tests**

Run:
```bash
PYTHONPATH=src python -m pytest tests/event_trail/test_hypothesis_modes.py tests/temporal/test_event_conditioned_dp.py -q
```
Expected: PASS.

- [ ] **Step 8: Commit/checkpoint**

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src/hcmai/event_trail src/hcmai/temporal src/hcmai/orchestration tests/event_trail
  git commit -m "feat: expose focused temporal hypothesis modes"
else
  echo "No git checkout; Task 8 checkpoint = passing tests"
fi
```

---

### Task 9: Replace cell-level Approve/Decline with Keep/Use/Reject-mode and non-mutating alternatives API

**Files:**
- Modify: `src/hcmai/event_trail/models.py:92-159`
- Modify: `src/hcmai/event_trail/actions/transitions.py:211-end`
- Modify: `src/hcmai/event_trail/actions/handler.py`
- Modify: `src/hcmai/event_trail/service.py`
- Modify: `src/hcmai/api/contracts/event_trail.py:18-261`
- Modify: `src/hcmai/api/routers/event_trail.py:18-120`
- Test: `tests/event_trail/test_hypothesis_actions.py`
- Test: `tests/api/test_hypothesis_explorer_api.py`

**Interfaces:**
- Replaces product mutation vocabulary with `KeepOccurrence(event_id)`, `UseAlternative(event_id, alternative_id)`, `RejectMode(event_id, mode_id)`.
- Keeps manual `UseFrame`, `ClearAnchor`, window controls, repair, and undo where still needed.
- Adds non-mutating endpoint `GET /api/v1/event-trail/{session_id}/alternatives?event_id=E2&expected_trail_revision=N`.
- `UseAlternative` anchors the alternative's representative frame and re-decodes.
- `RejectMode` appends the mode interval to the focused event's existing `rejected_cells` and re-decodes.
- Any committed mutation clears cached alternatives and focused event before the next alternatives request.

- [ ] **Step 1: Write result mutation tests**

```python
# tests/event_trail/test_hypothesis_actions.py
def test_fetching_alternatives_is_non_mutating(service, opened_session):
    before = service.get(opened_session.session_id)
    modes = service.alternatives(opened_session.session_id, before.trail_revision, "E2")
    after = service.get(opened_session.session_id)
    assert modes
    assert after.trail_revision == before.trail_revision
    assert after.constraints == before.constraints


def test_reject_mode_excludes_entire_mode_interval(service, session_with_modes):
    state, mode = session_with_modes
    updated = service.act(
        state.session_id,
        state.trail_revision,
        RejectMode(event_id="E2", mode_id=mode.mode_id),
    )
    assert updated.trail_revision == state.trail_revision + 1
    assert mode.interval in service.get(state.session_id).constraints.rejected_cells[1]


def test_use_alternative_commits_anchor_not_preview_path(service, session_with_modes):
    state, mode = session_with_modes
    service.act(state.session_id, state.trail_revision, UseAlternative(event_id="E2", alternative_id=mode.mode_id))
    stored = service.get(state.session_id)
    assert stored.constraints.anchors[1] == mode.representative_frame_id
```

- [ ] **Step 2: Run tests and verify failure**

Run:
```bash
PYTHONPATH=src python -m pytest tests/event_trail/test_hypothesis_actions.py -q
```
Expected: FAIL because new actions/service methods do not exist.

- [ ] **Step 3: Add actions and shared alternative lookup**

```python
@dataclass(frozen=True, slots=True)
class KeepOccurrence:
    event_id: str

@dataclass(frozen=True, slots=True)
class UseAlternative:
    event_id: str
    alternative_id: str

@dataclass(frozen=True, slots=True)
class RejectMode:
    event_id: str
    mode_id: str
```

In service/action code, alternative lookup verifies all of: current session, current trail revision, focused event, and ID membership. Failure code is `STALE_ALTERNATIVE` -> HTTP 409.

- [ ] **Step 4: Implement Keep/Use/Reject transitions**

Keep is the existing approve-anchor semantics renamed. UseAlternative sets the corresponding anchor then calls `decoder.decode()`. RejectMode appends `mode.interval` to that event's rejection tuple and calls `decoder.decode()`. Successful mutation pushes a checkpoint, increments revision, updates path/exhaustion, clears alternatives, and logs the explicit constraint.

Do not use `rejection_cell()` for `RejectMode`.

- [ ] **Step 5: Extend HTTP contracts**

```python
class KeepAction(BaseModel):
    type: Literal["keep"] = "keep"
    event_id: str

class UseAlternativeAction(BaseModel):
    type: Literal["use_alternative"] = "use_alternative"
    event_id: str
    alternative_id: str

class RejectModeAction(BaseModel):
    type: Literal["reject_mode"] = "reject_mode"
    event_id: str
    mode_id: str
```

Add `EventTrailAlternative` response rows containing `alternative_id`, representative frame/timestamp, interval, score, `is_current`, and complete `path`.

- [ ] **Step 6: Add non-mutating alternatives route**

```python
@router.get("/{session_id}/alternatives", response_model=EventTrailAlternativesResponse)
async def get_alternatives(session_id: str, event_id: str, expected_trail_revision: int):
    service = _get_event_trail_service(service_container)
    return EventTrailAlternativesResponse.from_domain(
        await run_in_threadpool(service.alternatives, session_id, expected_trail_revision, event_id)
    )
```

- [ ] **Step 7: Add API stale-alternative test**

```python
# tests/api/test_hypothesis_explorer_api.py
def test_old_alternative_is_rejected_after_commit(client, opened_trail):
    modes = client.get(
        f"/api/v1/event-trail/{opened_trail['session_id']}/alternatives",
        params={"event_id": "E2", "expected_trail_revision": opened_trail["trail_revision"]},
    ).json()["alternatives"]
    first = modes[0]
    committed = client.post(
        f"/api/v1/event-trail/{opened_trail['session_id']}/actions",
        json={"expected_trail_revision": opened_trail["trail_revision"], "action": {"type": "keep", "event_id": "E1"}},
    ).json()
    stale = client.post(
        f"/api/v1/event-trail/{opened_trail['session_id']}/actions",
        json={"expected_trail_revision": committed["trail_revision"], "action": {"type": "use_alternative", "event_id": "E2", "alternative_id": first["alternative_id"]}},
    )
    assert stale.status_code == 409
```

- [ ] **Step 8: Run result-side backend tests**

Run:
```bash
PYTHONPATH=src python -m pytest tests/temporal tests/event_trail tests/api/test_hypothesis_explorer_api.py -q
```
Expected: PASS.

- [ ] **Step 9: Commit/checkpoint**

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src/hcmai/event_trail src/hcmai/api/contracts/event_trail.py src/hcmai/api/routers/event_trail.py tests/event_trail tests/api/test_hypothesis_explorer_api.py
  git commit -m "feat: add branch-and-commit hypothesis actions"
else
  echo "No git checkout; Task 9 checkpoint = passing tests"
fi
```

---

### Task 10: Build frontend Query Hypothesis session/editor and make Search an explicit retrieval boundary

**Files:**
- Create: `frontend/src/api/queryHypothesis.js`
- Create: `frontend/src/features/kis/queryHypothesisSession.js`
- Create: `frontend/src/features/kis/queryHypothesisSession.test.js`
- Create: `frontend/src/features/kis/components/QueryHypothesisEditor.jsx`
- Create: `frontend/src/features/kis/components/QueryHypothesisEditor.test.jsx`
- Create: `frontend/src/features/kis/components/QueryHypothesisPreview.jsx`
- Modify: `frontend/src/features/kis/components/EventCard.jsx:7-99`
- Modify: `frontend/src/features/kis/components/EventList.jsx:7-38`
- Modify: `frontend/src/features/kis/components/KisPanel.jsx:11-end`
- Modify: `frontend/src/api/kis.js`
- Modify: `frontend/src/features/kis/session.js`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx:93-123, 655-870`
- Test: existing `frontend/src/features/kis/components/KisPanel.test.jsx`
- Test: existing `frontend/src/features/search/components/SearchWorkspace.test.jsx`

**Interfaces:**
- Client state owns `queryHypothesisSessionId`, current intent/revision, pending preview action, and `resultsQueryRevision`.
- Initial natural query first calls `openQueryHypothesis()`, displays the proposal, then performs explicit search using the returned session/revision.
- Structural editor operations call preview first; Apply calls commit; Cancel only drops local preview.
- Editing does not re-run search automatically.
- Existing results display a stale marker when `resultsQueryRevision !== currentIntent.revision`.

- [ ] **Step 1: Write pure client-state tests**

```javascript
// frontend/src/features/kis/queryHypothesisSession.test.js
import {
  createInitialQueryHypothesisState,
  receiveOpenedHypothesis,
  receivePreview,
  receiveCommit,
  markSearchResults,
} from './queryHypothesisSession';

test('preview does not change canonical revision', () => {
  let state = receiveOpenedHypothesis(createInitialQueryHypothesisState(), {
    session_id: 'qh_1', query_revision: 1, intent: { revision: 1, events: [] },
  });
  state = receivePreview(state, { base_revision: 1, intent: { revision: 2, events: [{ id: 'E1' }] } });
  expect(state.intent.revision).toBe(1);
  expect(state.preview.intent.revision).toBe(2);
});

test('committed edit makes existing results stale without deleting them', () => {
  let state = { ...createInitialQueryHypothesisState(), intent: { revision: 2 }, resultsQueryRevision: 2 };
  state = receiveCommit(state, { query_revision: 3, intent: { revision: 3 } });
  expect(state.resultsQueryRevision).toBe(2);
  expect(state.intent.revision).toBe(3);
});
```

- [ ] **Step 2: Run state tests and verify failure**

Run:
```bash
cd frontend
CI=true npm test -- --runInBand src/features/kis/queryHypothesisSession.test.js
```
Expected: FAIL because module does not exist.

- [ ] **Step 3: Add query-hypothesis API client**

```javascript
// frontend/src/api/queryHypothesis.js
import { requestJson } from './client';

export const openQueryHypothesis = ({ text, imageRefs = [], signal }) => requestJson(
  '/api/v1/kis/hypotheses/open',
  { method: 'POST', body: { text, image_refs: imageRefs }, signal },
);

export const previewQueryHypothesis = (sessionId, expectedQueryRevision, action, { signal } = {}) => requestJson(
  `/api/v1/kis/hypotheses/${encodeURIComponent(sessionId)}/preview`,
  { method: 'POST', body: { expected_query_revision: expectedQueryRevision, action }, signal },
);

export const commitQueryHypothesis = (sessionId, expectedQueryRevision, action, { signal } = {}) => requestJson(
  `/api/v1/kis/hypotheses/${encodeURIComponent(sessionId)}/commit`,
  { method: 'POST', body: { expected_query_revision: expectedQueryRevision, action }, signal },
);
```

Add `undoQueryHypothesis()` and `getQueryHypothesis()` with the same explicit revision/error handling.

- [ ] **Step 4: Implement client hypothesis state helpers**

State shape:
```javascript
{
  sessionId: null,
  intent: null,
  queryRevision: 0,
  preview: null,
  resultsQueryRevision: null,
  isMutating: false,
  error: null,
}
```

`receivePreview` stores only `preview`; `receiveCommit` updates intent/revision and clears preview; `markSearchResults(revision)` records the semantic world that produced the current grid.

- [ ] **Step 5: Build QueryHypothesisEditor and preview component**

`QueryHypothesisEditor` renders the original query, event cards, source/edited badges, and explicit controls for Split, Merge, Reorder, Edit, Add, Undo. It must not force per-event approval.

For Split, selection produces an integer `split_at` within the event text and explicit image assignment rows before preview. `QueryHypothesisPreview` shows current/proposed event sequences and only two canonical buttons: `Cancel` and `Apply`.

- [ ] **Step 6: Update EventCard/EventList/KisPanel to surface provenance without chat-driven editing**

Remove `handleEditEvent` behavior that prefixes the free-form composer with an event ID and colon as the primary edit path. Direct editor actions call structured callbacks instead. Keep the chat composer for conversational assistance.

- [ ] **Step 7: Change SearchWorkspace KIS lifecycle**

For the first natural-language KIS submission:
```javascript
const opened = await openQueryHypothesis({ text: draftText, imageRefs, signal });
setQueryHypothesisState(receiveOpenedHypothesis(opened));
const response = await searchKis({
  queryHypothesisSessionId: opened.session_id,
  baseIntent: null,
  expectedRevision: opened.query_revision,
  operation: { kind: 'search_only' },
  useDense,
  useBm25,
  topK,
  userId,
  signal,
});
```

For later editor commits, keep frames on screen but mark them stale. Search only when user explicitly presses Search. Do not auto-search on split/merge/edit/reorder/add.

- [ ] **Step 8: Extend `searchKis` client payload**

```javascript
body: {
  query_hypothesis_session_id: queryHypothesisSessionId ?? null,
  base_intent: baseIntent ?? null,
  expected_revision: expectedRevision,
  operation,
  use_dense: useDense,
  use_bm25: useBm25,
  top_k: topK,
}
```

- [ ] **Step 9: Add editor/stale-grid component tests**

```javascript
test('apply split commits preview but does not call search automatically', async () => {
  render(React.createElement(QueryHypothesisEditor, props));
  // select split -> preview -> Apply
  await user.click(screen.getByRole('button', { name: /apply/i }));
  expect(props.onCommit).toHaveBeenCalledTimes(1);
  expect(props.onSearch).not.toHaveBeenCalled();
});
```

Update `SearchWorkspace.test.jsx` to assert a stale-results notice appears after a committed query edit and disappears after a new search at that revision.

- [ ] **Step 10: Run frontend query tests**

Run:
```bash
CI=true npm test -- --runInBand \
  src/features/kis/queryHypothesisSession.test.js \
  src/features/kis/components/QueryHypothesisEditor.test.jsx \
  src/features/kis/components/KisPanel.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx
```
Expected: PASS.

- [ ] **Step 11: Commit/checkpoint**

```bash
cd ..
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add frontend/src/api frontend/src/features/kis frontend/src/features/search/components/SearchWorkspace.jsx
  git commit -m "feat: add inspectable query hypothesis editor"
else
  echo "No git checkout; Task 10 checkpoint = passing frontend tests"
fi
```

---

### Task 11: Migrate frontend EventTrail to Hypothesis Explorer with alternatives and Branch-and-Commit

**Files:**
- Modify: `frontend/src/api/eventTrail.js`
- Modify: `frontend/src/features/event-trail/hooks/useEventTrail.js`
- Modify: `frontend/src/features/event-trail/hooks/useEventTrail.test.js`
- Create: `frontend/src/features/event-trail/components/HypothesisAlternatives.jsx`
- Create: `frontend/src/features/event-trail/components/HypothesisAlternatives.test.jsx`
- Create: `frontend/src/features/event-trail/components/HypothesisPathPreview.jsx`
- Modify: `frontend/src/features/event-trail/components/EventTrailPanel.jsx`
- Modify: `frontend/src/features/event-trail/components/EventTrailPanel.test.jsx`
- Modify: `frontend/src/features/event-trail/components/EvidenceInspector.jsx`
- Modify: `frontend/src/features/event-trail/components/EventRail.jsx`

**Interfaces:**
- `useEventTrail` gains `focusEvent(eventId)`, `alternatives`, `previewAlternative`, `clearPreview`, `keep(eventId)`, `useAlternative(eventId, alternativeId)`, `rejectMode(eventId, modeId)`.
- Fetching/focusing alternatives never changes `trail_revision`.
- Any successful mutation clears preview and refetches alternatives only on explicit focus/request, not automatically for every event.
- User-facing title becomes `Hypothesis Explorer`; button vocabulary becomes Keep / Use / Reject occurrence.

- [ ] **Step 1: Write hook test for non-mutating alternatives**

```javascript
test('focus event loads alternatives without changing canonical session revision', async () => {
  openEventTrail.mockResolvedValue(baseSession);
  getEventTrailAlternatives.mockResolvedValue({
    session_id: baseSession.session_id,
    trail_revision: baseSession.trail_revision,
    event_id: 'E2',
    alternatives: [{ alternative_id: 'alt_1', mode_id: 'm1', path: [] }],
  });
  const { result } = renderHook(() => useEventTrail());
  await act(() => result.current.open(context));
  await act(() => result.current.focusEvent('E2'));
  expect(result.current.session.trail_revision).toBe(baseSession.trail_revision);
  expect(result.current.alternatives).toHaveLength(1);
});
```

- [ ] **Step 2: Run hook test and verify failure**

Run:
```bash
cd frontend
CI=true npm test -- --runInBand src/features/event-trail/hooks/useEventTrail.test.js
```
Expected: FAIL because alternatives transport/state does not exist.

- [ ] **Step 3: Add alternatives transport and new mutation helpers**

```javascript
export const getEventTrailAlternatives = async (
  sessionId,
  { eventId, expectedTrailRevision, signal },
) => requestJson(
  `/api/v1/event-trail/${encodeURIComponent(sessionId)}/alternatives?event_id=${encodeURIComponent(eventId)}&expected_trail_revision=${expectedTrailRevision}`,
  { method: 'GET', signal },
);
```

Keep existing `actOnEventTrail()` and send actions `{type:'keep'}`, `{type:'use_alternative'}`, `{type:'reject_mode'}`.

- [ ] **Step 4: Extend hook state and conflict handling**

Add refs/state for `focusedEventId`, `alternatives`, and `previewAlternative`. `focusEvent()` loads modes for the current `trail_revision`. `previewAlternative()` is local-only because the API already returns the complete induced path. A successful action clears alternatives/preview and updates canonical session. `STALE_ALTERNATIVE` and `TRAIL_REVISION_CONFLICT` refresh canonical state and clear preview.

- [ ] **Step 5: Build alternative strip and complete-path preview**

`HypothesisAlternatives` shows current/possible modes as compact cards with representative keyframes/timestamps. Selecting a card updates local preview; it must not fire a mutation call. `HypothesisPathPreview` compares active path and preview path and labels the focused event as selected while other changed events are marked `adjusted to maintain order`.

- [ ] **Step 6: Migrate inspector action vocabulary**

Replace:
```text
Approve -> Keep
Decline -> Reject occurrence
Use current player time -> Use (manual frame)
```

When an alternative is previewed, show `Use this occurrence` and call `useAlternative()` with its opaque ID. Reject sends the selected/current mode ID rather than just the event ID.

- [ ] **Step 7: Rename product-facing EventTrail title and preserve existing infrastructure**

`EventTrailPanel` heading becomes `Hypothesis Explorer`. Existing window controls, undo, submission selection, exhaustion banner, and event rail remain. Do not expose a separate parallel EventTrail entry point.

- [ ] **Step 8: Add component tests**

```javascript
test('previewing an alternative does not call mutation action', async () => {
  render(<HypothesisAlternatives alternatives={alts} onPreview={onPreview} onUse={onUse} />);
  await user.click(screen.getByRole('button', { name: /70\.00s/i }));
  expect(onPreview).toHaveBeenCalled();
  expect(onUse).not.toHaveBeenCalled();
});

test('reject occurrence submits mode id', async () => {
  render(React.createElement(EventTrailPanel, propsWithFocusedMode));
  await user.click(screen.getByRole('button', { name: /reject occurrence/i }));
  expect(propsWithFocusedMode.onRejectMode).toHaveBeenCalledWith('E2', 'mode_current');
});
```

- [ ] **Step 9: Run frontend result-hypothesis tests**

Run:
```bash
CI=true npm test -- --runInBand \
  src/features/event-trail/hooks/useEventTrail.test.js \
  src/features/event-trail/components/HypothesisAlternatives.test.jsx \
  src/features/event-trail/components/EventTrailPanel.test.jsx
```
Expected: PASS.

- [ ] **Step 10: Commit/checkpoint**

```bash
cd ..
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add frontend/src/api/eventTrail.js frontend/src/features/event-trail
  git commit -m "feat: turn EventTrail into hypothesis explorer"
else
  echo "No git checkout; Task 11 checkpoint = passing frontend tests"
fi
```

---

### Task 12: Integrate chat proposals with Query Hypothesis preview, isolate undo scopes, and guard stale semantic worlds

**Files:**
- Modify: `frontend/src/api/feedback.js`
- Modify: `frontend/src/features/kis/feedbackSession.js`
- Modify: `frontend/src/features/kis/components/FeedbackThread.jsx`
- Modify: `frontend/src/features/kis/components/KisPanel.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx:444-640`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `frontend/src/features/kis/components/FeedbackThread.test.jsx`

**Interfaces:**
- Chat proposal carries the same structured action shape accepted by Query Hypothesis preview/commit APIs.
- `Apply proposal` routes through `previewQueryHypothesis()` then the shared `QueryHypothesisPreview` component; chat never updates `currentIntent` directly.
- Query Undo calls query-hypothesis undo; Result Undo stays inside Hypothesis Explorer; feedback undo no longer restores canonical query topology.
- Opening a result records its source query revision. If current query revision changes, explorer is labelled stale relative to the new query but remains inspectable.

- [ ] **Step 1: Add frontend test that chat proposal does not replace intent**

```javascript
test('chat query proposal opens hypothesis preview without replacing current intent', async () => {
  sendFeedbackTurn.mockResolvedValue({
    status: 'proposal',
    intent: { revision: 2, events: currentEvents },
    query_proposal: { action: { type: 'split', event_id: 'E2', split_at: 12, image_assignments: {} } },
  });
  render(React.createElement(SearchWorkspace, props));
  // send feedback message
  expect(screen.getByText(/proposed query change/i)).toBeInTheDocument();
  expect(screen.getByText(/rev 2/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run targeted tests and verify failure**

Run:
```bash
cd frontend
CI=true npm test -- --runInBand \
  src/features/kis/components/FeedbackThread.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx
```
Expected: FAIL until proposal handling is wired.

- [ ] **Step 3: Route feedback proposals into the shared query preview flow**

In `handleFeedbackTurn`, remove the branch that directly writes `turnResp.intent` into `kisSession` for semantic proposals. If `turnResp.query_proposal` exists, call `previewQueryHypothesis(activeQuerySessionId, currentRevision, proposal.action)` and set the common preview state.

`RefineRetrievalAction` responses may still update result sets/overrides because they do not change canonical topology.

- [ ] **Step 4: Separate undo controls**

`KisPanel` exposes `Undo query edit` only when query-hypothesis session history allows it. `Hypothesis Explorer` keeps its existing result undo. Remove the implication that generic feedback undo can rewind both query and result semantic state.

- [ ] **Step 5: Add stale-world UI guard**

When `liveEventTrailContextRef.current.kisRevision !== currentQueryRevision`, show `Based on query revision N` in Hypothesis Explorer context. Do not auto-close the old session; do not allow it to rewrite the new query. A new search replaces `liveEventTrailContextRef` with a snapshot from the current revision.

- [ ] **Step 6: Run targeted frontend tests**

Run:
```bash
CI=true npm test -- --runInBand \
  src/features/kis/components/FeedbackThread.test.jsx \
  src/features/kis/components/KisPanel.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/features/event-trail/components/EventTrailPanel.test.jsx
```
Expected: PASS.

- [ ] **Step 7: Commit/checkpoint**

```bash
cd ..
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add frontend/src/api/feedback.js frontend/src/features/kis frontend/src/features/search/components/SearchWorkspace.jsx frontend/src/features/event-trail
  git commit -m "refactor: unify KIS hypothesis interaction lifecycle"
else
  echo "No git checkout; Task 12 checkpoint = passing frontend tests"
fi
```

---

### Task 13: Add structured hypothesis logging for replayable experiments

**Files:**
- Modify: `src/hcmai/event_trail/logging.py`
- Create: `src/hcmai/kis/hypothesis/logging.py`
- Modify: `src/hcmai/kis/hypothesis/service.py`
- Modify: `src/hcmai/event_trail/service.py`
- Modify: `src/hcmai/event_trail/actions/transitions.py`
- Test: `tests/kis/test_query_hypothesis_service.py`
- Test: `tests/event_trail/test_hypothesis_actions.py`

**Interfaces:**
- Query log records: session ID, query revision, action type, affected event IDs, preview/commit flag, latency.
- Result log records: snapshot/result/video/session IDs, query revision, result revision, focused event, alternative/mode ID, committed constraint, latency.
- No raw model chain-of-thought or hidden reasoning is logged.

- [ ] **Step 1: Add log-shape tests with captured logger**

```python
def test_query_commit_log_contains_revisions(caplog, service, opened_session):
    service.commit(opened_session.session_id, 1, EditEvent(event_id="E1", text="edited"))
    message = next(r.message for r in caplog.records if "query_hypothesis_commit" in r.message)
    assert '"query_revision": 2' in message
    assert '"action_type": "edit"' in message
```

Add the analogous result test asserting `focused_event_id`, `result_revision`, and committed mode/anchor metadata are present.

- [ ] **Step 2: Implement structured logging helpers**

```python
# src/hcmai/kis/hypothesis/logging.py
def log_query_hypothesis_event(*, event_type, session_id, query_revision, action_type, affected_event_ids, latency_ms):
    payload = {
        "event_type": event_type,
        "session_id": session_id,
        "query_revision": query_revision,
        "action_type": action_type,
        "affected_event_ids": list(affected_event_ids),
        "latency_ms": latency_ms,
    }
    logger.info("query_hypothesis_event %s", json.dumps(payload, sort_keys=True))
```

Extend `log_trail_event()` payload call sites rather than creating a second result-side logger.

- [ ] **Step 3: Log preview separately from committed mutation**

Preview log entries contain `committed=false` and must not imply a revision increment. Commit/undo contain `committed=true` and the resulting revision.

- [ ] **Step 4: Run logging-focused tests**

Run:
```bash
PYTHONPATH=src python -m pytest tests/kis/test_query_hypothesis_service.py tests/event_trail/test_hypothesis_actions.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit/checkpoint**

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src/hcmai/kis/hypothesis src/hcmai/event_trail tests
  git commit -m "feat: log structured hypothesis interactions"
else
  echo "No git checkout; Task 13 checkpoint = passing tests"
fi
```

---

### Task 14: Run full regression, remove old product vocabulary/unused canonical-mutation paths, and verify acceptance invariants

**Files:**
- Modify only files identified by failing regression or dead-code search from Tasks 1-13.
- Verify: all backend/frontend tests.

**Interfaces:**
- No `Decline` user-facing control remains in Hypothesis Explorer.
- No generic feedback branch constructs a new `KISIntent` from `new_events` or `replacement_texts`.
- No initial resolver prompt asks for English canonical event sentences.
- AVS imports/routes/tests remain unchanged and passing.

- [ ] **Step 1: Search for forbidden legacy semantics**

Run:
```bash
cd /mnt/data/src_v2_0_extracted
grep -R "one concise self-contained English" -n src/hcmai/kis || true
grep -R "RestructureAction\|EditIntentAction" -n src/hcmai/kis/feedback || true
grep -R ">Decline<\|Decline" -n frontend/src/features/event-trail || true
```
Expected after migration: no active canonical-mutation implementation for `RestructureAction`/`EditIntentAction`; no user-facing `Decline` button/text. Compatibility comments are unnecessary and should be removed rather than retained.

- [ ] **Step 2: Run all backend hypothesis tests**

Run:
```bash
PYTHONPATH=src python -m pytest \
  tests/kis \
  tests/temporal \
  tests/event_trail \
  tests/api/test_query_hypothesis_api.py \
  tests/api/test_hypothesis_explorer_api.py -q
```
Expected: PASS.

- [ ] **Step 3: Run all frontend tests**

Run:
```bash
cd frontend
CI=true npm test -- --runInBand
```
Expected: PASS.

- [ ] **Step 4: Build frontend production bundle**

Run:
```bash
npm run build
```
Expected: build exits 0 with no unresolved imports or compile errors.

- [ ] **Step 5: Run focused manual acceptance flow**

Start the existing backend/frontend stack and verify this exact sequence:

```text
1. Enter a multi-event Vietnamese KIS query.
2. Initial proposal shows source-language event fragments and original query text.
3. Preview a Split; cancel it; confirm revision does not change.
4. Preview Split again; Apply; confirm query revision increases and existing results are marked stale rather than auto-refreshed.
5. Press Search; confirm new results bind to the new revision.
6. Open one multi-event result in Hypothesis Explorer.
7. Focus E2; confirm several temporally separated complete-path alternatives are shown when available.
8. Preview an alternative; confirm path changes are shown without revision change.
9. Use the alternative; confirm E2 is anchored, revision increases, and path re-decodes.
10. Undo; confirm a newer result revision restores the previous constraints.
11. Reject occurrence; confirm the next candidate does not simply move to an adjacent frame in the same rejected mode.
12. Reject all feasible modes if desired; confirm explicit Exhausted state and recover with Undo.
13. Run an AVS query and confirm the AVS Harvest Workspace behaves exactly as before.
```

- [ ] **Step 6: Verify paper-facing metrics can be derived from logs**

From one manual session, confirm logs contain enough fields to compute:
```text
query edit count
query revision sequence
alternative preview count
mode reject count
result revision sequence
repeated-neighborhood reject behavior
LLM call latency from existing resolver telemetry
time-to-target timestamps
```

- [ ] **Step 7: Final commit/checkpoint**

```bash
cd /mnt/data/src_v2_0_extracted
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add src frontend tests docs/superpowers
  git commit -m "feat: complete structured hypothesis interaction"
else
  echo "No git checkout; final checkpoint = backend tests + frontend tests + production build + manual acceptance"
fi
```

---

## Plan Self-Review Results

### Spec coverage

- Source-grounded query hypothesis and provenance: Tasks 1-2.
- Explicit deterministic query topology editing, preview, revision, undo: Tasks 3-5 and 10.
- Chat de-authorization and proposal-only semantic changes: Tasks 6 and 12.
- Search/query revision binding and stale result semantics: Tasks 5, 10, 12.
- Event-conditioned complete-path alternatives: Task 7.
- Temporal modes and bounded rejection regions: Task 8.
- Keep/Use/Reject, preview non-mutation, explicit constraints, exhaustion/revision conflicts: Tasks 9 and 11.
- EventTrail absorption into one Hypothesis Explorer: Task 11.
- Structured research logging: Task 13.
- Optional interaction fast paths and AVS regression safety: Task 14.

### Type/interface consistency

- Query canonical revision field remains `KISIntent.revision`; API aliases it as `query_revision` only in query-hypothesis session views.
- Existing result revision field remains `trail_revision` during implementation for source compatibility; product semantics/UI call it hypothesis revision.
- `alternative_id`/`mode_id` are opaque server IDs; frontend never invents timestamps or path arrays for commit.
- `UseAlternative` commits the alternative representative frame as an anchor and re-decodes; preview path is never assigned directly.
- `KISRetrievalEvent.canonical_text` remains canonical committed event text; translation only feeds dense retrieval projection.

### Placeholder scan

The plan contains no unresolved `TBD`, `TODO`, “implement later”, or unspecified test steps. Every task has concrete files, interfaces, test commands, and an independently testable deliverable.
