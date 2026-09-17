# Language-Free Multilingual KIS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove language metadata and request-time translation from KIS so initial and update operations use canonical multilingual text directly.

**Architecture:** KIS domain models and LLM output contracts become language-free. Canonical event text feeds dense and BM25 retrieval unchanged, while a narrow input validator discards the legacy `language` key from historical `KISIntent` payloads. Generic translation code remains available outside KIS, but KIS composition, readiness, and latency calculation no longer depend on it.

**Tech Stack:** Python 3.12, Pydantic 2, FastAPI, pytest, React/Vitest fixtures

**Spec:** `docs/superpowers/specs/2026-09-17-language-free-multilingual-kis-design.md`

## Global Constraints

- Use the repository virtual environment at `aic/bin/python`.
- Preserve `video_id`, `frame_id`, `frame_idx`, and `timestamp_ms`; this migration does not touch canonical retrieval identity.
- Preserve event authorization, entity binding, image evidence, revision, and temporal-topology validation.
- Keep `SearchLatency.translation_ms` in the public schema and return `0.0` for KIS.
- Accept a legacy input-only `KISIntent.language`, but never serialize or advertise it in new schemas.
- Do not delete the generic `hcmai.retrieval.translation` package or its focused tests.
- The worktree already contains user-owned edits, including overlapping edits in `models.py`, `prompts.py`, `resolver.py`, and the KIS router. Preserve those edits and do not stage or commit implementation files automatically.
- Do not update `KNOWLEDGE.md`: this is a user-approved runtime contract simplification, not a new paper-supported or measured retrieval claim.

## File Structure

- Modify `src/hcmai/kis/models.py`: language-free LLM and canonical intent contracts plus legacy input compatibility.
- Modify `src/hcmai/kis/prompts.py`: stop requesting language metadata.
- Modify `src/hcmai/kis/resolver.py`: canonicalize initial output without language inference.
- Modify `src/hcmai/kis/scoped_resolver.py`: resolve/apply scoped events without language validation.
- Modify `src/hcmai/kis/rewriter.py`: rewrite topology without language comparison.
- Modify `src/hcmai/orchestration/pipeline.py`: build multilingual retrieval views directly and remove translator injection.
- Modify `src/hcmai/orchestration/workflows/kis.py`: remove internal translation timing input while preserving public zero latency.
- Modify `src/hcmai/orchestration/setup.py`: stop constructing and injecting an event translator for KIS.
- Modify `src/hcmai/orchestration/utils/health.py`: remove the obsolete KIS event-translation capability.
- Modify `src/hcmai/api/routers/kis.py`: remove obsolete translation-error mapping.
- Modify focused Python tests under `tests/kis`, `tests/hcmai/kis`, `tests/api`, `tests/orchestration`, `tests/event_trail`, and `tests/test_kis_acceptance_smoke.py`.
- Modify only frontend test fixtures containing obsolete `language` keys; production frontend behavior is already language-agnostic.

---

### Task 1: Make KIS Contracts Language-Free

**Requirements:** REQ-001, REQ-002, REQ-007

**Files:**
- Modify: `tests/kis/test_models.py`
- Modify: `src/hcmai/kis/models.py`

**Interfaces:**
- Consumes: existing `KISEvent`, `KISEntity`, and temporal-edge contracts.
- Produces: `KISResolution` and `KISIntent` with no runtime `language` field; `KISIntent.model_validate()` tolerates a legacy input key.

- [ ] **Step 1: Write failing contract tests**

Replace the language-based assertions with requirements-oriented tests:

```python
def test_REQ_001_kis_contracts_do_not_expose_language() -> None:
    assert "language" not in KISIntent.model_fields
    assert "language" not in KISIntent.model_json_schema()["properties"]
    assert "language" not in KISResolution.model_fields


def test_REQ_002_legacy_language_is_input_only() -> None:
    intent = KISIntent.model_validate({**VALID, "language": "vi"})
    assert "language" not in intent.model_dump()


def test_REQ_007_text_and_query_text_presence_match() -> None:
    with pytest.raises(ValidationError, match="query_text"):
        KISIntent.model_validate({**VALID, "query_text": None})
```

Update `VALID` so it no longer contains `language`, and retain the existing image-only, identity, binding, sequential-ID, and edge tests.

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
aic/bin/python -m pytest tests/kis/test_models.py -q
```

Expected: failures show `language` is still present and the legacy key is still serialized.

- [ ] **Step 3: Implement the minimum contract change**

In `models.py`:

```python
class KISResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_text: NonBlank | None = None
    entities: list[KISResolutionEntity] = Field(default_factory=list)
    events: list[KISResolutionEvent] = Field(min_length=1)


class KISIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=1)
    query_text: NonBlank | None
    entities: list[KISEntity] = Field(default_factory=list)
    events: list[KISEvent] = Field(min_length=1)
    temporal_edges: list[KISTemporalEdge] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_language(cls, value: object) -> object:
        if isinstance(value, dict) and "language" in value:
            return {key: item for key, item in value.items() if key != "language"}
        return value
```

Change graph validation to require `query_text` exactly when at least one event has text:

```python
has_text = any(event.text is not None for event in self.events)
if has_text != (self.query_text is not None):
    raise ValueError("query_text must be present exactly when the intent contains text")
```

Preserve all other graph validation unchanged.

- [ ] **Step 4: Run contract tests and confirm GREEN**

Run:

```bash
aic/bin/python -m pytest tests/kis/test_models.py -q
```

Expected: all model tests pass.

- [ ] **Step 5: Inspect the focused diff without staging user work**

Run:

```bash
git diff --check -- src/hcmai/kis/models.py tests/kis/test_models.py
git diff -- src/hcmai/kis/models.py tests/kis/test_models.py
```

Expected: no whitespace errors; unrelated pre-existing fallback edits remain preserved.

---

### Task 2: Remove Language From Semantic Resolution

**Requirements:** REQ-001, REQ-003, REQ-007

**Files:**
- Modify: `tests/hcmai/kis/test_resolver.py`
- Modify: `tests/hcmai/kis/test_scoped_resolver.py`
- Modify: `tests/hcmai/kis/test_rewriter.py`
- Modify: `src/hcmai/kis/prompts.py`
- Modify: `src/hcmai/kis/resolver.py`
- Modify: `src/hcmai/kis/scoped_resolver.py`
- Modify: `src/hcmai/kis/rewriter.py`

**Interfaces:**
- Consumes: language-free contracts from Task 1.
- Produces: initial, scoped, and global operations whose LLM schemas require semantic content only.

- [ ] **Step 1: Write failing resolver tests**

Remove `language=` from all resolution fixtures and add explicit schema coverage:

```python
def test_REQ_003_scoped_response_without_language_updates_textual_intent() -> None:
    llm = Mock()
    llm.generate_structured.return_value = ScopedResolutionBatch(
        events=[ScopedResolvedEvent(event_id="E2", text="Con chó được cập nhật")]
    )
    result = KISScopedResolver(llm).resolve(
        _base_intent(),
        [EventPatchInstruction(event_id="E2", instruction="cập nhật con chó")],
    )
    updated = apply_scoped_resolutions(_base_intent(), result, revision=2)
    assert updated.events[1].text == "Con chó được cập nhật"
    assert "language" not in result.model_dump()
```

Convert the image-only-to-text test to assert only `query_text` and event/image preservation. Add equivalent assertions that `KISResolution` and `GlobalRewriteResolution` JSON schemas omit `language`.

- [ ] **Step 2: Run semantic tests and confirm RED**

Run:

```bash
aic/bin/python -m pytest \
  tests/hcmai/kis/test_resolver.py \
  tests/hcmai/kis/test_scoped_resolver.py \
  tests/hcmai/kis/test_rewriter.py -q
```

Expected: construction or schema assertions fail because scoped/global models still require language.

- [ ] **Step 3: Implement language-free prompts and resolvers**

Apply these minimal changes:

- Delete the `language` property from all three LLM response models.
- Delete `_validate_language` and every language comparison or inference branch.
- In initial resolution, retain `query_text = resolution.query_text or " ".join(normalized)`.
- In scoped application, derive only `query_text = canonical_query_text(events)` and construct `KISIntent` without language.
- In global rewrite, retain `query_text = resolved.query_text or canonical_query_text(events)` and set it to `None` only when all events are image-only.
- Remove language instructions from initial/scoped prompts and update module docstrings that describe language reasoning.
- Keep event-ID scope, binding, image, revision, and topology checks byte-for-byte equivalent where possible.

The resulting scoped output shape is:

```python
class ScopedResolutionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[ScopedResolvedEvent] = Field(min_length=1)
```

The resulting global output shape is:

```python
class GlobalRewriteResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_text: NonBlank | None
    entities: list[KISEntity] = Field(default_factory=list)
    events: list[GlobalRewriteEvent] = Field(min_length=1)
```

- [ ] **Step 4: Run semantic tests and confirm GREEN**

Run the Step 2 command again.

Expected: all initial, scoped, and global resolution tests pass without language metadata.

- [ ] **Step 5: Check for residual language logic in the KIS domain**

Run:

```bash
rg -n "language|_validate_language" src/hcmai/kis tests/hcmai/kis tests/kis
```

Expected: only natural-language prose unrelated to a field may remain; no `.language`, `language=`, prompt requirement, or schema field remains.

---

### Task 3: Route Multilingual Text Directly Through KIS

**Requirements:** REQ-004, REQ-005, REQ-006, REQ-007

**Files:**
- Modify: `tests/orchestration/test_kis_revision.py`
- Modify: `tests/orchestration/workflows/test_kis_pipeline.py`
- Modify: `tests/orchestration/test_setup_modules.py`
- Modify: `tests/orchestration/test_health.py`
- Modify: `tests/api/test_kis_router.py`
- Modify: `tests/hcmai/inference/test_clients.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/workflows/kis.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/utils/health.py`
- Modify: `src/hcmai/api/routers/kis.py`

**Interfaces:**
- Consumes: canonical language-free `KISIntent.events`.
- Produces: a `KISRetrievalPlan` whose dense/BM25 views preserve original multilingual text and a latency response with `translation_ms=0.0`.

- [ ] **Step 1: Write failing orchestration tests**

Replace the Vietnamese translation test with direct multilingual projection:

```python
def test_REQ_004_vietnamese_text_uses_direct_multilingual_views(self) -> None:
    intent = KISIntent(
        revision=1,
        query_text="Một người phụ nữ trong bếp.",
        entities=[],
        events=[KISEvent(id="E1", text="Một người phụ nữ trong bếp.")],
        temporal_edges=[],
    )
    intent_resolver = Mock()
    intent_resolver.resolve_initial.return_value = intent
    service = self._make_service(intent_resolver=intent_resolver)

    response = service.search_kis(request)

    event = response.exploration_seed.events[0]
    assert event.dense_text == "Một người phụ nữ trong bếp."
    assert event.bm25_text == "Một người phụ nữ trong bếp."
    assert response.latency.translation_ms == 0.0
```

Add a setup/health assertion that `SearchService` has no `event_translator` dependency and health output has no `event_translation` capability. Change workflow timing coverage to assert `query_ms == intent_ms` and `translation_ms == 0.0` without passing a translation parameter.

- [ ] **Step 2: Run orchestration tests and confirm RED**

Run:

```bash
aic/bin/python -m pytest \
  tests/orchestration/test_kis_revision.py \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/orchestration/test_setup_modules.py \
  tests/orchestration/test_health.py \
  tests/api/test_kis_router.py -q
```

Expected: direct Vietnamese dense text and dependency-shape assertions fail under the current translator path.

- [ ] **Step 3: Implement direct multilingual orchestration**

In `SearchService.search_kis`, replace translation routing with direct projection:

```python
dense_map = {
    index: event.text
    for index, event in enumerate(intent.events)
    if request.use_dense and event.text is not None
}
```

Then:

- Remove `EventTranslator` from `SearchService.__init__`, type-checking imports, attributes, and setup injection.
- Remove `_load_event_translator()` and its setup tests; do not touch the generic translation package.
- Remove the event-translation readiness flag from `build_health_report()`.
- Remove `translation_ms` from `KISPipeline.execute()` parameters; calculate `query_ms = intent_ms` and construct `SearchLatency` without overriding its default zero translation value.
- Stop passing translation timing from `SearchService` into the workflow.
- Remove `EventTranslationError` from the KIS router imports and 502 tuple.
- Remove language construction/update keys from image-only, explicit-initial, and patch assembly paths.
- Keep `query_text` recomputation and every canonical event/image/topology operation unchanged.

- [ ] **Step 4: Run orchestration tests and confirm GREEN**

Run the Step 2 command again.

Expected: all focused orchestration, setup, health, workflow, and router tests pass.

- [ ] **Step 5: Add and verify the original failure boundary regression**

Add a fake raw LLM transport test that returns scoped JSON without `language`:

```json
{"events":[{"event_id":"E1","text":"Con chó được cập nhật","bindings":[]}]}
```

Run:

```bash
aic/bin/python -m pytest \
  tests/hcmai/inference/test_clients.py \
  tests/hcmai/kis/test_scoped_resolver.py \
  tests/orchestration/test_kis_revision.py -q
```

Expected: the raw response validates and the update produces revision 2 without a 502-path exception.

---

### Task 4: Migrate Fixtures and Verify the Full KIS Surface

**Requirements:** REQ-001 through REQ-007

**Files:**
- Modify: `tests/api/test_kis_contracts.py`
- Modify: `tests/api/test_dres_result_logging.py`
- Modify: `tests/event_trail/test_kis_integration.py`
- Modify: `tests/orchestration/test_kis_multimodal_prerequisites.py`
- Modify: `tests/test_kis_acceptance_smoke.py`
- Modify: `frontend/src/api/kis.test.js`
- Modify: `frontend/src/features/kis/parser.test.js`
- Modify: `frontend/src/features/kis/session.test.js`
- Modify: `frontend/src/features/kis/components/KisPanel.test.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `frontend/src/features/workspace/queryHistory.test.js`

**Interfaces:**
- Consumes: the completed language-free runtime.
- Produces: fixtures and acceptance coverage matching the new public response shape.

- [ ] **Step 1: Mechanically migrate remaining fixtures**

Remove obsolete `language=` constructor arguments, `"language": ...` fixture keys, translator mocks/injections, and language assertions. Replace translation assertions with direct canonical-text assertions. Do not alter unrelated VBS, EventTrail, retrieval identity, or frontend behavior.

- [ ] **Step 2: Run focused backend suites**

Run:

```bash
aic/bin/python -m pytest \
  tests/kis \
  tests/hcmai/kis \
  tests/api/test_kis_contracts.py \
  tests/api/test_kis_router.py \
  tests/api/test_dres_result_logging.py \
  tests/orchestration/test_kis_revision.py \
  tests/orchestration/test_kis_multimodal_prerequisites.py \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/event_trail/test_kis_integration.py \
  tests/test_kis_acceptance_smoke.py -q
```

Expected: all focused backend tests pass.

- [ ] **Step 3: Run focused frontend suites**

Run from `frontend/`:

```bash
CI=true npm test -- --watchAll=false \
  src/api/kis.test.js \
  src/features/kis/parser.test.js \
  src/features/kis/session.test.js \
  src/features/kis/components/KisPanel.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/features/workspace/queryHistory.test.js
```

Expected: all focused frontend tests pass with language-free intent fixtures.

- [ ] **Step 4: Run residual scans**

Run:

```bash
rg -n "\.language\b|language=|\"language\"|event_translator|EventTranslationError" \
  src/hcmai/kis src/hcmai/orchestration src/hcmai/api/routers/kis.py \
  tests/kis tests/hcmai/kis tests/orchestration tests/event_trail \
  tests/api/test_kis_contracts.py tests/api/test_kis_router.py \
  tests/test_kis_acceptance_smoke.py frontend/src
```

Expected: no KIS runtime or fixture references remain. Generic translation service/config tests outside the KIS runtime may still contain language parameters by design.

- [ ] **Step 5: Run broad Python verification**

Run:

```bash
aic/bin/python -m pytest -q
```

Expected: full Python suite passes. If failures are unrelated pre-existing failures, record exact test names and do not alter unrelated user work.

- [ ] **Step 6: Run final diff and schema checks**

Run:

```bash
git diff --check
PYTHONPATH=src aic/bin/python - <<'PY'
from hcmai.kis.models import KISIntent, KISResolution
from hcmai.kis.rewriter import GlobalRewriteResolution
from hcmai.kis.scoped_resolver import ScopedResolutionBatch

for model in (KISIntent, KISResolution, ScopedResolutionBatch, GlobalRewriteResolution):
    assert "language" not in model.model_json_schema().get("properties", {})
print("language-free KIS schemas verified")
PY
git status --short
```

Expected: no whitespace errors, schema assertion succeeds, and only intended files plus pre-existing user changes are present.

## Completion Report

Report:

- language removed from new KIS schemas and responses;
- legacy `KISIntent.language` input compatibility status;
- original scoped-update reproduction result;
- focused and broad test counts;
- frontend test result;
- retained `translation_ms=0.0` compatibility;
- generic translation package intentionally retained;
- no `KNOWLEDGE.md` update because no retrieval-quality claim was made.
