# Unified Multimodal Progressive KIS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the incomplete KIS stabilization work, then replace clue-list/separate-image KIS with one stateless revisioned multimodal event pipeline supporting initial natural text, explicit `E#:` scoped updates, `/llm-rewrite`, image-only/text+image events, and retrieval-only re-search.

**Architecture:** Preserve `KISIntent` as the only canonical semantic snapshot, keep event IDs and image ownership server-controlled, and project each event into a `KISRetrievalPlan` consumed by the existing temporal evidence fusion + DP path. Complete S0 first (translation ownership, retrieval-plan alignment, transactional UI, readiness/history cleanup), then land S1 operation contracts, image assets, scoped/global resolution, multimodal retrieval, and the Query Composer. EventTrail is intentionally excluded from this plan.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, NumPy, Pillow, existing SigLIP/BGE retrieval stack, React 19, Jest/React Testing Library.

**Spec:** [`../specs/2026-09-15-unified-multimodal-kis-design.md`](../specs/2026-09-15-unified-multimodal-kis-design.md) (repository path: `docs/superpowers/specs/2026-09-15-unified-multimodal-kis-design.md`)

## Global Constraints

- EventTrail UI and confirm/reject/anchor redesign are out of scope for this plan.
- Once an intent exists, unscoped free-form progressive mutation is invalid; use explicit `E#:` patches or `/llm-rewrite`.
- `E#:` routes semantic resolution to named events; it is not literal text replacement.
- `/llm-rewrite` may rebuild text/entities/bindings/query text but must preserve event IDs, event count, event order, image ownership, and adjacent temporal topology.
- Image attachment/removal is deterministic and never delegated to the LLM.
- Initial image-only and unscoped text+image requests create exactly one `E1`; initial natural text-only may decompose into multiple events.
- A successful semantic operation increments revision exactly once; `SearchOnly` leaves revision unchanged.
- Images remain canonical user evidence; do not auto-caption them into canonical text or BM25 evidence. Image-only intents use `language=None` and `query_text=None` rather than invented text/language.
- DP remains modality-agnostic and consumes one score row per event.
- Do not introduce a new image-text embedding-fusion research method; image evidence enters the existing calibrated/reliability-aware component fusion.
- Replace/migrate/delete obsolete query-expansion, clue-list semantic state, and separate product-level image-KIS paths; do not keep compatibility shims without a verified independent consumer.
- Frontend semantic updates are transactional: committed intent/results/exploration remain usable while pending and survive failure.

---

## File Structure Target

Create or converge toward these focused ownership units before starting task implementation:

```text
src/hcmai/kis/
  models.py                 canonical KIS semantic models + multimodal event/image refs
  parser.py                 deterministic E#: and /llm-rewrite command grammar
  resolver.py               initial natural-text resolution
  scoped_resolver.py        named-event semantic updates
  rewriter.py               global topology-preserving rewrite
  assets.py                 bounded content-addressed query image storage

src/hcmai/retrieval/
  plan.py                   KISRetrievalEvent / KISRetrievalPlan
  translation/
    __init__.py
    cache.py
    models.py
    prompts.py
    service.py              EventTranslator only
  evidence/image_query.py   full-corpus image-query temporal component

src/hcmai/api/contracts/kis.py
src/hcmai/api/routers/kis.py
src/hcmai/orchestration/pipeline.py
src/hcmai/orchestration/workflows/kis.py
src/hcmai/orchestration/workflows/temporal_search.py
src/hcmai/orchestration/setup.py
src/hcmai/orchestration/health.py

frontend/src/features/kis/
  session.js
  parser.js
  components/KisPanel.jsx
  components/IntentSummary.jsx
  components/EventList.jsx
  components/EventCard.jsx
  components/QueryComposer.jsx
frontend/src/features/search/components/SearchWorkspace.jsx
frontend/src/api/kis.js
```

Do not split a file merely to satisfy this tree. Create the focused modules named above when their responsibilities are new; preserve existing files when they already have a single clear responsibility.

---

### Task 1: Finish S0 Translation Ownership and Delete Query Expansion

**Files:**

- Create: `src/hcmai/retrieval/translation/__init__.py`
- Create: `src/hcmai/retrieval/translation/cache.py`
- Create: `src/hcmai/retrieval/translation/models.py`
- Create: `src/hcmai/retrieval/translation/prompts.py`
- Create: `src/hcmai/retrieval/translation/service.py`
- Modify: `src/hcmai/common/config.py:477-516`
- Modify: `configs/baseline.yaml:119-123`
- Modify: `src/hcmai/orchestration/setup.py:60-100,139-170`
- Modify: `src/hcmai/orchestration/pipeline.py:9-43,49-76,287-305`
- Modify: `src/hcmai/orchestration/health.py:102-120`
- Modify: `src/hcmai/api/contracts/__init__.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/app.py:20-30,200-210`
- Delete: `src/hcmai/query_preparation/`
- Delete: `src/hcmai/api/contracts/query_candidates.py`
- Delete: `src/hcmai/api/routers/query_candidates.py`
- Delete: `tests/query_preparation/`
- Test: `tests/retrieval/translation/test_service.py`
- Test: `tests/api/test_removed_routes.py`
- Test: `tests/orchestration/test_setup_modules.py`
- Test: `tests/orchestration/test_health.py`

**Interfaces:**

- Consumes: `LLMClient.generate_structured(messages, response_model, temperature=0.0)` and `LLMClient.model`.
- Produces: `EventTranslator.translate(events: Sequence[str], language: str) -> tuple[str, ...]`.
- Produces: `EventTranslationConfig(prompt_version, cache_enabled, cache_ttl_seconds, cache_max_entries)` under `AppConfig.event_translation`.

- [X] **Step 1: Write failing translation-service tests that encode the reduced responsibility**

```python
from unittest.mock import Mock

from hcmai.common.config import EventTranslationConfig
from hcmai.retrieval.translation.models import LiteralTranslation
from hcmai.retrieval.translation.service import EventTranslationError, EventTranslator


def test_translator_returns_english_input_without_llm_call() -> None:
    llm = Mock(model="model-a")
    service = EventTranslator(llm, EventTranslationConfig())

    assert service.translate(["woman enters", "she sits"], language="en") == (
        "woman enters",
        "she sits",
    )
    llm.generate_structured.assert_not_called()


def test_translator_uses_actual_llm_model_in_cache_identity() -> None:
    llm = Mock(model="provider/model-a")
    llm.generate_structured.return_value = LiteralTranslation(events=["a woman enters"])
    service = EventTranslator(llm, EventTranslationConfig())

    first = service.translate(["một phụ nữ bước vào"], language="vi")
    second = service.translate(["một phụ nữ bước vào"], language="vi")

    assert first == second == ("a woman enters",)
    assert llm.generate_structured.call_count == 1
```

Also add a failure case asserting event-count mismatch raises `EventTranslationError` and no candidate-generation API exists on the service.

- [X] **Step 2: Run the new tests and verify they fail because translation still lives under `query_preparation`**

Run:

```bash
PYTHONPATH=src python -m pytest tests/retrieval/translation/test_service.py -q
```

Expected: import failures for `hcmai.retrieval.translation` and `EventTranslationConfig`.

- [X] **Step 3: Introduce `EventTranslationConfig` and remove query-expansion config**

Replace `QueryPreparationConfig` with:

```python
class EventTranslationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_version: str = Field(default="event-translation-v1", min_length=1)
    cache_enabled: bool = True
    cache_ttl_seconds: float = Field(default=3600, gt=0)
    cache_max_entries: int = Field(default=2048, ge=1)
```

Change `AppConfig` to:

```python
event_translation: EventTranslationConfig = Field(default_factory=EventTranslationConfig)
```

Change `configs/baseline.yaml` to:

```yaml
event_translation:
  prompt_version: "event-translation-v1"
  cache_enabled: true
  cache_ttl_seconds: 3600
  cache_max_entries: 2048
```

Do not retain `model_name`, `model_revision`, `candidate_count`, or candidate prompt settings.

- [X] **Step 4: Move literal translation into `EventTranslator` and make cache identity use `llm.model`**

`service.py` must expose exactly:

```python
class EventTranslationError(RuntimeError):
    pass


class EventTranslator:
    def __init__(
        self,
        llm: LLMClient,
        config: EventTranslationConfig,
        cache: EventTranslationCache | None = None,
    ) -> None:
        self._llm = llm
        self._config = config
        self._cache = cache or EventTranslationCache(
            max_entries=config.cache_max_entries,
            ttl_seconds=config.cache_ttl_seconds,
        )

    def translate(self, events: Sequence[str], language: str) -> tuple[str, ...]:
        normalized = normalize_event_texts(events)
        if language == "en":
            return normalized
        key = translation_cache_key(
            events=normalized,
            model=self._llm.model,
            prompt_version=self._config.prompt_version,
        )
        # cached lookup, structured LiteralTranslation call, strict aligned validation
```

`translation_cache_key()` must contain operation name, `llm.model`, prompt version, and case-preserving normalized event text. There is no model revision field because endpoint model identity is the configured client identity.

- [X] **Step 5: Remove the query-candidate HTTP/service chain and update route-registration tests**

Delete imports/exports and `SearchService.generate_query_candidates`. Update `test_removed_routes.py` to assert:

```python
paths = {route.path for route in app.routes}
assert "/api/v1/query-candidates" not in paths
```

Also assert `SearchService` no longer exposes `generate_query_candidates`.

- [X] **Step 6: Wire `EventTranslator` in setup and readiness**

Replace `_load_query_preparation()` with:

```python
def _load_event_translator(
    settings: AppConfig,
    messages: list[str],
    llm: LLMClient | None,
) -> EventTranslator | None:
    if llm is None:
        messages.append("Event translation unavailable: LLM client not configured")
        return None
    return EventTranslator(llm, settings.event_translation)
```

`SearchService` constructor field becomes `event_translator`. Health capabilities expose `event_translation` rather than `query_preparation`.

- [X] **Step 7: Run S0 translation/route/setup tests**

Run:

```bash
PYTHONPATH=src python -m pytest \
  tests/retrieval/translation/test_service.py \
  tests/api/test_removed_routes.py \
  tests/orchestration/test_setup_modules.py \
  tests/orchestration/test_health.py -q
```

Expected: PASS.

- [X] **Step 8: Commit the S0 translation/query-expansion removal**

```bash
git add configs/baseline.yaml src/hcmai tests/retrieval/translation tests/api/test_removed_routes.py tests/orchestration

git commit -m "refactor: retire query expansion and isolate event translation"
```

---

### Task 2: Finish S0 Textual `KISRetrievalPlan` and Product Response Cleanup

**Files:**

- Create: `src/hcmai/retrieval/plan.py`
- Modify: `src/hcmai/orchestration/pipeline.py:166-220`
- Modify: `src/hcmai/orchestration/workflows/kis.py:24-104`
- Modify: `src/hcmai/orchestration/workflows/temporal_exploration.py`
- Modify: `src/hcmai/api/contracts/kis.py:55-66`
- Modify: `src/hcmai/api/contracts/exploration.py`
- Modify: `src/hcmai/api/routers/exploration.py`
- Modify: `src/hcmai/api/contracts/latency.py`
- Modify: `tests/orchestration/test_kis_revision.py`
- Modify: `tests/orchestration/workflows/test_kis_pipeline.py`
- Modify: `tests/orchestration/workflows/test_temporal_exploration.py`
- Modify: `tests/api/test_kis_contracts.py`
- Modify: `tests/api/test_exploration_router.py`
- Modify: `frontend/src/features/alignment/hooks/useTemporalExploration.js`
- Modify: `frontend/src/features/alignment/hooks/useTemporalExploration.test.js`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`

**Interfaces:**

- Consumes: canonical `KISIntent.events` and `EventTranslator.translate()`.
- Produces: immutable `KISRetrievalEvent` and `KISRetrievalPlan` aligned by `event_id`.
- Produces: `KISExplorationSeed` as the dedicated selected-video scoring snapshot; top-level `dense_events` / `bm25_events` disappear only after this migration.

- [X] **Step 1: Write failing retrieval-plan alignment tests**

```python
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan


def test_plan_requires_sequential_event_ids_and_aligned_views() -> None:
    plan = KISRetrievalPlan(
        events=(
            KISRetrievalEvent(
                event_id="E1",
                canonical_text="một phụ nữ vào bếp",
                dense_text="a woman enters a kitchen",
                bm25_text="một phụ nữ vào bếp",
                image_refs=(),
            ),
        )
    )
    assert plan.event_ids == ("E1",)
    assert plan.dense_texts == ("a woman enters a kitchen",)
```

Add failure cases for non-sequential IDs and for an event with no text and no image refs once the final S1 model is available; in this S0 task only text rows are constructed.

- [X] **Step 2: Run the plan tests and verify import failure**

```bash
PYTHONPATH=src python -m pytest tests/orchestration/workflows/test_kis_pipeline.py -q
```

Expected: FAIL because `hcmai.retrieval.plan` does not exist and KIS pipeline still takes loose `retrieval_events`.

- [X] **Step 3: Implement the initial text-capable retrieval plan**

Create:

```python
@dataclass(frozen=True, slots=True)
class KISRetrievalEvent:
    event_id: str
    canonical_text: str | None
    dense_text: str | None
    bm25_text: str | None
    image_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KISRetrievalPlan:
    events: tuple[KISRetrievalEvent, ...]

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.events)

    @property
    def dense_texts(self) -> tuple[str, ...] | None:
        values = tuple(event.dense_text for event in self.events)
        return None if any(value is None for value in values) else tuple(str(value) for value in values)

    @property
    def bm25_texts(self) -> tuple[str, ...] | None:
        values = tuple(event.bm25_text for event in self.events)
        return None if any(value is None for value in values) else tuple(str(value) for value in values)
```

Validate event IDs are exactly `E1..En`. Keep `image_refs` as an empty tuple in S0 so Task 9 can extend it after the canonical `KISImageRef` type exists.

- [X] **Step 4: Build the plan in `SearchService` after semantic resolution**

For each canonical event:

```python
canonical = tuple(event.text for event in intent.events)
translation_started = perf_counter()
if request.use_dense and intent.language != "en":
    if self.event_translator is None:
        raise SearchServiceUnavailableError("Event translation capability is unavailable")
    dense = self.event_translator.translate(canonical, intent.language)
else:
    dense = canonical if request.use_dense else None
translation_ms = (perf_counter() - translation_started) * 1_000

plan = KISRetrievalPlan(
    events=tuple(
        KISRetrievalEvent(
            event_id=event.id,
            canonical_text=event.text,
            dense_text=None if dense is None else dense[index],
            bm25_text=event.text if request.use_bm25 else None,
        )
        for index, event in enumerate(intent.events)
    )
)
```

Validate `plan.event_ids == tuple(event.id for event in intent.events)` before execution.

- [X] **Step 5: Change `KISPipeline.execute()` to consume the plan, not parallel lists**

Use:

```python
def execute(
    self,
    *,
    intent: KISIntent,
    retrieval_plan: KISRetrievalPlan,
    use_dense: bool,
    use_bm25: bool,
    top_k: int,
    intent_ms: float = 0.0,
    translation_ms: float = 0.0,
) -> KISSearchExecution:
```

For the S0 textual path, derive the existing temporal arguments from the plan and reject missing required text rows. Do not re-derive retrieval strings from `KISIntent` elsewhere.

- [X] **Step 6: Make latency stage ownership explicit**

`SearchLatency` must support:

```python
intent_ms: float = Field(default=0.0, ge=0)
translation_ms: float = Field(default=0.0, ge=0)
retrieval_ms: float = Field(default=0.0, ge=0)
alignment_ms: float = Field(default=0.0, ge=0)
materialization_ms: float = Field(default=0.0, ge=0)
total_ms: float = Field(default=0.0, ge=0)
```

If `query_ms` is retained for another endpoint, derive KIS query timing from `intent_ms + translation_ms`; do not count translation outside user-visible latency.

- [X] **Step 7: Add a dedicated exploration seed before removing prepared-string response fields**

Add transport models in `src/hcmai/api/contracts/kis.py`:

```python
class KISExplorationEventSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: EventId
    canonical_text: str | None = None
    dense_text: str | None = None
    bm25_text: str | None = None


class KISExplorationSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    semantic_revision: int = Field(ge=1)
    events: list[KISExplorationEventSeed] = Field(min_length=1)
    use_dense: bool
    use_bm25: bool
```

Build the seed from the committed `KISRetrievalPlan`, never by reconstructing strings in the frontend. Task 9 extends `KISExplorationEventSeed` with typed `image_refs` after `KISImageRef` exists; adding that optional field is the only S1 extension to this seed shape.

The KIS response becomes:

```python
class KISRevisionSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: KISIntent
    use_dense: bool
    use_bm25: bool
    exploration_seed: KISExplorationSeed
    results: list[SearchResult] = Field(default_factory=list)
    latency: SearchLatency
    warnings: list[str] = Field(default_factory=list)
```

Only after `exploration_seed` is wired may top-level `dense_events` and `bm25_events` be deleted.

- [X] **Step 8: Migrate temporal exploration to consume the seed**

Replace `ExplorationOpenRequest.events/retrieval_events/caption_events/query` with the dedicated seed plus selected-video/window fields:

```python
class ExplorationOpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: KISExplorationSeed
    video_id: _NonBlankString
    window: _Interval
```

Update `QueryBinding` to retain the ordered seed/plan projection instead of three parallel text tuples. During S0, `_open_branch()` may derive the existing textual `TemporalSearchService.score_videos()` arguments from the seed, but it must validate event IDs/order and reject a seed row with no textual scoring source. Task 9 removes that textual-only restriction when `score_plan()` and image evidence exist.

Update `useTemporalExploration.validSnapshot()` and `buildExplorationOpenBody()` so the frontend stores `response.exploration_seed` unchanged and sends `{ seed, video_id, window }`. Do not inspect `dense_events`/`bm25_events` aliases. Add a regression test proving exploration still opens after those top-level fields are removed.

- [X] **Step 9: Run textual plan, response, and exploration migration tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/orchestration/test_kis_revision.py \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/orchestration/workflows/test_temporal_exploration.py \
  tests/api/test_kis_contracts.py \
  tests/api/test_exploration_router.py -q
cd frontend
CI=true npm test -- --watchAll=false src/features/alignment/hooks/useTemporalExploration.test.js
```

Expected: PASS.

- [X] **Step 10: Commit textual retrieval-plan and exploration-snapshot migration**

```bash
git add src/hcmai/retrieval/plan.py src/hcmai/orchestration src/hcmai/api/contracts src/hcmai/api/routers/exploration.py frontend/src/features/alignment tests/orchestration tests/api

git commit -m "refactor: align KIS retrieval and exploration through event plans"
```

---

### Task 3: Finish S0 Transactional Frontend, Frame Selection, Replay, and Best-Effort History

**Files:**

- Modify: `frontend/src/features/kis/session.js`
- Modify: `frontend/src/features/kis/session.test.js`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx:223-274,321-461`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `frontend/src/App.jsx:16-105`
- Modify: `frontend/src/App.test.jsx`
- Modify: `frontend/src/features/workspace/components/ReplayResults.jsx`
- Modify: `frontend/src/features/workspace/components/ReplayResults.test.jsx`

**Interfaces:**

- Consumes: committed KIS response from Task 2.
- Produces: pending semantic request state that never destroys committed results before success.
- Produces: one frame selection callback carrying the committed exploration snapshot.

- [X] **Step 1: Add regression tests for pending success/failure and single frame callback**

Add tests equivalent to:

```javascript
const frameResult = (id) => ({
  frame_id: id,
  video_id: 'V01',
  frame_idx: id === 'frame-1' ? 100 : 200,
  timestamp_ms: id === 'frame-1' ? 4_000 : 8_000,
  fps: 25,
  caption: id,
  scores: { final: 0.9 },
});

test('keeps committed results visible while the next revision is pending', async () => {
  let resolveSecond;
  searchKis
    .mockResolvedValueOnce(mockKisResponse({ queryText: 'first', revision: 1, results: [frameResult('frame-1')] }))
    .mockImplementationOnce(() => new Promise((resolve) => { resolveSecond = resolve; }));

  renderSearch({ topK: 20, setTopK: jest.fn() });
  submit('first');
  expect(await screen.findByAltText('Frame frame-1')).toBeTruthy();

  submit('second');
  expect(screen.getByAltText('Frame frame-1')).toBeTruthy();

  resolveSecond(mockKisResponse({ queryText: 'second', revision: 2, results: [frameResult('frame-2')] }));
  expect(await screen.findByAltText('Frame frame-2')).toBeTruthy();
});


test('opens a KIS result exactly once and records one viewed-frame write', async () => {
  const onFrameClick = jest.fn();
  renderSearch({ topK: 20, setTopK: jest.fn(), onFrameClick });
  submit('red boat');
  const frame = await screen.findByAltText('Frame frame-explore');
  fireEvent.click(frame);

  expect(onFrameClick).toHaveBeenCalledTimes(1);
  expect(onFrameClick).toHaveBeenCalledWith(expect.objectContaining({
    explorationSnapshot: expect.any(Object),
  }));
  await waitFor(() => expect(markFrameViewed).toHaveBeenCalledTimes(1));
  expect(markFrameViewed).toHaveBeenCalledWith(expect.objectContaining({
    frameId: 'frame-explore',
  }));
});
```

Also add a failed-second-request case asserting first intent/results remain and the draft remains editable.

- [X] **Step 2: Run the frontend regression tests and verify failures on current clearing/double-callback behavior**

```bash
cd frontend
CI=true npm test -- --watchAll=false src/features/search/components/SearchWorkspace.test.jsx src/features/kis/session.test.js
```

Expected: pending-result and callback-count tests FAIL.

- [X] **Step 3: Make session state distinguish committed state from pending request metadata**

For this S0 task, retain the current clue request format but stop committing draft changes early. `prepareSearchRequest()` returns a payload plus pending metadata; it must not alter committed inputs/revision/current intent.

```javascript
export const prepareSearchRequest = (state) => {
  const draft = state.draft.trim();
  if (!draft) throw new Error('Search clue cannot be empty');
  return {
    nextState: { ...state, isSearching: true, error: null },
    requestPayload: {
      inputs: [...state.committedInputs, draft].map((text) => ({ text })),
      expectedRevision: state.revision,
    },
  };
};
```

Only `commitSearchSuccess()` changes committed semantic state.

- [X] **Step 4: Stop clearing committed KIS UI before the request succeeds**

In `SearchWorkspace.submit()` remove pre-request calls that clear:

```text
frames
kisEvents
searchLatencyMs
liveKisSnapshotRef
active committed query context
```

Set only pending/error UI fields before request. On success, atomically update committed response-derived state. On failure, leave committed state unchanged and call `commitSearchFailure()` so draft survives.

- [X] **Step 5: Fix `openCanonicalFrame()` to call `onFrameClick` once**

Replace the current two calls with:

```javascript
const openCanonicalFrame = useCallback((frame) => {
  recordViewed(frame);
  onFrameClick?.({
    frame,
    ...(liveKisSnapshotRef.current
      ? { explorationSnapshot: liveKisSnapshotRef.current }
      : {}),
  });
}, [onFrameClick, recordViewed]);
```

- [X] **Step 6: Make global active query follow committed intent, never draft**

Remove draft-driven `onQueryChange` effects/callbacks. After successful commit call:

```javascript
onQueryChange?.(response.intent?.query_text || '');
```

Typing a new draft must not change `App.modalQuery`/inspector query until a semantic request succeeds.

- [X] **Step 7: Move history persistence after live commit and serialize writes per query**

The live KIS commit and search lock release happen immediately after retrieval success. History is best-effort, but it is not unordered. Create a per-query write queue in `SearchWorkspace` (or a focused helper) keyed by `queryId`:

```javascript
const historyQueuesRef = useRef(new Map());

const enqueueHistoryWrite = useCallback((queryId, write) => {
  const prior = historyQueuesRef.current.get(queryId) || Promise.resolve();
  const next = prior.then(write); // a failed query-row create skips dependent writes
  historyQueuesRef.current.set(queryId, next);
  next.catch(() => undefined); // observe rejection without converting queue state to success
  return next;
}, []);
```

On live commit, establish the session identity immediately, then enqueue `createQueryHistory()` as the first write for that `queryId`. `recordViewed(frame)` keeps its optimistic local update but enqueues `markFrameViewed()` behind query creation, so a fast click cannot race a missing row. Interaction-event writes added in Task 12 use the same queue.

Late history promises must never replace the active session. Capture the `queryId`/generation and only update history status or warnings when it still matches the relevant session; do not call `setActiveQuerySession()` from a late create response to reactivate an older query. If creation fails, queued dependent writes for that query are skipped/rejected best-effort and the live retrieval result remains committed.

Add tests for: (1) a frame clicked before `createQueryHistory` resolves does not call `markFrameViewed` until creation resolves; (2) a late history completion from search A cannot overwrite active search B; and (3) history failure never rolls back live results.

- [X] **Step 8: Make replay read-only instead of copying historical query into draft**

On replay set a replay snapshot/mode and initialize live KIS session separately. `setDraft(createInitialKisSessionState(), item.query_text)` must disappear. Replay rendering uses stored snapshot only; `New Search` returns to a clean live session.

- [X] **Step 9: Run frontend S0 tests**

```bash
cd frontend
CI=true npm test -- --watchAll=false \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/features/kis/session.test.js \
  src/App.test.jsx \
  src/features/workspace/components/ReplayResults.test.jsx
```

Expected: PASS.

- [X] **Step 10: Commit the transactional frontend stabilization**

```bash
git add frontend/src/features/kis frontend/src/features/search/components/SearchWorkspace.jsx frontend/src/features/search/components/SearchWorkspace.test.jsx frontend/src/App.jsx frontend/src/App.test.jsx frontend/src/features/workspace

git commit -m "fix: make KIS revisions transactional in the frontend"
```

---

### Task 4: Finish S0 KIS Readiness and HTTP Error Boundaries

**Files:**

- Modify: `src/hcmai/orchestration/health.py:31-124`
- Modify: `src/hcmai/api/routers/kis.py:29-75`
- Modify: `src/hcmai/orchestration/errors.py`
- Modify: `tests/orchestration/test_health.py`
- Modify: `tests/api/test_kis_router.py`

**Interfaces:**

- Consumes: `SearchService.intent_resolver`, `SearchService.event_translator`, temporal evidence readiness.
- Produces: explicit `kis`, `intent_resolution`, `event_translation`, `retrieval` readiness fields.
- Produces: stable 409/422/502/503 routing semantics.

- [X] **Step 1: Write readiness tests for missing mandatory semantic capabilities**

```python
from types import SimpleNamespace


def _service(*, intent_resolver, event_translator, temporal_evidence):
    return SimpleNamespace(
        intent_resolver=intent_resolver,
        event_translator=event_translator,
        temporal_evidence=temporal_evidence,
    )


def test_kis_not_ready_when_intent_resolver_is_missing() -> None:
    service = _service(intent_resolver=None, event_translator=object(), temporal_evidence=object())
    report = build_health_report(service)
    assert report["capabilities"]["intent_resolution"] is False
    assert report["capabilities"]["kis"] is False


def test_kis_ready_with_resolver_and_temporal_retrieval() -> None:
    service = _service(intent_resolver=object(), event_translator=object(), temporal_evidence=object())
    report = build_health_report(service)
    assert report["capabilities"]["kis"] is True
```

- [X] **Step 2: Write router mapping tests before changing exception handling**

Assert:

```text
RevisionConflictError -> 409
InvalidQueryInputError / request validation -> 422
KISResolutionError / EventTranslationError semantic-response contract -> 502
InferenceResponseError -> 502
InferenceUnavailableError / SearchServiceUnavailableError -> 503
```

- [X] **Step 3: Implement capability-based readiness without making live LLM calls**

Set:

```python
intent_resolution_ready = service.intent_resolver is not None
event_translation_ready = service.event_translator is not None
retrieval_ready = corpus_ready and temporal_evidence is not None
kis_ready = retrieval_ready and intent_resolution_ready
```

Translation is reported independently because English text and image-only queries can remain valid without translation; KIS as a whole must not be marked ready when all initial text semantic resolution would fail.

- [X] **Step 4: Normalize router exception mapping**

Keep Pydantic/FastAPI request-shape validation as 422. Catch domain/model failures explicitly rather than mapping generic `ValidationError` to user-input failure.

- [X] **Step 5: Run health/router tests**

```bash
PYTHONPATH=src python -m pytest tests/orchestration/test_health.py tests/api/test_kis_router.py -q
```

Expected: PASS.

- [X] **Step 6: Commit the S0 readiness/error completion gate**

```bash
git add src/hcmai/orchestration/health.py src/hcmai/orchestration/errors.py src/hcmai/api/routers/kis.py tests/orchestration/test_health.py tests/api/test_kis_router.py

git commit -m "fix: align KIS readiness and inference error semantics"
```

---

### Task 5: Introduce the S1 Canonical Multimodal Intent and Deterministic Command Parser

**Files:**

- Modify: `src/hcmai/kis/models.py`
- Create: `src/hcmai/kis/parser.py`
- Create: `tests/hcmai/kis/test_parser.py`
- Modify: `tests/kis/test_models.py`
- Modify: `tests/hcmai/kis/test_resolver.py`

**Interfaces:**

- Consumes: existing entity/binding/edge types.
- Produces: `KISImageRef`, multimodal `KISEvent`, `EventPatchInstruction`, parsed composer command types.
- Produces: `parse_kis_command(text: str, *, base_event_count: int) -> ParsedKISCommand`.

- [X] **Step 1: Write failing model tests for text-only, image-only, and mixed events**

```python
def test_kis_event_accepts_image_only() -> None:
    event = KISEvent(
        id="E1",
        text=None,
        images=[KISImageRef(asset_id="sha256:abc", content_type="image/png")],
        bindings=[],
    )
    assert event.text is None
    assert event.images[0].asset_id == "sha256:abc"


def test_kis_event_rejects_empty_text_and_no_images() -> None:
    with pytest.raises(ValueError, match="text or image"):
        KISEvent(id="E1", text=None, images=[], bindings=[])
```

Update `KISIntent` tests to construct intents without `inputs` and with revision independent of event count.

- [X] **Step 2: Write failing parser tests for legal/illegal explicit batches and global rewrite**

```python
def test_parser_extracts_scoped_batch() -> None:
    command = parse_kis_command(
        "E2: chef wears black\nE4: then she takes a plate",
        base_event_count=3,
    )
    assert command.kind == "patch_events"
    assert [patch.event_id for patch in command.patches] == ["E2", "E4"]


def test_parser_rejects_gap() -> None:
    with pytest.raises(KISCommandError, match="missing E4"):
        parse_kis_command("E5: new event", base_event_count=3)


def test_parser_routes_global_rewrite() -> None:
    command = parse_kis_command(
        "/llm-rewrite\nResolve all pronouns explicitly.",
        base_event_count=3,
    )
    assert command.kind == "global_rewrite"
    assert command.instruction == "Resolve all pronouns explicitly."
```

Also test an initial explicit `E1:/E2:` batch returns `initial_explicit`, plus duplicate IDs, malformed `E:` prefixes, and unscoped progressive text with `base_event_count > 0`.

- [X] **Step 3: Run model/parser tests and verify failures**

```bash
PYTHONPATH=src python -m pytest tests/kis/test_models.py tests/hcmai/kis/test_parser.py -q
```

Expected: FAIL because multimodal models/parser are absent.

- [X] **Step 4: Change canonical models**

Add:

```python
class KISImageRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: NonBlank
    content_type: Literal["image/jpeg", "image/png", "image/webp"]


class KISEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: EventId
    text: NonBlank | None = None
    images: list[KISImageRef] = Field(default_factory=list)
    bindings: list[KISEntityBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.text is None and not self.images:
            raise ValueError("KIS event requires text or image evidence")
        return self
```

Remove `inputs` from `KISIntent`; change canonical `language` to `Literal["vi", "en"] | None` and `query_text` to `NonBlank | None` so an image-only intent does not need invented metadata. Keep revision `ge=1`, sequential event IDs, binding integrity, max count, and complete adjacent edge chain. Initial natural-text `KISResolution.language/query_text` remain required.

- [X] **Step 5: Implement a strict parser with no LLM involvement**

Use anchored event headers:

```python
_EVENT_HEADER = re.compile(r"(?m)^E([1-9]\d*):[ \t]*(.*)$")
```

`parse_kis_command()` must return one of these concrete dataclasses:

```python
@dataclass(frozen=True, slots=True)
class EventPatchInstruction:
    event_id: str
    instruction: str


@dataclass(frozen=True, slots=True)
class ParsedInitialNatural:
    kind: Literal["initial_natural"]
    text: str


@dataclass(frozen=True, slots=True)
class ParsedInitialExplicit:
    kind: Literal["initial_explicit"]
    patches: tuple[EventPatchInstruction, ...]


@dataclass(frozen=True, slots=True)
class ParsedPatchEvents:
    kind: Literal["patch_events"]
    patches: tuple[EventPatchInstruction, ...]


@dataclass(frozen=True, slots=True)
class ParsedGlobalRewrite:
    kind: Literal["global_rewrite"]
    instruction: str
```

When `base_event_count == 0`, an explicit contiguous `E1..Ek` batch returns `ParsedInitialExplicit`; when an intent already exists, explicit headers return `ParsedPatchEvents`. Collect continuation lines until the next `E#:` header, reject duplicate IDs, reject event gaps, and sort nothing: user header order must already be increasing.

- [X] **Step 6: Update initial resolver tests for server-owned revision without raw inputs**

Change the resolver contract to receive one natural query and explicit revision:

```python
intent = resolver.resolve_initial(Q1, revision=1)
assert intent.revision == 1
assert not hasattr(intent, "inputs")
```

The initial resolver still may return multiple events for one natural query.

- [X] **Step 7: Run KIS domain tests**

```bash
PYTHONPATH=src python -m pytest tests/kis/test_models.py tests/hcmai/kis/test_parser.py tests/hcmai/kis/test_resolver.py -q
```

Expected: PASS.

- [X] **Step 8: Commit the canonical multimodal domain model/parser**

```bash
git add src/hcmai/kis tests/kis tests/hcmai/kis

git commit -m "feat: add multimodal KIS events and explicit command grammar"
```

---

### Task 6: Add Scoped Event Resolution and Topology-Preserving Global Rewrite

**Files:**

- Modify: `src/hcmai/kis/prompts.py`
- Modify: `src/hcmai/kis/resolver.py`
- Create: `src/hcmai/kis/scoped_resolver.py`
- Create: `src/hcmai/kis/rewriter.py`
- Create: `tests/hcmai/kis/test_scoped_resolver.py`
- Create: `tests/hcmai/kis/test_rewriter.py`

**Interfaces:**

- Consumes: `LLMClient`, canonical `KISIntent`, parsed patch targets.
- Produces: `KISIntentResolver.resolve_initial(text: str, revision: int) -> KISIntent`.
- Produces: `KISScopedResolver.resolve(base: KISIntent | None, instructions: Sequence[EventPatchInstruction]) -> ScopedResolutionBatch`. `EventPatchInstruction` is defined in Task 5; do not introduce a second scoped-instruction type.
- Produces: `KISGlobalRewriter.rewrite(base: KISIntent, instruction: str) -> KISIntent`.

- [X] **Step 1: Write scoped-resolver tests that prove model scope is enforced**

```python
def _base_intent() -> KISIntent:
    return KISIntent(
        revision=1,
        language="en",
        query_text="A woman talks to a man, then takes a plate.",
        entities=[
            KISEntity(id="X1", kind="person", description="woman"),
            KISEntity(id="X2", kind="person", description="man"),
        ],
        events=[
            KISEvent(id="E1", text="A woman talks to a man.", bindings=[]),
            KISEvent(id="E2", text="The woman takes a plate.", bindings=[]),
        ],
        temporal_edges=[KISTemporalEdge(source="E1", target="E2", relation="before")],
    )


def _base_multimodal_intent() -> KISIntent:
    base = _base_intent()
    return base.model_copy(update={
        "events": [
            base.events[0].model_copy(update={"images": [KISImageRef(asset_id="sha256:a", content_type="image/png")]}),
            base.events[1],
        ]
    })


def test_scoped_resolver_returns_only_named_events() -> None:
    llm = Mock()
    llm.generate_structured.return_value = ScopedResolutionBatch(
        language="en",
        events=[
            ScopedResolvedEvent(
                event_id="E2",
                text="The woman talks to a chef.",
                bindings=[KISEntityBinding(entity_id="X1", role="speaker")],
            )
        ],
    )
    resolver = KISScopedResolver(llm)

    result = resolver.resolve(
        _base_intent(),
        [EventPatchInstruction(event_id="E2", instruction="chef")],
    )

    assert [event.event_id for event in result.events] == ["E2"]
    assert result.events[0].text == "The woman talks to a chef."
    assert result.events[0].bindings == [KISEntityBinding(entity_id="X1", role="speaker")]
```

Add a failure test where model returns `E1` and `E2` while only `E2` was granted; expect `KISResolutionError`.

- [X] **Step 2: Write global-rewrite tests for topology/image preservation**

```python
def test_global_rewrite_preserves_ids_order_and_images() -> None:
    llm = Mock()
    llm.generate_structured.return_value = GlobalRewriteResolution(
        language="en",
        query_text="A woman talks to a man, then takes a plate.",
        entities=[],
        events=[
            GlobalRewriteEvent(event_id="E1", text="A woman talks to a man.", bindings=[]),
            GlobalRewriteEvent(event_id="E2", text="The woman takes a plate.", bindings=[]),
        ],
    )
    base = _base_multimodal_intent()
    rewriter = KISGlobalRewriter(llm)
    rewritten = rewriter.rewrite(base, "Resolve pronouns globally")

    assert [event.id for event in rewritten.events] == ["E1", "E2"]
    assert [event.images for event in rewritten.events] == [
        base.events[0].images,
        base.events[1].images,
    ]
```

Add a test where model tries to return a different event count; expect `KISResolutionError`.

- [X] **Step 3: Run the new resolver tests and verify import failures**

```bash
PYTHONPATH=src python -m pytest tests/hcmai/kis/test_scoped_resolver.py tests/hcmai/kis/test_rewriter.py -q
```

Expected: FAIL because both services are absent.

- [X] **Step 4: Define narrow structured output schemas**

In domain modules define:

```python
class ScopedResolvedEvent(BaseModel):
    event_id: EventId
    text: NonBlank
    bindings: list[KISEntityBinding] = Field(default_factory=list)


class ScopedResolutionBatch(BaseModel):
    language: Literal["vi", "en"]
    events: list[ScopedResolvedEvent] = Field(min_length=1)


class GlobalRewriteEvent(BaseModel):
    event_id: EventId
    text: NonBlank | None = None
    bindings: list[KISEntityBinding] = Field(default_factory=list)


class GlobalRewriteResolution(BaseModel):
    language: Literal["vi", "en"] | None
    query_text: NonBlank | None
    entities: list[KISEntity] = Field(default_factory=list)
    events: list[GlobalRewriteEvent] = Field(min_length=1)
```

The scoped service validates exact target-ID equality. Every returned binding must include both `entity_id` and non-blank `role`; the server must not invent roles. When `base` exists, every returned binding entity ID must exist in `base.entities`; it never mutates entity descriptions. Language validation is:

```text
base.language in {vi,en} -> resolved.language must equal base.language
base.language is None     -> resolved.language becomes the new intent language for the first textual update
```

When `base is None` for an initial explicit `E1..Ek` batch, scoped bindings must be empty because there is no canonical entity table yet.

- [X] **Step 5: Apply scoped results server-side without touching unrelated events**

Provide a pure helper:

```python
def apply_scoped_resolutions(
    base: KISIntent | None,
    resolved: ScopedResolutionBatch,
    *,
    revision: int,
) -> KISIntent:
```

When `base` exists, replace only named event text/bindings and preserve unrelated events/images. Validate every scoped binding against the existing entity table and retain the model-supplied role verbatim. When `base.language is None` and this is the first textual update, adopt `resolved.language`; otherwise require language equality. When `base is None`, create the initial contiguous events from the scoped batch with an empty global entity table and empty bindings. For a new event, create it with no images unless deterministic patch data later adds images. Rebuild the complete adjacent edge chain and set canonical `query_text` with a pure helper that joins non-empty event texts in event order; if every event is image-only, `query_text` is `None`.

Add a dedicated test: start from an image-only `E1` with `language=None`, apply `E1: the woman is holding a plate`, have the scoped model return `language="en"`, and assert the new intent is text+image with `language="en"`.

- [X] **Step 6: Implement global rewrite with fixed topology**

Use a structured response containing `language`, optional `query_text`, global entities, and exactly one text/binding resolution for every existing event ID. The model may rewrite only events that already contain text; image-only events remain `text=None` because the LLM is not given image pixels and must not invent image semantics. After validation, copy image arrays from the base intent by event ID and derive the same adjacent chain. Set revision only from the caller-provided next revision.

- [X] **Step 7: Run scoped/global resolver tests**

```bash
PYTHONPATH=src python -m pytest tests/hcmai/kis/test_scoped_resolver.py tests/hcmai/kis/test_rewriter.py tests/hcmai/kis/test_resolver.py -q
```

Expected: PASS.

- [X] **Step 8: Commit semantic operation services**

```bash
git add src/hcmai/kis tests/hcmai/kis

git commit -m "feat: add scoped and global KIS semantic resolution"
```

---

### Task 7: Add Content-Addressed KIS Image Assets and Upload API

**Files:**

- Create: `src/hcmai/kis/assets.py`
- Modify: `src/hcmai/api/contracts/kis.py`
- Modify: `src/hcmai/api/routers/kis.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Create: `tests/hcmai/kis/test_assets.py`
- Modify: `tests/api/test_kis_router.py`

**Interfaces:**

- Consumes: existing image byte/pixel limits from `ApiConfig` and the safe static-image decoding rules in `ImageSearchService`.
- Produces: `KISImageAssetStore.put(payload, content_type) -> KISImageRef`, `open(asset_id) -> PIL.Image.Image`, and `read(asset_id) -> tuple[bytes, str]` for transport-safe thumbnail retrieval.
- Produces: `POST /api/v1/kis/assets/images` returning `KISImageRef` and `GET /api/v1/kis/assets/images/{asset_id}` returning the validated original bytes.

- [X] **Step 1: Write asset-store tests for validation, dedupe, and stable IDs**

```python
from io import BytesIO
from PIL import Image


def png_bytes(width: int, height: int) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(stream, format="PNG")
    return stream.getvalue()


def test_asset_store_deduplicates_identical_payloads(tmp_path) -> None:
    store = KISImageAssetStore(tmp_path, max_upload_bytes=1024 * 1024, max_pixels=10_000)
    payload = png_bytes(width=10, height=10)

    first = store.put(payload, "image/png")
    second = store.put(payload, "image/png")

    assert first == second
    assert first.asset_id.startswith("sha256:")
    assert len(list(tmp_path.iterdir())) == 1
```

Also test unsupported MIME, empty payload, byte limit, pixel limit, animated image rejection, `open()` returning RGB, and `read()` returning the exact validated bytes plus normalized content type for a known asset. Unknown asset IDs must fail without path traversal.

- [X] **Step 2: Run asset tests and verify failure**

```bash
PYTHONPATH=src python -m pytest tests/hcmai/kis/test_assets.py -q
```

Expected: import failure.

- [X] **Step 3: Implement the store using canonical bytes hash and safe decoding**

Use:

```python
digest = hashlib.sha256(payload).hexdigest()
asset_id = f"sha256:{digest}"
```

Validate first, detect the actual static format (`JPEG`, `PNG`, or `WEBP`), then atomically write original validated bytes as `<sha256>.<normalized-extension>` using a temporary file + `Path.replace()`. `asset_id` remains `sha256:<digest>`. Add `ref(asset_id) -> KISImageRef` that resolves the one stored extension back to the normalized content type, and `open(asset_id)` that re-validates and returns an RGB copy detached from the file handle. This lets event patches carry only asset IDs while canonical events retain full `KISImageRef` values.

Add `ApiConfig.kis_query_asset_dir` with a repository-relative default such as `data/query-assets` and resolve it through the existing repository-path helper in setup. `read(asset_id)` resolves only canonical `sha256:<hex>` IDs, reads the stored original bytes, and returns `(payload, normalized_content_type)`; it must not expose filesystem paths.

- [X] **Step 4: Wire the asset store into `SearchService` and add the upload contract/endpoint**

Add `kis_image_assets: KISImageAssetStore | None` to the `SearchService` constructor and assign `self.kis_image_assets`. In setup, create it from `ApiConfig.kis_query_asset_dir`, `image_max_upload_bytes`, and `image_max_pixels`; record a startup message and leave it `None` only when storage initialization fails.

Response model is exactly `KISImageRef`. Router endpoint:

```python
@router.post("/api/v1/kis/assets/images", response_model=KISImageRef)
async def upload_kis_image(file: UploadFile = File(...)) -> KISImageRef:
    payload = await file.read(service.api_config.image_max_upload_bytes + 1)
    return await run_in_threadpool(
        service.kis_image_assets.put,
        payload,
        file.content_type,
    )
```

Add the matching display route:

```python
@router.get("/api/v1/kis/assets/images/{asset_id}")
async def get_kis_image(asset_id: str) -> Response:
    payload, content_type = await run_in_threadpool(service.kis_image_assets.read, asset_id)
    return Response(
        content=payload,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
```

Map invalid upload to 422, unknown asset to 404, and asset-store unavailable to 503. Add an upload-then-GET test that asserts byte equality, content type, and immutable cache header. This route is the canonical source for event-card thumbnails after replay/reload; browser object URLs are temporary preview-only state.

- [X] **Step 5: Run asset/router tests**

```bash
PYTHONPATH=src python -m pytest tests/hcmai/kis/test_assets.py tests/api/test_kis_router.py -q
```

Expected: PASS.

- [X] **Step 6: Commit query image asset support**

```bash
git add src/hcmai/kis/assets.py src/hcmai/api src/hcmai/orchestration src/hcmai/common/config.py tests/hcmai/kis/test_assets.py tests/api/test_kis_router.py

git commit -m "feat: add reusable KIS query image assets"
```

---

### Task 8: Replace Clue-List API With Stateless Semantic Operations

**Files:**

- Modify: `src/hcmai/api/contracts/kis.py`
- Modify: `src/hcmai/api/routers/kis.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `tests/api/test_kis_contracts.py`
- Modify: `tests/api/test_kis_router.py`
- Rewrite: `tests/orchestration/test_kis_revision.py`

**Interfaces:**

- Consumes: Task 5 parser/models, Task 6 semantic services, Task 7 image refs, Task 2 retrieval execution.
- Produces: discriminated `KISSearchRequest` with `base_intent`, `expected_revision`, and one operation.
- Produces: `KISOperationSummary(kind, affected_event_ids)` in `KISSearchResponse`.

- [X] **Step 1: Write failing contract tests for all operation types**

Define request examples:

```python
def base_intent() -> KISIntent:
    return KISIntent(
        revision=2,
        language="en",
        query_text="woman enters, then talks",
        entities=[],
        events=[
            KISEvent(id="E1", text="A woman enters.", images=[], bindings=[]),
            KISEvent(id="E2", text="The woman talks.", images=[], bindings=[]),
        ],
        temporal_edges=[KISTemporalEdge(source="E1", target="E2", relation="before")],
    )


base = base_intent()
initial = KISSearchRequest.model_validate({
    "base_intent": None,
    "expected_revision": 0,
    "operation": {"kind": "initial_resolve", "text": "woman enters kitchen", "image_refs": []},
    "use_dense": True,
    "use_bm25": True,
    "top_k": 20,
})

search_only = KISSearchRequest.model_validate({
    "base_intent": base.model_dump(),
    "expected_revision": base.revision,
    "operation": {"kind": "search_only"},
    "use_dense": True,
    "use_bm25": False,
    "top_k": 100,
})
```

Add patch/global variants and reject base-intent/revision mismatches.

- [X] **Step 2: Run contract/orchestration tests and verify old clue-list assumptions fail**

```bash
PYTHONPATH=src python -m pytest tests/api/test_kis_contracts.py tests/orchestration/test_kis_revision.py -q
```

Expected: FAIL because `inputs`/`previous_revision` still define the API.

- [X] **Step 3: Replace the old revision/clue contract names and implement discriminated operation models**

Rename `KISRevisionSearchRequest` -> `KISSearchRequest`, `KISRevisionSearchResponse` -> `KISSearchResponse`, and `SearchService.search_kis_revision()` -> `SearchService.search_kis()`. Keep the published route path `/api/v1/kis/search`; the payload contract changes in place. Update the KIS router to call `service.search_kis(request)`.

Use Pydantic discriminators:

```python
class EventPatch(BaseModel):
    event_id: EventId
    instruction: str | None = None
    add_image_ids: list[str] = Field(default_factory=list)
    remove_image_ids: list[str] = Field(default_factory=list)


class InitialResolveOperation(BaseModel):
    kind: Literal["initial_resolve"]
    text: NonBlank | None = None
    image_refs: list[KISImageRef] = Field(default_factory=list)
    patches: list[EventPatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_initial_route(self) -> Self:
        natural_route = self.text is not None or bool(self.image_refs)
        explicit_route = bool(self.patches)
        if natural_route == explicit_route:
            raise ValueError("initial_resolve requires exactly one natural/image route or explicit E# batch")
        return self


class PatchEventsOperation(BaseModel):
    kind: Literal["patch_events"]
    patches: list[EventPatch] = Field(min_length=1)


class GlobalRewriteOperation(BaseModel):
    kind: Literal["global_rewrite"]
    instruction: NonBlank


class SearchOnlyOperation(BaseModel):
    kind: Literal["search_only"]
```

The request operation type is exactly:

```python
KISOperation = Annotated[
    InitialResolveOperation
    | PatchEventsOperation
    | GlobalRewriteOperation
    | SearchOnlyOperation,
    Field(discriminator="kind"),
]
```

- [X] **Step 4: Implement pure operation application in orchestration**

Add a private method or focused helper with this exact semantic behavior:

```python
def _resolve_operation(self, request: KISSearchRequest) -> tuple[KISIntent, KISOperationSummary, float]:
```

Rules:

```text
initial_resolve + no base intent:
  text only -> full initial resolver, revision 1; decomposition is allowed
  image only -> deterministic image-only E1 with language/query_text=None, revision 1; no LLM call
  text + image -> scoped resolver for exactly E1, attach image deterministically, revision 1; no decomposition
  explicit patches E1..Ek -> scoped resolver with base=None only for patches containing text instructions; construct image-only patches deterministically; revision 1
patch_events:
  validate IDs/gaps/asset ownership before inference
  call scoped resolver only for patches with instruction
  apply add/remove images deterministically
  revision = base + 1
global_rewrite:
  call global rewriter
  revision = base + 1
search_only:
  reuse base intent byte-for-byte semantically
  revision unchanged
```

An image-only patch must not call any LLM service.

- [X] **Step 5: Implement operation summary and response**

```python
class KISOperationSummary(BaseModel):
    kind: Literal["initial_resolve", "patch_events", "global_rewrite", "search_only"]
    affected_event_ids: list[EventId] = Field(default_factory=list)
```

Response returns intent, summary, results, retrieval settings, latency, warnings. No raw input history and no prepared retrieval strings. In DRES/search logging, use `intent.query_text` when present; for a purely image-only intent use the fixed transport label `[image-only KIS]` without writing that label back into canonical semantic state.

- [X] **Step 6: Normalize revision errors before model/retrieval work**

If `base_intent is None`, require `expected_revision == 0` and only allow `initial_resolve`. For an explicit initial batch, require contiguous patch IDs starting at `E1`. If base exists, require exact equality to `base_intent.revision`; reject `initial_resolve`. Raise `RevisionConflictError` before calling LLM.

- [X] **Step 7: Run API/orchestration operation tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/api/test_kis_contracts.py \
  tests/api/test_kis_router.py \
  tests/orchestration/test_kis_revision.py -q
```

Expected: PASS.

- [X] **Step 8: Commit the stateless operation API**

```bash
git add src/hcmai/api/contracts/kis.py src/hcmai/api/routers/kis.py src/hcmai/orchestration tests/api tests/orchestration/test_kis_revision.py

git commit -m "feat: make KIS revisions explicit semantic operations"
```

---

### Task 9: Add Full-Corpus Image Evidence to the Existing Temporal Fusion Path

**Files:**

- Create: `src/hcmai/retrieval/evidence/image_query.py`
- Modify: `src/hcmai/retrieval/plan.py`
- Modify: `src/hcmai/retrieval/evidence/hybrid.py`
- Modify: `src/hcmai/retrieval/evidence/fusion.py`
- Modify: `src/hcmai/common/config.py:366-397`
- Modify: `configs/baseline.yaml:52-60`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/orchestration/workflows/temporal_exploration.py`
- Modify: `src/hcmai/api/contracts/exploration.py`
- Modify: `src/hcmai/api/contracts/kis.py`
- Modify: `src/hcmai/orchestration/retrieval_setup.py:100-215`
- Modify: `src/hcmai/orchestration/workflows/temporal_search.py`
- Modify: `src/hcmai/orchestration/workflows/kis.py`
- Create: `tests/retrieval/evidence/test_image_query.py`
- Modify: `tests/orchestration/workflows/test_kis_pipeline.py`
- Modify: `tests/orchestration/test_pipeline.py`
- Modify: `tests/orchestration/test_kis_revision.py`
- Modify: `tests/orchestration/workflows/test_temporal_exploration.py`
- Modify: `tests/api/test_exploration_router.py`

**Interfaces:**

- Consumes: `KISRetrievalPlan.image_refs`, `KISImageAssetStore.open()`, existing visual index and `ImageEmbeddingAdapter`.
- Produces: `TemporalScoreComponent(name="visual_image", raw_scores=[event_count, frame_count])`.
- Produces: multimodal KIS execution through the same fusion and DP decoder.

- [X] **Step 1: Write image-evidence scorer tests including MAX pooling**

```python
import numpy as np
from PIL import Image


class FakeVisualIndex:
    frame_ids = np.asarray(["f1", "f2", "f3"])
    _vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)

    def score_subset(self, query_vectors, positions, chunk_size):
        del chunk_size
        return np.asarray(query_vectors, dtype=np.float32) @ self._vectors[positions].T


class FakeAssetStore:
    def open(self, asset_id):
        image = Image.new("RGB", (1, 1))
        image.info["asset_id"] = asset_id
        return image


class FakeImageEncoder:
    def encode_images(self, images, stats=None):
        del stats
        vectors = {
            "sha256:a": [1.0, 0.0],
            "sha256:b": [0.0, 1.0],
        }
        return np.asarray([vectors[image.info["asset_id"]] for image in images], dtype=np.float32)


def test_image_query_scorer_max_pools_multiple_exemplars_per_event() -> None:
    visual_index = FakeVisualIndex()
    scorer = ImageQueryTemporalScorer(
        visual_index=visual_index,
        image_encoder=FakeImageEncoder(),
        asset_store=FakeAssetStore(),
        chunk_size=128,
    )
    component = scorer.score_events([
        (KISImageRef(asset_id="sha256:a", content_type="image/png"),
         KISImageRef(asset_id="sha256:b", content_type="image/png")),
        (),
    ])

    scores_a = np.asarray([[1.0, 0.0]], dtype=np.float32) @ visual_index._vectors.T
    scores_b = np.asarray([[0.0, 1.0]], dtype=np.float32) @ visual_index._vectors.T
    assert component.name == "visual_image"
    assert component.raw_scores.shape == (2, 3)
    np.testing.assert_allclose(component.raw_scores[0], np.maximum(scores_a[0], scores_b[0]))
    np.testing.assert_allclose(component.raw_scores[1], np.zeros(3, dtype=np.float32))
```

Assert row 0 equals `maximum(score(image_a), score(image_b))`, row 1 is zero, and the `EventEvidenceProfile.has_images` flag prevents an image-less row from receiving image weight. Visual image evidence covers the canonical visual frame index, so it does not misuse the component `coverage` field to represent per-event availability.

- [X] **Step 2: Run evidence tests and verify failure**

```bash
PYTHONPATH=src python -m pytest tests/retrieval/evidence/test_image_query.py -q
```

Expected: import failure.

- [X] **Step 3: Implement `ImageQueryTemporalScorer`**

Use `asset_store.open()` for refs, batch `image_encoder.encode_images(images)`, score against all canonical visual positions with `visual_index.score_subset()`, then MAX pool vectors belonging to the same event. Return a component with a per-frame coverage mask and an event-availability vector exposed separately or encoded in the fusion input descriptor.

Do not call standalone top-K `ImageSearchService.search()`.

- [X] **Step 4: Extend `KISRetrievalPlan` and the SearchService plan builder for sparse text + image rows**

Change `KISRetrievalEvent.image_refs` from tuple of strings to `tuple[KISImageRef, ...]`. Extend `KISExplorationEventSeed` with:

```python
image_refs: list[KISImageRef] = Field(default_factory=list)
```

Add helpers:

```python
@property
def event_count(self) -> int:
    return len(self.events)

@property
def image_ref_rows(self) -> tuple[tuple[KISImageRef, ...], ...]:
    return tuple(event.image_refs for event in self.events)
```

Text may be `None` for image-only events.

Replace the S0 all-text builder in `SearchService` with a modality-aware builder. It must:

```python
text_positions = [i for i, event in enumerate(intent.events) if event.text is not None]
text_values = tuple(intent.events[i].text for i in text_positions)
translated = translator.translate(text_values, intent.language)  # only when required
```

Restore translated values to their original event indices, leave text fields `None` for image-only rows, and copy `tuple(event.images)` into every `KISRetrievalEvent.image_refs`. Do not compact or reorder `plan.events`. Add a `SearchService.search_kis()`/orchestration test for `E1 text-only, E2 image-only, E3 text+image` that asserts event IDs/order, translated text at positions 0/2 only, and exact image refs at positions 1/2. The test must exercise the real plan builder rather than manually constructing `KISRetrievalPlan`.

- [X] **Step 5: Add `visual_image` to adaptive fusion without creating a new fusion algorithm**

Add a default base component weight for `visual_image` and structural availability routing. Update `configs/baseline.yaml` in the same step: because YAML supplies the entire `base_component_weights` dictionary, it must include a positive `visual_image` entry or production baseline fusion will assign image evidence zero mass. Use this exact additive baseline entry:

```yaml
visual_image: 0.35
```

Do not rescale the existing text/BM25 weights in this task: the fusion implementation already normalizes by effective local denominator, so adding the component preserves text-only behavior while giving text+image events an explicit image prior. Do not rely on the Python default to merge into YAML.

Add a regression test that loads the actual baseline configuration, asserts `base_component_weights["visual_image"] == 0.35`, constructs an image-only event with a non-flat `visual_image` component, runs fusion, and asserts the resulting row is non-zero and preserves the expected ranking.

Keep the existing text-only `fuse()` API for TRAKE by adapting it internally, and add a profile-aware fusion entrypoint:

```python
@dataclass(frozen=True, slots=True)
class EventEvidenceProfile:
    original_text: str | None
    retrieval_text: str | None
    has_text: bool
    has_images: bool
```

Add:

```python
def fuse_profiles(
    self,
    *,
    profiles: Sequence[EventEvidenceProfile],
    bundle: TemporalScoreBundle,
) -> np.ndarray:
```

The existing `fuse(original_events=..., retrieval_events=..., bundle=...)` constructs profiles with `has_text=True, has_images=False` and delegates to `fuse_profiles()`. `visual_image` receives zero multiplier when `has_images` is false. Every text/BM25 component receives zero multiplier when `has_text` is false. Text cue boosts continue to apply only when text exists. Preserve existing calibration/confidence gating/local renormalization.

- [X] **Step 6: Let `TemporalEvidenceScorer` merge text/BM25/image components before one fusion call**

Add a plan-oriented method:

```python
def score_plan(
    self,
    plan: KISRetrievalPlan,
    *,
    image_component: TemporalScoreComponent | None,
    use_dense: bool,
    use_bm25: bool,
) -> list[VideoEventScores]:
```

Build `text_indices = [i for i, event in enumerate(plan.events) if event.canonical_text is not None]`. Run Dense/BM25 only on that compact text subset, then expand every returned component back to `[plan.event_count, frame_count]` with zero rows at image-only indices before merging `visual_image`. Use a focused helper:

```python
def expand_component_rows(
    component: TemporalScoreComponent,
    *,
    event_indices: Sequence[int],
    event_count: int,
) -> TemporalScoreComponent:
```

For image-only events, structural `has_text=False` makes the zero text rows carry zero effective weight, while `visual_image` supplies the event row. Existing legacy `score_events()` remains for TRAKE/older non-KIS callers during this plan.

- [X] **Step 7: Add `TemporalSearchService.search_plan()` and keep DP unchanged**

```python
def search_plan(
    self,
    plan: KISRetrievalPlan,
    *,
    image_component: TemporalScoreComponent | None,
    use_dense: bool,
    use_bm25: bool,
    top_k: int,
) -> TemporalSearchResult:
```

This method calls `evidence.score_plan()`, validates matrix shape by `plan.event_count`, then uses the same `rank_paths()` and materialization code as `search()`.

- [X] **Step 8: Wire KIS pipeline and temporal exploration through the same multimodal plan scoring**

`KISPipeline.execute()` resolves image refs to the image component, calls `search_plan()`, and materializes results. Add a test with:

```text
E1 text-only
E2 image-only
E3 text+image
```

Assert the final score matrix has three rows and aligned path order stays `E1,E2,E3`.

Then migrate `TemporalExploration.open()` from the S0 textual fallback to `TemporalSearchService.score_plan()` using the `KISExplorationSeed` supplied by the frontend. Reconstruct a typed `KISRetrievalPlan` from the seed, resolve image refs through the same image scorer, and select the requested video's cached scores. `ExplorationOpenRequest` must accept image-only seed rows; it must no longer require non-empty text for every event. Add an API/workflow test that opens exploration for an image-only event and obtains a valid branch/view.

- [X] **Step 9: Run multimodal retrieval, baseline-config, SearchService-builder, and exploration tests**

```bash
PYTHONPATH=src python -m pytest \
  tests/retrieval/evidence/test_image_query.py \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/orchestration/test_pipeline.py \
  tests/orchestration/test_kis_revision.py \
  tests/orchestration/workflows/test_temporal_exploration.py \
  tests/api/test_exploration_router.py -q
```

Expected: PASS.

- [X] **Step 10: Commit multimodal temporal evidence**

```bash
git add src/hcmai/retrieval src/hcmai/orchestration src/hcmai/api/contracts/exploration.py src/hcmai/api/contracts/kis.py src/hcmai/common/config.py configs/baseline.yaml tests/retrieval tests/orchestration tests/api/test_exploration_router.py

git commit -m "feat: score KIS image events in temporal fusion"
```

---

### Task 10: Build the Frontend Command Parser and Stateless Multimodal Session Model

**Files:**

- Create: `frontend/src/features/kis/parser.js`
- Create: `frontend/src/features/kis/parser.test.js`
- Rewrite: `frontend/src/features/kis/session.js`
- Rewrite: `frontend/src/features/kis/session.test.js`
- Modify: `frontend/src/api/kis.js`
- Modify: `frontend/src/api/kis.test.js`

**Interfaces:**

- Consumes: backend operation shapes from Task 8.
- Produces: `parseComposerDraft(draft, baseIntent) -> preview` for deterministic UI feedback.
- Produces: session state based on `currentIntent`, staged image refs, draft, pending operation, and replay/live mode; no `committedInputs`.

- [X] **Step 1: Write parser tests mirroring backend grammar**

```javascript
const BASE_INTENT = {
  revision: 3,
  language: 'en',
  query_text: 'woman enters, talks, then leaves',
  entities: [],
  events: [
    { id: 'E1', text: 'woman enters', images: [], bindings: [] },
    { id: 'E2', text: 'woman talks', images: [], bindings: [] },
    { id: 'E3', text: 'woman leaves', images: [], bindings: [] },
  ],
  temporal_edges: [
    { source: 'E1', target: 'E2', relation: 'before' },
    { source: 'E2', target: 'E3', relation: 'before' },
  ],
};

test('previews a scoped batch', () => {
  expect(parseComposerDraft('E2: chef wears black\nE4: takes a plate', BASE_INTENT)).toEqual({
    kind: 'patch_events',
    affectedEventIds: ['E2', 'E4'],
    error: null,
  });
});


test('reports an illegal event gap before submit', () => {
  expect(parseComposerDraft('E5: new event', BASE_INTENT).error).toMatch(/missing E4/i);
});
```

Also cover `/llm-rewrite`, initial natural text, initial explicit `E1..Ek` mapped to `initial_resolve` with patches, duplicate IDs, and unscoped progressive text.

- [X] **Step 2: Write session-state tests for operation/revision semantics**

State target:

```javascript
{
  draft: '',
  currentIntent: null,
  revision: 0,
  stagedImages: {},
  pendingOperation: null,
  isSearching: false,
  error: null,
  mode: 'live',
}
```

Test semantic success increments according to response intent, `search_only` keeps revision, failure preserves draft/staged images/current intent, and replay mode cannot create a live search payload.

- [X] **Step 3: Run parser/session tests and verify old clue-list model fails**

```bash
cd frontend
CI=true npm test -- --watchAll=false src/features/kis/parser.test.js src/features/kis/session.test.js
```

Expected: FAIL because parser is absent and session still stores `committedInputs`.

- [X] **Step 4: Implement frontend parser as a preview only**

The parser mirrors structural rules for immediate feedback but never becomes authoritative. Return preview kinds `initial_resolve`, `patch_events`, `global_rewrite`, `invalid` and affected IDs. Backend errors still surface on submit.

- [X] **Step 5: Rewrite session payload creation around `base_intent + operation`**

Expose:

```javascript
export const prepareSemanticRequest = (state, preview) => ({ nextState, requestPayload });
export const prepareSearchOnlyRequest = (state) => ({ nextState, requestPayload });
export const commitSearchSuccess = (state, response) => nextState;
export const commitSearchFailure = (state, error) => nextState;
```

No payload contains `inputs` or derives expected revision from clue count.

- [X] **Step 6: Update `searchKis()` API client**

Signature:

```javascript
export const searchKis = async ({
  baseIntent,
  expectedRevision,
  operation,
  useDense = true,
  useBm25 = true,
  topK = 20,
  userId,
  signal,
}) => requestJson('/api/v1/kis/search', {
  method: 'POST',
  body: {
    base_intent: baseIntent,
    expected_revision: expectedRevision,
    operation,
    use_dense: useDense,
    use_bm25: useBm25,
    top_k: topK,
  },
  signal,
  headers: userId?.trim() ? { 'X-VBS-User-ID': userId.trim() } : {},
});
```

Validate response requires `intent`, `operation_summary`, `results`, and latency.

- [X] **Step 7: Run frontend parser/session/API tests**

```bash
cd frontend
CI=true npm test -- --watchAll=false \
  src/features/kis/parser.test.js \
  src/features/kis/session.test.js \
  src/api/kis.test.js
```

Expected: PASS.

- [X] **Step 8: Commit frontend operation/session contracts**

```bash
git add frontend/src/features/kis frontend/src/api/kis.js frontend/src/api/kis.test.js

git commit -m "feat: model KIS frontend state as semantic operations"
```

---

### Task 11: Build the Unified Multimodal Query Composer and Remove the Image-Search Bypass

**Files:**

- Create: `frontend/src/features/kis/components/IntentSummary.jsx`
- Create: `frontend/src/features/kis/components/EventList.jsx`
- Create: `frontend/src/features/kis/components/EventCard.jsx`
- Create: `frontend/src/features/kis/components/QueryComposer.jsx`
- Modify: `frontend/src/features/kis/components/KisPanel.jsx`
- Modify: `frontend/src/features/kis/components/KisPanel.test.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx`
- Modify: `frontend/src/features/search/components/SearchWorkspace.test.jsx`
- Modify: `frontend/src/styles/layout.css`
- Modify: `frontend/src/styles/controls.css`
- Modify: `frontend/src/styles/responsive.css`
- Modify: `frontend/src/api/kis.js`
- Delete after caller migration: `frontend/src/features/search/components/ImageSearchWorkspace.jsx`
- Delete after caller migration: `frontend/src/features/search/components/ImageSearchWorkspace.test.jsx`

**Interfaces:**

- Consumes: Task 10 session/parser, `POST /api/v1/kis/assets/images`, `GET /api/v1/kis/assets/images/{asset_id}`, Task 8 search API.
- `KisPanel` receives `sessionState`, `onDraftChange`, `onSubmit`, `onAttachImage(file, eventId | null)`, and `onRemoveImage(eventId, assetId)`.
- `QueryComposer` owns the temporary event-target chooser when an attached image has no explicit `E#` scope; it does not mutate canonical intent before submit succeeds.
- Produces: one keyboard-first composer for initial natural text, `E#:` patches, `/llm-rewrite`, event-targeted image attach/remove, `+ Event`, and search-only reruns.

- [X] **Step 1: Write component tests for event cards, dynamic labels, and explicit image targeting**

```javascript
const STATE_WITH_E1_E2 = {
  draft: '',
  revision: 2,
  currentIntent: {
    revision: 2,
    language: 'en',
    query_text: 'woman enters, then chef appears',
    entities: [],
    events: [
      { id: 'E1', text: 'woman enters', images: [], bindings: [] },
      { id: 'E2', text: 'chef appears', images: [], bindings: [] },
    ],
    temporal_edges: [{ source: 'E1', target: 'E2', relation: 'before' }],
  },
  stagedImages: {},
  pendingOperation: null,
  isSearching: false,
  error: null,
  mode: 'live',
};

test('renders committed multimodal events and prefills the next event', () => {
  const onDraftChange = jest.fn();
  render(<KisPanel sessionState={STATE_WITH_E1_E2} onDraftChange={onDraftChange} />);
  expect(screen.getByText('E1')).toBeTruthy();
  expect(screen.getByText('E2')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: /add event/i }));
  expect(onDraftChange).toHaveBeenCalledWith('E3: ');
});


test('asks for an event target when an image is attached without E# scope', async () => {
  const file = new File(['png-bytes'], 'chef.png', { type: 'image/png' });
  const onAttachImage = jest.fn();
  render(
    <KisPanel
      sessionState={STATE_WITH_E1_E2}
      onDraftChange={jest.fn()}
      onAttachImage={onAttachImage}
    />,
  );
  fireEvent.change(screen.getByLabelText(/attach image/i), { target: { files: [file] } });
  expect(screen.getByText(/attach image to/i)).toBeTruthy();
  expect(screen.getByRole('button', { name: 'E1' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'E2' })).toBeTruthy();
  expect(onAttachImage).not.toHaveBeenCalled();
});
```

Add tests for initial image-only => E1, initial text+image => one E1, `E2:` + pasted image => E2, remove-image patch, and `/llm-rewrite` submit label `Rewrite`.

- [X] **Step 2: Run component/workspace tests and verify current separate image path fails**

```bash
cd frontend
CI=true npm test -- --watchAll=false \
  src/features/kis/components/KisPanel.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx
```

Expected: FAIL because `selectedImageFile` still bypasses KIS.

- [X] **Step 3: Implement the focused display components and persisted-thumbnail URL contract**

`IntentSummary` shows only committed canonical query + revision. `EventList` maps stable event IDs to `EventCard`. `EventCard` shows text if present, image thumbnails, and actions `Edit`, `Add image`, `Remove image`; entities/bindings stay inside a collapsed `Semantic details` block in `KisPanel`.

Add a frontend helper in `frontend/src/api/kis.js`:

```javascript
export const kisImageAssetUrl = (assetId) => (
  `/api/v1/kis/assets/images/${encodeURIComponent(assetId)}`
);
```

Committed/replayed `KISImageRef` thumbnails use this URL. Browser `URL.createObjectURL()` may be used only for a temporary pre-upload preview and must be revoked after upload/cancel; it is never persisted in semantic or history state. Add an `EventCard`/replay test that constructs a saved `KISImageRef` and asserts the rendered image `src` points to `kisImageAssetUrl(asset_id)`, proving thumbnails survive reload without the original `File` object.

- [X] **Step 4: Implement `QueryComposer` as the sole textual input surface**

Requirements:

```text
Enter -> submit
Shift+Enter -> newline
+ Event -> prefill E(n+1): and focus
Edit E2 -> prefill "E2: " and focus
parser preview -> "Will update: E2, E4" / "Global rewrite" / structural error
Search/Update/Rewrite dynamic submit copy
```

Do not create inline permanent text fields for every event.

- [X] **Step 5: Add image upload client and staged attachment flow**

Import `requestFormData` from `frontend/src/api/client.js`, then add:

```javascript
export const uploadKisImage = async ({ imageFile, signal }) => {
  const body = new FormData();
  body.append('file', imageFile);
  return requestFormData('/api/v1/kis/assets/images', body, { method: 'POST', signal });
};
```

Upload once, store returned ref in staged event patch state, and reuse the ref in subsequent revisions. Do not resend old files.

- [X] **Step 6: Remove `selectedImageFile` / `submitImageSearch()` branching from `SearchWorkspace`**

All KIS submits call `searchKis()`. `searchFramesByImage()` may remain in the generic search API only if another confirmed non-KIS consumer exists; `SearchWorkspace` must not branch around `KISIntent` for image input.

- [X] **Step 7: Add SearchOnly rerun when only retrieval controls change**

When an intent exists and user changes top-k/Dense/BM25 without a semantic draft, construct `operation: { kind: 'search_only' }` with the same base intent/revision. Do not increase revision or add history semantics indicating a semantic change.

- [X] **Step 8: Preserve transactional UI across all operation types**

While initial/patch/rewrite/search-only is pending, keep committed event cards and result grid interactive. On failure keep draft and staged image refs. On success atomically replace intent/results and clear only staged data consumed by the accepted operation.

- [X] **Step 9: Run unified composer tests**

```bash
cd frontend
CI=true npm test -- --watchAll=false \
  src/features/kis/components/KisPanel.test.jsx \
  src/features/search/components/SearchWorkspace.test.jsx \
  src/features/workspace/components/ReplayResults.test.jsx \
  src/api/kis.test.js
```

Expected: PASS.

- [X] **Step 10: Commit the unified multimodal Query Composer**

```bash
git add frontend/src/features/kis frontend/src/features/search/components frontend/src/api/kis.js frontend/src/styles

git commit -m "feat: unify text and image KIS in the query composer"
```

---

### Task 12: Persist Operation-Level Research Metadata and Interaction Events

**Files:**

- Modify: `src/hcmai/api/contracts/history.py`
- Modify: `src/hcmai/api/routers/history.py`
- Modify: `src/hcmai/api/history.py`
- Modify: `frontend/src/api/history.js`
- Modify: `frontend/src/api/history.test.js`
- Modify: `frontend/src/features/search/components/SearchWorkspace.jsx`
- Modify: `frontend/src/features/workspace/queryHistory.js`
- Modify: `frontend/src/features/workspace/queryHistory.test.js`
- Create: `tests/api/test_history.py`

**Interfaces:**

- Consumes: `KISOperationSummary`, canonical intent revision, history snapshots, frame/video activity, and submission actions.
- Produces: replay metadata plus append-only `result_open` and `submission` interaction events keyed by query ID. Existing `viewed_frame_ids` remains the canonical frame-inspection log. This same event boundary can be extended with EventTrail-specific event types in the next design cycle.

- [X] **Step 1: Write history-contract tests for operation metadata**

Extend `QueryHistoryCreate`/record with a structured metadata object:

```python
class QueryOperationMetadata(BaseModel):
    semantic_revision: int = Field(ge=0)
    operation_kind: str
    affected_event_ids: list[str] = Field(default_factory=list)
    image_added: list[str] = Field(default_factory=list)
    image_removed: list[str] = Field(default_factory=list)
    search_only: bool = False
```

Add a second contract for append-only interaction events:

```python
class QueryInteractionEventCreate(BaseModel):
    event_type: Literal["result_open", "submission"]
    semantic_revision: int = Field(ge=0)
    event_id: str | None = None
    frame_id: str | None = None
    video_id: str | None = None
    timestamp_ms: int | None = Field(default=None, ge=0)
```

Test round-trip persistence and migration from existing v3 rows with metadata defaulting to an empty/legacy value. Test interaction events retain insertion order for one query.

- [X] **Step 2: Run history tests and verify schema failure**

```bash
PYTHONPATH=src python -m pytest tests/api -q -k history
```

Expected: FAIL because DB schema has no operation metadata column.

- [X] **Step 3: Migrate workspace DB to the next version without losing query history**

Increment `_DATABASE_VERSION`. Add `operation_metadata_json TEXT NOT NULL DEFAULT '{}'` to `query_history`, and create `query_interaction_events(query_id, sequence_id, event_type, semantic_revision, event_id, frame_id, video_id, timestamp_ms, created_at)` with a foreign key to `query_history`. Migrate old rows preserving `query_id`, user, query text, result snapshot, viewed frames, and created time. Keep `viewed_frame_ids_json` as the frame-inspection record rather than duplicating it into the interaction table.

- [X] **Step 4: Add an append-only interaction-event endpoint**

Expose:

```text
POST /api/v1/query-history/{query_id}/events
```

The store assigns monotonically increasing `sequence_id` per query. Missing query IDs return 404; invalid event payloads return 422.

- [X] **Step 5: Persist semantic operation summary only after successful live commit**

Frontend history create call sends operation metadata from the response plus deterministic image add/remove IDs known from the submitted operation. `SearchOnly` stores `search_only: true` and the same semantic revision.

- [X] **Step 6: Emit current S1 research events through the per-query history queue**

Add `recordQueryInteraction()` in `frontend/src/api/history.js`. Reuse the per-query queue established in Task 3: `createQueryHistory()` remains the first write for a new query, and `markFrameViewed()` / `recordQueryInteraction()` are serialized behind it. `SearchWorkspace.openCanonicalFrame()` keeps recording the viewed frame and additionally enqueues `result_open`; wrap `onOpenSubmission` so a submission initiated from an active query enqueues `submission` with the current semantic revision before delegating to the existing submission callback.

These writes are best-effort and must not block search, inspection, or submission. Add a test where the user opens a result before history creation resolves: neither `markFrameViewed` nor `recordQueryInteraction` may hit the API until creation succeeds, and their order must remain deterministic. Also assert that completion of an older query's queue cannot change the newer active session.

- [X] **Step 7: Keep replay semantic state inside snapshot, not draft reconstruction**

History snapshot stores the canonical intent required for read-only replay. Replay UI may display `Rev N · Updated E2` from operation metadata, but must not regenerate live composer state automatically.

- [X] **Step 8: Run backend/frontend history tests**

```bash
PYTHONPATH=src python -m pytest tests/api -q -k history
cd frontend
CI=true npm test -- --watchAll=false src/api/history.test.js src/features/workspace/queryHistory.test.js
```

Expected: PASS.

- [X] **Step 9: Commit research-ready operation logging**

```bash
git add src/hcmai/api frontend/src/api/history.js frontend/src/api/history.test.js frontend/src/features/search/components/SearchWorkspace.jsx frontend/src/features/workspace tests/api

git commit -m "feat: persist KIS operation metadata for replay and evaluation"
```

---

### Task 13: S0/S1 End-to-End Acceptance, Dead-Code Gate, and Generated-Artifact Cleanup

**Files:**

- Modify: `tests/test_kis_acceptance_smoke.py`
- Modify: `src/hcmai/README.md`
- Modify: `src/hcmai/orchestration/README.md`
- Modify: frontend docs/help text that still describes clue-history/separate image search
- Modify/Create: `.gitignore`
- Delete tracked/generated artifacts: `**/__pycache__/`, `*.pyc`, `.pytest_cache/`, `frontend/build/` if present in source tracking/snapshot
- Remove any remaining dead source identified by repository-wide scans

**Interfaces:**

- Consumes: all prior tasks.
- Produces: verified single production path ready for the separate EventTrail design cycle.

- [ ] **Step 1: Rewrite the acceptance smoke test around semantic operations**

The test must execute this sequence through the real orchestration boundary with deterministic inference/retrieval doubles:

```text
1. InitialResolve:
   "A woman is standing in a kitchen, then talks to a man."
   -> Rev 1, multiple canonical events allowed

2. PatchEvents:
   E2: the man is actually a chef wearing black
   -> Rev 2, only E2 text/bindings change

3. Upload image + PatchEvents:
   attach image to E2 with no text instruction
   -> Rev 3, LLM call count unchanged for this operation

4. Start a separate image-only base intent, then patch its existing E1 with text:
   image-only E1 has language=None
   E1: the woman is holding a plate
   -> text+image E1 adopts returned language="en" without dropping the image

5. Append on the main flow:
   E3: then she takes a white plate
   -> Rev 4

6. GlobalRewrite:
   resolve all references explicitly
   -> Rev 5, same IDs/order/image ownership

7. SearchOnly with top_k/source change
   -> still Rev 5

8. Force semantic provider failure on a later E2 patch
   -> response fails, previously committed Rev 5 remains the frontend base snapshot
```

Assert DP receives event rows in exact canonical order on each successful retrieval. Also assert the SearchService-built retrieval plan translates only text-bearing rows and preserves image refs, the loaded `configs/baseline.yaml` gives `visual_image` positive effective weight, and an image-only result can open temporal exploration from `exploration_seed`.

- [ ] **Step 2: Run focused backend acceptance and full backend suite**

```bash
PYTHONPATH=src python -m pytest tests/test_kis_acceptance_smoke.py -q
PYTHONPATH=src python -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 3: Run the full frontend test suite and production build**

```bash
cd frontend
CI=true npm test -- --watchAll=false
npm run build
```

Expected: all tests PASS and production build succeeds.

- [ ] **Step 4: Run compile/import validation**

```bash
python -m compileall -q src llm/server
PYTHONPATH=src python - <<'PY'
from hcmai.kis.models import KISIntent, KISEvent, KISImageRef
from hcmai.kis.parser import parse_kis_command
from hcmai.retrieval.plan import KISRetrievalPlan
from hcmai.retrieval.translation.service import EventTranslator
print("imports-ok")
PY
```

Expected output: `imports-ok`.

- [ ] **Step 5: Run the legacy-reference hard gate**

Run:

```bash
rg -n \
  "KISIntent\.inputs|revision\s*==\s*len\(inputs\)|committedInputs|KISRevisionSearch(Request|Response)|search_kis_revision|QueryPreparationService|generate_candidates|query-candidates|submitImageSearch|selectedImageFile" \
  src frontend tests llm \
  -g '!**/__pycache__/**'
```

Expected: no production references. `dense_events` / `bm25_events` are checked separately: they must not exist as top-level KIS response/frontend snapshot fields, but internal legacy non-KIS APIs may retain their own names until independently migrated. Search specifically for `response.dense_events`, `response.bm25_events`, and `liveKisSnapshotRef.*dense_events|bm25_caption_events` and require zero KIS-production matches. Test fixtures may mention removed names only when the test explicitly asserts absence; remove stale fixture usage rather than whitelisting it.

Also run:

```bash
rg -n "query_preparation" src frontend tests configs llm -g '!**/__pycache__/**'
```

Expected: no production query-preparation subsystem references.

- [ ] **Step 6: Clean generated artifacts and establish ignore rules**

`.gitignore` must include:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
frontend/build/
frontend/node_modules/
.env
```

Remove generated directories/files from the working tree snapshot before the final commit.

- [ ] **Step 7: Update docs to the final architecture**

Docs must describe this single product flow:

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

Explicitly state that EventTrail is the next design cycle, not part of S0/S1.

- [ ] **Step 8: Perform one manual browser acceptance run**

Use a running backend/inference service and verify:

```text
natural initial text search
E2 scoped update
+ Event -> append E(n+1)
image-only event attach
image-only -> first textual update preserves image and establishes language
text+image event update
/llm-rewrite
SearchOnly by changing retrieval options
open temporal exploration for a text event and an image-only event
reload/replay and verify persisted image thumbnail loads through the asset GET endpoint
click a result before history creation resolves and verify ordered history writes
failed semantic update preserves prior grid and draft
read-only replay does not populate live composer draft
```

Record failures as blocking defects; do not mark S0/S1 complete while any item above is broken.

- [ ] **Step 9: Commit the completion gate**

```bash
git add .gitignore docs src frontend tests configs

git commit -m "chore: complete multimodal progressive KIS stabilization"
```

---

## Final Verification Checklist

Before handing the codebase to the EventTrail design cycle, verify all of these statements are true:

- [ ] `/api/v1/query-candidates` and candidate-generation code are gone.
- [ ] `EventTranslator` is the only KIS translation owner and cache identity uses `LLMClient.model`.
- [ ] KIS product responses do not expose loose prepared Dense/BM25 strings; committed selected-video rescoring uses `exploration_seed`.
- [ ] `KISIntent` has no raw clue history and revision is independent of clue/event count.
- [ ] `KISEvent` supports text-only, image-only, and text+image evidence.
- [ ] `E#:` commands are parsed deterministically and only named events can change.
- [ ] Image-only patches make zero LLM calls.
- [ ] `/llm-rewrite` preserves IDs/order/cardinality/images.
- [ ] Image uploads are content-addressed, reused by opaque refs, and fetchable through the canonical asset GET route after replay/reload.
- [ ] SearchService plan construction translates only text-bearing rows, restores original positions, and preserves image refs.
- [ ] The loaded baseline configuration assigns positive effective weight to `visual_image`; image-only ranking is non-zero.
- [ ] Image evidence enters full-corpus temporal scoring and the existing adaptive fusion; standalone top-K image search is not the KIS path.
- [ ] Temporal exploration consumes `exploration_seed` and supports image-only events after multimodal scoring lands.
- [ ] Semantic operations increment once and `SearchOnly` leaves revision unchanged.
- [ ] Frontend pending operations never clear committed results/exploration before success.
- [ ] Frame click propagates exactly once with the committed exploration snapshot and records exactly one viewed-frame write.
- [ ] History persistence is best-effort but per-query ordered; activity waits for query creation and late older responses cannot replace the active session. Replay is read-only.
- [ ] Scoped resolver outputs carry full binding roles and image-only intents may adopt their first textual language on a scoped update.
- [ ] Operation-level metadata is persisted for later EventTrail evaluation.
- [ ] Full backend/frontend suites and production frontend build pass.
- [ ] Legacy/dead-code scans are clean.
