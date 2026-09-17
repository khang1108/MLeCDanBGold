# EventTrail Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the snapshot-backed EventTrail backend so a ranked KIS result can open an exact selected-video temporal hypothesis and accept revisioned `Approve`, `Decline`, `Use`, window, clear, and undo feedback without rerunning full-corpus retrieval.

**Architecture:** First finish the S0/S1 readiness blockers that EventTrail would otherwise inherit. Then make temporal search return one immutable artifact containing both ranked paths and the exact per-video scores from the same scoring generation, materialize bounded `EvidenceSnapshot`s from that artifact, and introduce a reusable `TemporalConstraintDecoder` plus one `EventTrailService` owning session actions/revisions/diffs. FastAPI exposes only the new EventTrail contracts; the existing public TemporalExploration API remains only until the separately planned frontend migration is complete, after which it is deleted rather than maintained in parallel.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, NumPy, existing HCMAI `TemporalSearchService`/DP/`Conditions`, in-process locks and `OrderedDict` TTL/LRU stores, existing structured logging.

**Spec:** `../specs/2026-09-16-event-trail-design.md`

## Global Constraints

- EventTrail implementation does not start until the S0/S1 prerequisite tests in Task 1 and Task 2 are green.
- EventTrail feedback is local to one selected ranked result/video; it never reranks other videos.
- A Trail session is permanently bound to one KIS semantic revision and one frozen selected-video score matrix.
- Trail hot-loop actions must not call an LLM, embedding model, BM25 search, image scorer, or full-corpus scorer.
- `Approve` and `Use` are hard anchors. Existing anchors never move silently.
- `Decline` rejects the current canonical occurrence using a deterministic midpoint-derived temporal cell and accumulates rejections.
- Contradictory hard actions are transactional and do not increment Trail revision.
- A `Decline` that exhausts the selected video is committed, increments Trail revision, and retains `last_valid_path`.
- Trail revision is monotonic, including Undo.
- Snapshot/session state is bounded, process-local, fixed-TTL state; no Redis/database/persistence is introduced.
- Snapshot materialization failure may degrade to `evidence_snapshot_id=null`; retrieval correctness failures may not be hidden.
- Do not expose raw score matrices or retrieval internals in HTTP responses.
- Do not add path carousels, global relevance feedback, confidence scoring, learned feedback, or query-semantic edits to this backend phase.
- Production code must not branch on `unittest.mock.Mock` identity and must not silently switch embedding providers.
- Current `src_v1.2.zip` contains backend source only. The frontend companion plan must be written from the then-current frontend tree before binding EventTrail UI behavior to concrete frontend files. Historical frontend plans are not authoritative for execution.

---

## File Structure and Ownership

### New backend modules

- `src/hcmai/event_trail/__init__.py` — public internal exports for EventTrail domain/service.
- `src/hcmai/event_trail/config.py` — fixed-TTL/capacity environment settings.
- `src/hcmai/event_trail/errors.py` — stable EventTrail domain error codes.
- `src/hcmai/event_trail/models.py` — immutable snapshot/session/checkpoint/transition dataclasses.
- `src/hcmai/event_trail/store.py` — bounded fixed-TTL LRU snapshot/session stores.
- `src/hcmai/event_trail/decoder.py` — selected-video constraint construction, rejection-cell derivation, and decoding only.
- `src/hcmai/event_trail/service.py` — snapshot capture, Trail open/get/action orchestration, revision/history/diff semantics.
- `src/hcmai/event_trail/logging.py` — structured Trail interaction records for ET-6 analysis.
- `src/hcmai/api/contracts/event_trail.py` — discriminated HTTP actions and UI-ready Trail state.
- `src/hcmai/api/routers/event_trail.py` — thin FastAPI transport/error mapping.

### Existing backend files modified

- `src/hcmai/orchestration/pipeline.py` — S0/S1 hardening; own `EventTrailService`; capture snapshot after successful KIS search.
- `src/hcmai/orchestration/workflows/kis.py` — always use multimodal plan search and return the internal temporal artifact.
- `src/hcmai/orchestration/workflows/temporal_search.py` — introduce `TemporalSearchArtifact`; score once, rank from the same score objects.
- `src/hcmai/orchestration/workflows/temporal_exploration.py` — prerequisite mock-branch removal first; deleted only after frontend migration.
- `src/hcmai/orchestration/retrieval_setup.py` — eliminate silent configured-remote -> local encoder fallback.
- `src/hcmai/kis/scoped_resolver.py` — preserve validated bindings for appended textual events.
- `src/hcmai/api/contracts/kis.py` — image-aware source validation, KIS-specific `result_id`, `evidence_snapshot_id`, snapshot timing.
- `src/hcmai/api/contracts/latency.py` — add `snapshot_ms` to KIS-visible latency.
- `src/hcmai/api/contracts/__init__.py` — export EventTrail contracts.
- `src/hcmai/api/routers/__init__.py` — export EventTrail router.
- `src/hcmai/app.py` — register EventTrail router; remove legacy exploration registry only after frontend migration.

### Targeted tests created/modified

- `tests/api/test_kis_contracts.py`
- `tests/api/test_kis_router.py`
- `tests/api/test_event_trail_routes.py`
- `tests/kis/test_scoped_resolver.py`
- `tests/orchestration/test_kis_multimodal_prerequisites.py`
- `tests/orchestration/test_retrieval_setup.py`
- `tests/orchestration/workflows/test_kis_pipeline.py`
- `tests/orchestration/workflows/test_temporal_search_artifact.py`
- `tests/event_trail/test_store.py`
- `tests/event_trail/test_decoder.py`
- `tests/event_trail/test_service.py`
- `tests/event_trail/test_kis_integration.py`

Do not build a new broad testing framework. Each test file above exists to protect one new boundary or one prerequisite regression.

---

### Task 1: Close KIS Multimodal Prerequisite Gaps

**Files:**
- Modify: `src/hcmai/orchestration/pipeline.py:233-430`
- Modify: `src/hcmai/kis/scoped_resolver.py:70-141`
- Modify: `src/hcmai/api/contracts/kis.py:39-115`
- Create/modify: `tests/orchestration/test_kis_multimodal_prerequisites.py`
- Create/modify: `tests/kis/test_scoped_resolver.py`
- Create/modify: `tests/api/test_kis_contracts.py`

**Interfaces:**
- Produces: `SearchService._canonical_image_refs(refs: Sequence[KISImageRef | str]) -> list[KISImageRef]`, which resolves every incoming ref through `KISImageAssetStore.ref(asset_id)` before it enters `KISIntent`; this exact helper is the single canonicalization boundary.
- Produces: `apply_scoped_resolutions()` that preserves validated `KISEntityBinding(entity_id, role)` for both existing and newly appended textual events.
- Produces: `KISSearchRequest` validation that permits image-only retrieval with `use_dense=False` and `use_bm25=False`, while still rejecting text-only requests with no enabled text source.
- Keeps: one semantic revision per successful patch batch and no LLM call for image-only patches.

- [ ] **Step 1: Add failing regressions for initial image canonicalization and image-only retrieval**

```python
# tests/api/test_kis_contracts.py
from hcmai.api.contracts.kis import KISSearchRequest


def test_image_only_initial_request_allows_text_sources_disabled() -> None:
    request = KISSearchRequest.model_validate({
        "base_intent": None,
        "expected_revision": 0,
        "operation": {
            "kind": "initial_resolve",
            "image_refs": [{"asset_id": "img_abc", "content_type": "image/png"}],
        },
        "use_dense": False,
        "use_bm25": False,
        "top_k": 20,
    })
    assert request.use_dense is False
    assert request.use_bm25 is False


def test_text_only_initial_request_rejects_all_text_sources_disabled() -> None:
    try:
        KISSearchRequest.model_validate({
            "base_intent": None,
            "expected_revision": 0,
            "operation": {"kind": "initial_resolve", "text": "woman enters"},
            "use_dense": False,
            "use_bm25": False,
        })
    except ValueError as exc:
        assert "retrieval source" in str(exc).lower()
    else:
        raise AssertionError("text-only request must require a text retrieval source")
```

Add an orchestration regression where the client supplies `KISImageRef(asset_id="img_abc", content_type="image/jpeg")` but the asset store's canonical `ref("img_abc")` returns `content_type="image/png"`; the committed intent must contain the canonical store value, not the client-authored metadata.

- [ ] **Step 2: Run the contract/canonicalization tests and confirm the current behavior fails**

Run:

```bash
python -m pytest \
  tests/api/test_kis_contracts.py \
  tests/orchestration/test_kis_multimodal_prerequisites.py -q
```

Expected before the fix: the image-only request is rejected when both text toggles are false and/or the natural image route retains unvalidated client-authored refs.

- [ ] **Step 3: Make source validation evidence-aware and canonicalize natural image refs**

Replace the unconditional source check in `KISSearchRequest.validate_sources()` with evidence-aware validation:

```python
def _request_may_have_image_evidence(self) -> bool:
    if self.base_intent is not None and any(event.images for event in self.base_intent.events):
        return True
    operation = self.operation
    if operation.kind == "initial_resolve":
        return bool(operation.image_refs) or any(patch.add_image_ids for patch in operation.patches)
    if operation.kind == "patch_events":
        return any(patch.add_image_ids for patch in operation.patches)
    return False

@model_validator(mode="after")
def validate_sources(self) -> Self:
    if not self.use_dense and not self.use_bm25 and not self._request_may_have_image_evidence():
        raise ValueError("at least one retrieval source or image evidence must be available")
    return self
```

The final per-event authority remains `KISRetrievalPlan.validate_text_sources()` after semantic resolution, so a mixed plan containing a text-only event still fails if both text sources are disabled.

In `SearchService`, canonicalize `InitialResolveOperation.image_refs` before either natural image branch constructs a `KISEvent`:

```python
def _canonical_image_refs(self, refs: Sequence[KISImageRef | str]) -> list[KISImageRef]:
    if not refs:
        return []
    if self.kis_image_assets is None:
        raise SearchServiceUnavailableError("KIS image asset store is unavailable")
    canonical = []
    for ref in refs:
        asset_id = ref if isinstance(ref, str) else ref.asset_id
        try:
            canonical.append(self.kis_image_assets.ref(asset_id))
        except (KeyError, ValueError) as exc:
            raise InvalidQueryInputError(f"Unknown image asset: {asset_id}") from exc
    return canonical
```

Use only the returned canonical refs in the image-only and text+image branches.

- [ ] **Step 4: Add failing regressions for image-only append and appended textual bindings**

```python
# tests/orchestration/test_kis_multimodal_prerequisites.py

def test_patch_can_append_image_only_event_atomically(search_service, base_intent, image_store) -> None:
    response = search_service.search_kis(make_patch_request(
        base_intent,
        patches=[{"event_id": "E2", "add_image_ids": ["img_plate"]}],
    ))
    event = response.intent.events[1]
    assert event.id == "E2"
    assert event.text is None
    assert [image.asset_id for image in event.images] == ["img_plate"]


# tests/kis/test_scoped_resolver.py

def test_appended_text_event_preserves_validated_binding(base_intent) -> None:
    resolved = ScopedResolutionBatch.model_validate({
        "language": "en",
        "events": [{
            "event_id": "E2",
            "text": "The same woman lifts a plate.",
            "bindings": [{"entity_id": "X1", "role": "actor"}],
        }],
    })
    updated = apply_scoped_resolutions(base_intent, resolved, revision=base_intent.revision + 1)
    assert updated.events[1].bindings[0].entity_id == "X1"
    assert updated.events[1].bindings[0].role == "actor"
```

- [ ] **Step 5: Run the new regressions and confirm they fail against the current append path**

Run:

```bash
python -m pytest \
  tests/orchestration/test_kis_multimodal_prerequisites.py \
  tests/kis/test_scoped_resolver.py -q
```

Expected before the fix: image-only append attempts to construct an empty `KISEvent`, and appended textual events drop model-returned bindings.

- [ ] **Step 6: Assemble patch events atomically and preserve appended bindings**

In `apply_scoped_resolutions()`, replace the appended-event branch with validated bindings:

```python
for number in new_numbers:
    patch = resolved_by_id[f"E{number}"]
    events.append(
        KISEvent(
            id=f"E{number}",
            text=patch.text,
            bindings=_validated_bindings(patch.bindings, base),
        )
    )
```

In `SearchService._resolve_operation()` do not create placeholder `KISEvent(text=None, images=[])` rows for new image-only IDs. Build a mutable description per target first, apply text resolution if present, apply image deltas, and construct each new `KISEvent` only after at least one of `text` or `images` is available. Existing event IDs preserve untouched text/bindings/images unless their patch changes them.

A minimal assembly shape is:

```python
assembled = {event.id: event for event in intermediate_intent.events}
for patch in op.patches:
    previous = assembled.get(patch.event_id)
    images = list(previous.images) if previous is not None else []
    images = [image for image in images if image.asset_id not in set(patch.remove_image_ids)]
    image_by_id = {image.asset_id: image for image in images}
    for image in self._canonical_image_refs(patch.add_image_ids):
        image_by_id.setdefault(image.asset_id, image)
    text = previous.text if previous is not None else None
    bindings = list(previous.bindings) if previous is not None else []
    assembled[patch.event_id] = KISEvent(
        id=patch.event_id,
        text=text,
        images=list(image_by_id.values()),
        bindings=bindings,
    )
```

This assembly must use `_canonical_image_refs`; do not introduce a second asset-ID canonicalization helper. It must never instantiate an invalid empty event as an intermediate state.

- [ ] **Step 7: Run the focused prerequisite tests**

```bash
python -m pytest \
  tests/api/test_kis_contracts.py \
  tests/kis/test_scoped_resolver.py \
  tests/orchestration/test_kis_multimodal_prerequisites.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the prerequisite semantic fixes**

```bash
git add \
  src/hcmai/api/contracts/kis.py \
  src/hcmai/orchestration/pipeline.py \
  src/hcmai/kis/scoped_resolver.py \
  tests/api/test_kis_contracts.py \
  tests/kis/test_scoped_resolver.py \
  tests/orchestration/test_kis_multimodal_prerequisites.py
git commit -m "fix: harden multimodal kis prerequisites"
```

---

### Task 2: Remove Runtime Test-Double Branching and Silent Embedding Fallback

**Files:**
- Modify: `src/hcmai/orchestration/workflows/kis.py:9-121`
- Modify: `src/hcmai/orchestration/workflows/temporal_exploration.py:118-154`
- Modify: `src/hcmai/orchestration/retrieval_setup.py:350-370`
- Create/modify: `tests/orchestration/workflows/test_kis_pipeline.py`
- Create/modify: `tests/orchestration/test_retrieval_setup.py`

**Interfaces:**
- `KISPipeline.execute()` always consumes `TemporalSearchService.search_plan(...)`; tests supply a small fake implementing that interface instead of relying on `Mock` identity.
- Legacy `TemporalExploration.open()` always consumes `score_plan(...)` until that public flow is removed after frontend migration.
- `_query_encoder(config, index, source="text")` uses local encoding only when all remote embedding configuration is absent. Remove the legacy third `embedding_client/llm` argument from this helper and both callers. If any `HCMAI_EMBEDDING_*` variable is present but the remote configuration is incomplete/invalid, `load_embedding_endpoint()` fails explicitly; it is never converted into local inference.

- [ ] **Step 1: Write a KIS pipeline regression that would fail if legacy `search()` is selected by mock identity**

```python
class FakeTemporal:
    def __init__(self, result):
        self.result = result
        self.search_plan_calls = 0

    def search_plan(self, *args, **kwargs):
        self.search_plan_calls += 1
        return self.result


def test_kis_pipeline_uses_search_plan_only(fake_temporal_result, corpus, retrieval_plan, intent):
    temporal = FakeTemporal(fake_temporal_result)
    pipeline = KISPipeline(corpus, temporal)  # type: ignore[arg-type]
    pipeline.execute(
        intent=intent,
        retrieval_plan=retrieval_plan,
        use_dense=True,
        use_bm25=False,
        top_k=5,
    )
    assert temporal.search_plan_calls == 1
```

Add a source guard test or simple `rg` gate asserting neither production workflow imports `unittest.mock`.

- [ ] **Step 2: Add embedding-provider tests for absent versus invalid remote configuration**

Test these two distinct cases:

```python
def test_query_encoder_uses_local_when_remote_env_is_absent(monkeypatch, config, index):
    monkeypatch.delenv("HCMAI_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("HCMAI_EMBEDDING_MODEL", raising=False)
    encoder = _query_encoder(config, index)
    assert encoder is not None


def test_query_encoder_does_not_fallback_when_remote_env_is_partial(monkeypatch, config, index):
    monkeypatch.setenv("HCMAI_EMBEDDING_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.delenv("HCMAI_EMBEDDING_MODEL", raising=False)
    try:
        _query_encoder(config, index)
    except KeyError as exc:
        assert "HCMAI_EMBEDDING_MODEL" in str(exc)
    else:
        raise AssertionError("partial remote configuration must fail explicitly")
```

- [ ] **Step 3: Run focused tests and confirm current behavior is exposed**

```bash
python -m pytest \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/orchestration/test_retrieval_setup.py -q
```

- [ ] **Step 4: Remove both `Mock` branches**

In `workflows/kis.py`, delete the `unittest.mock` import and the `use_plan` branch; call `self.temporal.search_plan(...)` directly.

In `workflows/temporal_exploration.py`, delete the runtime `Mock` inspection and call `self._temporal.score_plan(...)` directly. This file is still temporary product compatibility until the frontend migration, but production behavior must already match the real interface.

- [ ] **Step 5: Make embedding mode explicit by environment presence**

Remove the legacy third argument from `_query_encoder` and from the two call sites in `load_retrieval()` and `_load_fast_track_retrieval()`. Remote embedding is selected only by environment presence:

```python
def _query_encoder(config, index, source="text"):
    remote_requested = any(
        key in os.environ
        for key in (
            "HCMAI_EMBEDDING_BASE_URL",
            "HCMAI_EMBEDDING_MODEL",
            "HCMAI_EMBEDDING_API_KEY",
            "HCMAI_EMBEDDING_TIMEOUT_SECONDS",
        )
    )
    if not remote_requested:
        return EmbeddingService.create_text_adapter(config)

    client = EmbeddingClient(load_embedding_endpoint())
    return EmbeddingService.create_remote_adapter(
        client, config, index.metadata.embedding_dim, source
    )
```

Change the callers to `_query_encoder(models.visual_embedding, visual, "visual")` and `_query_encoder(models.resolved_evidence_embedding, sample, "text")`. Do not catch `KeyError`, `ValueError`, or provider construction errors inside `_query_encoder` and silently create a local adapter.

- [ ] **Step 6: Verify real fusion configuration before proceeding**

Run from the full repository root used for deployment:

```bash
rg -n "visual_image|base_component_weights" configs src/hcmai
```

The gate passes only if the production-loaded configuration assigns `visual_image` a positive value. If `configs/baseline.yaml` overrides `base_component_weights`, it must explicitly contain a positive `visual_image` entry; do not rely on the Python default when YAML replaces the dictionary.

Then run the configuration loader test used by the repository and assert:

```python
assert loaded.search.temporal.adaptive.base_component_weights["visual_image"] > 0
```

If the full repository does not contain the deployment config/test surface, stop execution here and perform this gate in the deployment repository before Task 3.

- [ ] **Step 7: Run the focused tests and source scan**

```bash
python -m pytest \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/orchestration/test_retrieval_setup.py -q
rg -n "unittest\.mock|isinstance\([^)]*Mock" \
  src/hcmai/orchestration/workflows/kis.py \
  src/hcmai/orchestration/workflows/temporal_exploration.py
```

Expected: tests PASS; `rg` returns no production match.

- [ ] **Step 8: Commit runtime-boundary cleanup**

```bash
git add \
  src/hcmai/orchestration/workflows/kis.py \
  src/hcmai/orchestration/workflows/temporal_exploration.py \
  src/hcmai/orchestration/retrieval_setup.py \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/orchestration/test_retrieval_setup.py
git commit -m "refactor: make kis retrieval runtime explicit"
```

---

### Task 3: Return Ranked Paths and Scores as One Temporal Search Artifact

**Files:**
- Modify: `src/hcmai/orchestration/workflows/temporal_search.py:25-173`
- Modify: `src/hcmai/orchestration/workflows/kis.py:26-138`
- Create: `tests/orchestration/workflows/test_temporal_search_artifact.py`
- Modify: `tests/orchestration/workflows/test_kis_pipeline.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class TemporalSearchArtifact:
    result: TemporalSearchResult
    video_scores: tuple[VideoEventScores, ...]
    decoder_config: DecoderConfigSnapshot
```

- Produces: `TemporalSearchService.search_plan_artifact(...) -> TemporalSearchArtifact`.
- Keeps: `TemporalSearchService.search_plan(...) -> TemporalSearchResult` as a compatibility wrapper that returns `.result` without rescoring.
- Changes: `KISSearchExecution` gains `temporal_artifact: TemporalSearchArtifact` so `SearchService.search_kis()` can capture snapshot evidence from the exact scoring generation that produced `results`.

- [ ] **Step 1: Write a score-once artifact test**

```python
def test_search_plan_artifact_scores_once_and_ranks_same_scores(temporal_service, plan, monkeypatch):
    calls = 0
    original = temporal_service.score_plan

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(temporal_service, "score_plan", counted)
    artifact = temporal_service.search_plan_artifact(plan, top_k=5)

    assert calls == 1
    assert artifact.result.paths
    assert artifact.video_scores
    assert artifact.decoder_config == temporal_service.snapshot_decoder_config()
```

Also assert every returned path video ID exists in `artifact.video_scores` and the path frame IDs are drawn from that exact video's score metadata.

Add a regression against the current `score_plan()` validation defect: the method must return `tuple[VideoEventScores, ...]`, never a tuple of `None`, and validation must call `_validate_video_scores(plan.event_count, video)` in that argument order.

- [ ] **Step 2: Run the new test and confirm the method is absent**

```bash
python -m pytest tests/orchestration/workflows/test_temporal_search_artifact.py -q
```

Expected before implementation: FAIL because `search_plan_artifact` does not exist.

- [ ] **Step 3: Add `TemporalSearchArtifact` and factor `search_plan()` through it**

Implement `TemporalSearchArtifact` exactly as declared above, then add `search_plan_artifact` with the same parameters/defaults as the current `search_plan`: `plan`, keyword-only `image_component=None`, `use_dense=True`, `use_bm25=False`, and `top_k=20`. Its body must:

1. reject `top_k <= 0` exactly as current `search_plan` does;
2. call `score_plan(plan, image_component=image_component, use_dense=use_dense, use_bm25=use_bm25)` once;
3. build `score_by_video` from those returned objects;
4. call `rank_paths` with the existing alignment configuration fields and `max_rows=top_k`;
5. materialize paths against the same `score_by_video` objects;
6. construct `TemporalSearchResult(paths=paths, retrieval_ms=retrieval_ms, alignment_ms=alignment_ms)`;
7. return `TemporalSearchArtifact(result=result, video_scores=scores, decoder_config=self.snapshot_decoder_config())`.

Replace current `search_plan` body with one call to `search_plan_artifact` using its received arguments and return `.result`; it must not call `score_plan` itself.

Before `search_plan_artifact()` is used, fix the existing `score_plan()` validator loop so the scored objects are preserved:

```python
validated = tuple(scores)
for video in validated:
    self._validate_video_scores(plan.event_count, video)
return validated, retrieval_ms
```

Do not call `score_plan()` from both methods.

- [ ] **Step 4: Make KIS execution retain the artifact**

Change `KISSearchExecution` to:

```python
@dataclass(frozen=True, slots=True)
class KISSearchExecution:
    results: list[SearchResult]
    latency: SearchLatency
    temporal_artifact: TemporalSearchArtifact
```

`KISPipeline.execute()` calls only `search_plan_artifact()` and materializes product results from `artifact.result.paths`.

- [ ] **Step 5: Run temporal/KIS workflow tests**

```bash
python -m pytest \
  tests/orchestration/workflows/test_temporal_search_artifact.py \
  tests/orchestration/workflows/test_kis_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the score-preserving search artifact**

```bash
git add \
  src/hcmai/orchestration/workflows/temporal_search.py \
  src/hcmai/orchestration/workflows/kis.py \
  tests/orchestration/workflows/test_temporal_search_artifact.py \
  tests/orchestration/workflows/test_kis_pipeline.py
git commit -m "feat: retain temporal search evidence artifact"
```

---

### Task 4: Add Immutable Evidence Snapshots and KIS Snapshot Handoff

**Files:**
- Create: `src/hcmai/event_trail/__init__.py`
- Create: `src/hcmai/event_trail/config.py`
- Create: `src/hcmai/event_trail/errors.py`
- Create: `src/hcmai/event_trail/models.py`
- Create: `src/hcmai/event_trail/store.py`
- Modify: `src/hcmai/api/contracts/kis.py:91-175`
- Modify: `src/hcmai/api/contracts/latency.py:13-24`
- Modify: `src/hcmai/orchestration/pipeline.py:70-156,458-531`
- Create: `tests/event_trail/test_store.py`
- Modify: `tests/api/test_kis_contracts.py`
- Create: `tests/event_trail/test_kis_integration.py`

**Interfaces:**
- Produces `EventTrailSettings.from_env()` with concrete defaults:
  - `snapshot_ttl_seconds=900`
  - `session_ttl_seconds=1800`
  - `max_snapshots=64`
  - `max_sessions=128`
  via `HCMAI_EVENT_TRAIL_SNAPSHOT_TTL_SECONDS`, `HCMAI_EVENT_TRAIL_SESSION_TTL_SECONDS`, `HCMAI_EVENT_TRAIL_MAX_SNAPSHOTS`, and `HCMAI_EVENT_TRAIL_MAX_SESSIONS`. All four values must parse as positive integers.
- Produces immutable domain types:

```python
@dataclass(frozen=True, slots=True)
class SnapshotResult:
    result_id: str
    video_id: str
    initial_path: tuple[str, ...]
    path_score: float

@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    snapshot_id: str
    kis_revision: int
    scoring_revision: str
    event_ids: tuple[str, ...]
    decoder_config: DecoderConfigSnapshot
    results: dict[str, SnapshotResult]
    video_evidence: dict[str, VideoEventScores]
    created_at: datetime
    expires_at: datetime
```

`video_evidence` values are frozen `VideoEventScores` copies whose NumPy arrays have `writeable=False`. `SnapshotResult.initial_path` is `tuple[str, ...]` of canonical frame IDs in event order, and `path_score` is the exact ranked `AlignedPath.score`. `EvidenceSnapshot.scoring_revision` is the single `SearchService.event_trail_scoring_revision` UUID generated once when the loaded indexes/configuration generation is constructed; it is not generated per query.
- Produces `EvidenceSnapshotStore.put(snapshot) -> None`, `get(snapshot_id) -> EvidenceSnapshot`, and `remove(snapshot_id) -> None` with fixed TTL and bounded LRU; reads may update LRU order but never extend expiration. Time-expired IDs are retained as bounded payload-free tombstones so a later `get` returns `SNAPSHOT_EXPIRED` rather than silently becoming `SNAPSHOT_NOT_FOUND`; capacity-evicted IDs are not tombstoned and return `SNAPSHOT_NOT_FOUND`.
- Changes KIS response to expose:

```python
class KISSearchResult(SearchResult):
    result_id: str

# add these fields to the existing KISSearchResponse model
evidence_snapshot_id: str | None = None
results: list[KISSearchResult] = Field(default_factory=list)
```

- Adds `snapshot_ms` to `SearchLatency`.
- `SearchService.search_kis()` captures only returned-result video slices, never the full corpus matrix. Degradation to `evidence_snapshot_id=None` is allowed only for `EventTrailError(code="SNAPSHOT_UNAVAILABLE", ...)` raised by the snapshot-store boundary; validation/programming/retrieval errors propagate normally.

- [ ] **Step 1: Write store immutability/TTL/LRU tests**

Test fixed expiration with an injected clock rather than sleeps:

```python
def test_snapshot_expiration_is_fixed_not_sliding(snapshot_store, clock, snapshot):
    snapshot_store.put(snapshot)
    clock.advance(500)
    assert snapshot_store.get(snapshot.snapshot_id) is snapshot
    clock.advance(401)
    with pytest.raises(EventTrailError) as exc:
        snapshot_store.get(snapshot.snapshot_id)
    assert exc.value.code == "SNAPSHOT_EXPIRED"
```

Also test that stored `VideoEventScores.scores.flags.writeable` and timestamp/frame arrays are false, and that multiple results for the same video reference one `video_evidence[video_id]` entry.

- [ ] **Step 2: Add failing KIS handoff integration tests**

Through `SearchService.search_kis()` assert:

```python
response = service.search_kis(request)
assert response.evidence_snapshot_id is not None
assert all(result.result_id for result in response.results)
assert response.latency.snapshot_ms >= 0

snapshot = service.event_trail_snapshots.get(response.evidence_snapshot_id)
assert set(snapshot.video_evidence) == {result.video_id for result in response.results}
assert len(snapshot.results) == len(response.results)
```

Count the underlying temporal scorer and assert exactly one full scoring call for the search.

- [ ] **Step 3: Run the new tests and confirm snapshot types/response fields are absent**

```bash
python -m pytest \
  tests/event_trail/test_store.py \
  tests/event_trail/test_kis_integration.py \
  tests/api/test_kis_contracts.py -q
```

- [ ] **Step 4: Implement EventTrail settings and stable domain errors**

`event_trail/errors.py` should use one exception carrying a stable code instead of one class per HTTP status:

```python
class EventTrailError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
```

`config.py` reads the four exact environment variables named in the Interfaces block. Missing values use `900`, `1800`, `64`, and `128`; blank, non-integer, zero, or negative values raise `ValueError` during service construction.

- [ ] **Step 5: Implement immutable snapshot materialization**

Move the useful freeze behavior currently duplicated in `temporal_exploration._freeze_video()` into `event_trail/models.py`:

```python
def freeze_video_scores(video: VideoEventScores) -> VideoEventScores:
    arrays = {}
    for name in ("frame_ids", "frame_idx", "timestamps_ms", "scores"):
        value = getattr(video, name).copy()
        value.setflags(write=False)
        arrays[name] = value
    return replace(video, **arrays)
```

Require `len(execution.results) == len(artifact.result.paths)` before snapshot construction. Build snapshot results by zipping KIS product-result order with `artifact.result.paths`, and construct each public result explicitly as:

```python
KISSearchResult(result_id=result_id, **result.model_dump())
```

Generate opaque result IDs (`r_<uuid>`) once per response. Keep `initial_path=tuple(path.frame_ids)` and `path_score=path.score` from that exact aligned path. A length/order mismatch is a correctness error and must not be downgraded to EventTrail unavailability.

- [ ] **Step 6: Implement fixed-TTL bounded stores**

Use `OrderedDict` for LRU ordering and an injected monotonic clock for tests. On expiration, delete the payload and move the ID into a second bounded `OrderedDict` tombstone set; `get()` checks live entry, then tombstone, then not-found. Bound tombstones to `max_snapshots` so error fidelity cannot create unbounded state. Do not extend `expires_at` on read.

- [ ] **Step 7: Add KIS response fields and snapshot timing**

Add `KISSearchResult`, `evidence_snapshot_id`, and `SearchLatency.snapshot_ms`.

In `SearchService.search_kis()`:

1. run KIS execution once;
2. assign result IDs;
3. time snapshot materialization/store and set `snapshot_ms` to the full attempted snapshot duration;
4. if freezing/copying evidence raises `MemoryError`, convert only that optional-snapshot resource failure to `EventTrailError("SNAPSHOT_UNAVAILABLE", ...)`; catch that one code at the KIS boundary, return `evidence_snapshot_id=None`, and append `EVENT_TRAIL_UNAVAILABLE` to warnings;
5. re-raise every other `EventTrailError` and all validation/programming/retrieval errors;
6. return latency using `execution.latency.model_copy(update={"snapshot_ms": snapshot_ms, "total_ms": execution.latency.total_ms + snapshot_ms})` so public total latency includes work the request actually waited for, including a failed optional snapshot attempt.

Add an optional `event_trail_settings: EventTrailSettings | None = None` parameter to `SearchService.__init__` for deterministic tests. Initialize `self.event_trail_settings = event_trail_settings or EventTrailSettings.from_env()`, `self.event_trail_snapshots = EvidenceSnapshotStore(ttl_seconds=self.event_trail_settings.snapshot_ttl_seconds, max_entries=self.event_trail_settings.max_snapshots)`, and one process-generation `self.event_trail_scoring_revision = str(uuid4())`, not per request. Snapshot metadata uses `created_at=datetime.now(timezone.utc)` and `expires_at=created_at + timedelta(seconds=self.event_trail_settings.snapshot_ttl_seconds)`; the store independently enforces expiration using its monotonic clock so wall-clock jumps cannot extend lifetime. Task 4 does **not** create `EventTrailService` yet; Task 6 creates the session store/decoder/service and reuses these exact snapshot fields.

- [ ] **Step 8: Run store and KIS integration tests**

```bash
python -m pytest \
  tests/event_trail/test_store.py \
  tests/event_trail/test_kis_integration.py \
  tests/api/test_kis_contracts.py -q
```

Expected: PASS and one scorer invocation per KIS search.

- [ ] **Step 9: Commit evidence handoff**

```bash
git add \
  src/hcmai/event_trail \
  src/hcmai/api/contracts/kis.py \
  src/hcmai/api/contracts/latency.py \
  src/hcmai/orchestration/pipeline.py \
  tests/event_trail/test_store.py \
  tests/event_trail/test_kis_integration.py \
  tests/api/test_kis_contracts.py
git commit -m "feat: snapshot temporal evidence for event trail"
```

---

### Task 5: Extract the Selected-Video Temporal Constraint Decoder

**Files:**
- Create: `src/hcmai/event_trail/decoder.py`
- Reuse without semantic change: `src/hcmai/orchestration/workflows/temporal_search.py:227-320`
- Create: `tests/event_trail/test_decoder.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class ConstraintSnapshot:
    anchors: tuple[str | None, ...]
    rejected_cells: tuple[tuple[Interval, ...], ...]
    window: Interval | None

@dataclass(frozen=True, slots=True)
class DecodeOutcome:
    status: Literal["ok", "contradictory_conditions", "no_indexed_frames", "no_valid_path"]
    path: AlignedPath | None
    constraint_ms: float
    dp_ms: float
```

Exact callable surface:

```text
TemporalConstraintDecoder(temporal: TemporalSearchService)
TemporalConstraintDecoder.decode(video: VideoEventScores, constraints: ConstraintSnapshot, decoder_config: DecoderConfigSnapshot) -> DecodeOutcome
TemporalConstraintDecoder.rejection_cell(video: VideoEventScores, frame_id: str) -> Interval
```

- `window=None` means `[first_timestamp_ms, last_timestamp_ms]`.
- Hard anchor frame IDs are converted to exact `[timestamp, timestamp]` confirmations.
- Rejection cells are derived in integer milliseconds from the selected canonical timestamp group. For strictly increasing neighbors around timestamp `t`:
  - left boundary: `floor((t_prev + t) / 2) + 1`;
  - right boundary: `floor((t + t_next) / 2)`;
  - the first timestamp group starts at the full-video window start;
  - the last timestamp group ends at the full-video window end.
  If multiple indexed frames share the same timestamp, the temporal mask cannot distinguish those frames. Treat the equal-timestamp run as one occurrence and use the nearest strictly smaller/larger timestamps as midpoint neighbors. A video with one distinct timestamp rejects exactly that timestamp.
- Decoder uses only the supplied frozen `VideoEventScores`, conditions, and decoder config. It reports structural decode status only; it does **not** decide whether an action commits. `EventTrailService` maps any non-`ok` outcome to transactional conflict for hard actions (`Approve`, `Use`, `SetWindow`), but commits an exhausting `Decline`; clear/undo actions commit the restored/relaxed constraints and may remain exhausted if other rejections still eliminate every path.

- [ ] **Step 1: Write midpoint-cell tests including first/last candidates**

```python
def test_rejection_cell_uses_neighbor_midpoints(decoder, scored_video):
    # timestamps: [1000, 3000, 7000]
    assert decoder.rejection_cell(scored_video, "f2") == (2001, 5000)
    assert decoder.rejection_cell(scored_video, "f1") == (1000, 2000)
    assert decoder.rejection_cell(scored_video, "f3") == (5001, 7000)
```

Add a duplicate-timestamp fixture `[1000, 3000, 3000, 7000]` and assert either frame at `3000` yields `(2001, 5000)`. Use the actual frame IDs/timestamps in the fixture; do not hard-code seconds in production.

- [ ] **Step 2: Write constraint behavior tests**

Cover exactly these behaviors:

```text
anchor E2 -> E2 frame remains fixed while E1/E3 may re-align
reject E2 candidate twice -> both cells remain excluded
anchor outside new window -> conflict
rejections removing all valid paths -> exhausted
clear anchor/window -> decoder can recover a path
```

Assert that an anchored event's frame ID never changes under unrelated decoding.

- [ ] **Step 3: Run decoder tests and confirm the new boundary is absent**

```bash
python -m pytest tests/event_trail/test_decoder.py -q
```

- [ ] **Step 4: Implement `TemporalConstraintDecoder` as a thin adapter over existing algorithms**

Build existing `Conditions`:

```python
confirmed = []
for frame_id in constraints.anchors:
    if frame_id is None:
        confirmed.append(None)
    else:
        timestamp = timestamp_for_frame(video, frame_id)
        confirmed.append((timestamp, timestamp))

window = constraints.window or (
    int(video.timestamps_ms[0]),
    int(video.timestamps_ms[-1]),
)
conditions = Conditions(
    window=window,
    confirmed=tuple(confirmed),
    rejected=constraints.rejected_cells,
)
```

Time condition construction plus `build_mask()` into `constraint_ms`. If `build_mask()` returns `contradictory_conditions` or `no_indexed_frames`, return that status without calling DP. Otherwise time `self.temporal.decode_video(video, allowed=allowed, decoder_config=decoder_config)` into `dp_ms`; return `ok` with the first path when one exists, otherwise `no_valid_path`. Do not duplicate DP recurrence or ranking logic.

- [ ] **Step 5: Run decoder tests**

```bash
python -m pytest tests/event_trail/test_decoder.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the algorithm boundary**

```bash
git add \
  src/hcmai/event_trail/decoder.py \
  tests/event_trail/test_decoder.py
git commit -m "refactor: isolate temporal constraint decoding"
```

---

### Task 6: Implement Revisioned EventTrail Sessions and Actions

**Files:**
- Expand: `src/hcmai/event_trail/models.py`
- Expand: `src/hcmai/event_trail/store.py`
- Create: `src/hcmai/event_trail/service.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Create: `tests/event_trail/test_service.py`

**Interfaces:**
- Produces immutable session/checkpoint state:

```python
@dataclass(frozen=True, slots=True)
class SubmissionSelection:
    event_id: str
    frame_id: str
    frame_idx: int
    timestamp_ms: int

@dataclass(frozen=True, slots=True)
class TrailCheckpoint:
    constraints: ConstraintSnapshot
    submission_selection: SubmissionSelection | None

@dataclass(frozen=True, slots=True)
class EventTrailSession:
    session_id: str
    snapshot_id: str
    result_id: str
    video_id: str
    kis_revision: int
    scoring_revision: str
    event_ids: tuple[str, ...]
    video_evidence: VideoEventScores
    decoder_config: DecoderConfigSnapshot
    trail_revision: int
    constraints: ConstraintSnapshot
    current_path: AlignedPath | None
    last_valid_path: AlignedPath | None
    history: tuple[TrailCheckpoint, ...]
    submission_selection: SubmissionSelection | None
    status: Literal["active", "exhausted"]
```

- Produces domain action types in `event_trail/models.py`:

```python
@dataclass(frozen=True, slots=True)
class ApproveEvent:
    event_id: str

@dataclass(frozen=True, slots=True)
class UseFrame:
    event_id: str
    frame_id: str

@dataclass(frozen=True, slots=True)
class DeclineCandidate:
    event_id: str

@dataclass(frozen=True, slots=True)
class ClearAnchor:
    event_id: str

@dataclass(frozen=True, slots=True)
class SetWindow:
    start_ms: int
    end_ms: int

@dataclass(frozen=True, slots=True)
class ClearWindow:
    pass

@dataclass(frozen=True, slots=True)
class Undo:
    pass

TrailAction = ApproveEvent | UseFrame | DeclineCandidate | ClearAnchor | SetWindow | ClearWindow | Undo
```

- Produces `EventTrailSessionStore` with fixed TTL and bounded LRU. Exact store surface: `put(session)`, `get(session_id)`, `remove(session_id)`, and `locked(session_id) -> ContextManager[SessionSlot]`. The private mutable slot is:

```python
@dataclass(slots=True)
class SessionSlot:
    session: EventTrailSession
    lock: RLock
    in_use: int
    inserted_at: float
```

`locked()` increments `in_use` under the store map lock before acquiring/yielding the slot, then decrements it in `finally`. Capacity eviction skips `in_use > 0` slots; service mutation replaces `slot.session` while `slot.lock` is held. If every slot is in use at capacity, opening another Trail raises `EventTrailError("EVENT_TRAIL_UNAVAILABLE", "EventTrail session capacity is busy")`, which the router maps to infrastructure HTTP 503. Expired session IDs use bounded tombstones like snapshots so they map to `TRAIL_SESSION_EXPIRED`.
- Task 6 initializes `self.event_trail_sessions = EventTrailSessionStore(ttl_seconds=self.event_trail_settings.session_ttl_seconds, max_entries=self.event_trail_settings.max_sessions)`. If `self.kis.temporal is not None`, initialize `self.event_trail_decoder = TemporalConstraintDecoder(self.kis.temporal)` and `self.event_trail = EventTrailService(snapshot_store=self.event_trail_snapshots, session_store=self.event_trail_sessions, decoder=self.event_trail_decoder)`; otherwise set both fields to `None`. The router maps a missing service to infrastructure HTTP 503 and never constructs a second temporal service.
- Produces the internal UI-ready `TrailView` dataclass containing the fields later mirrored by `EventTrailStateResponse`: session/result/video IDs, KIS/Trail revisions, status, `path`, `last_valid_path`, approved IDs, rejection counts, window, `SubmissionSelection | None`, and `TrailTransition | None`.
- Produces `EventTrailService`:

Exact service surface:

```text
EventTrailService.open(snapshot_id: str, result_id: str, expected_kis_revision: int) -> TrailView
EventTrailService.get(session_id: str) -> TrailView
EventTrailService.act(session_id: str, expected_trail_revision: int, action: TrailAction) -> TrailView
```

- `open()` copies/retains selected `VideoEventScores` independently enough that later snapshot eviction does not invalidate the session.
- `open()` sets `trail_revision=0` and `current_path` from the stored ranked result without unconstrained re-decoding.
- Every committed mutation increments revision exactly once. Rejected hard conflicts do not increment.

- [ ] **Step 1: Write open invariants and snapshot-lifetime tests**

```python
def test_open_uses_exact_ranked_path_without_scoring_or_redecode(service, snapshot):
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    assert [row.frame_id for row in view.path] == list(snapshot.results["r_1"].initial_path)
    assert view.trail_revision == 0


def test_open_session_survives_snapshot_eviction(service, stores, snapshot):
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    stores.snapshots.remove(snapshot.snapshot_id)
    assert service.get(view.session_id).path == view.path
```

Instrument the scorer/temporal search service and assert neither is called by `open()`. Also use the injected clock to verify session reads/actions do not extend the original `session_ttl_seconds`, and that a time-expired session returns `TRAIL_SESSION_EXPIRED` while a capacity-evicted session returns `TRAIL_SESSION_NOT_FOUND`.

- [ ] **Step 2: Write action semantics tests before service implementation**

Cover:

1. `ApproveEvent(E2)` anchors the current E2 frame and may re-align neighbors.
2. `UseFrame(E2, f)` resolves `f` in the frozen selected-video arrays, then anchors it and sets `SubmissionSelection(event_id="E2", frame_id=f, frame_idx=resolved_frame_idx, timestamp_ms=resolved_timestamp_ms)`.
3. `DeclineCandidate(E2)` rejects the current E2 cell and re-decodes.
4. `DeclineCandidate` on anchored E2 returns `CONSTRAINT_CONFLICT` without mutation; `ApproveEvent` or `UseFrame` on an already anchored event also returns `CONSTRAINT_CONFLICT` until `ClearAnchor`/Undo removes that anchor.
5. `ApproveEvent` or `DeclineCandidate` while `current_path is None` returns `CONSTRAINT_CONFLICT`; a contradictory unanchored `Approve`, `UseFrame`, or `SetWindow` returns conflict with unchanged revision/path/history.
6. an exhausting `Decline` commits, increments revision, sets `status="exhausted"`, `current_path=None`, `last_valid_path=old_path`.
7. `Undo` restores the previous checkpoint but increments revision monotonically.
8. `ClearAnchor` and `ClearWindow` relax only the requested constraint.
9. stale `expected_trail_revision` returns `TRAIL_REVISION_CONFLICT` without mutation.
10. invalid event/frame/window and empty undo return their specified 422 error codes.

- [ ] **Step 3: Run service tests and confirm the API is absent**

```bash
python -m pytest tests/event_trail/test_service.py -q
```

- [ ] **Step 4: Implement UI-ready path and transition models internally**

Define these exact internal view types:

```python
@dataclass(frozen=True, slots=True)
class EventCandidate:
    event_id: str
    frame_id: str
    frame_idx: int
    timestamp_ms: int

@dataclass(frozen=True, slots=True)
class CandidateDiff:
    event_id: str
    before_frame_id: str | None
    after_frame_id: str | None
    before_timestamp_ms: int | None
    after_timestamp_ms: int | None

@dataclass(frozen=True, slots=True)
class TrailTransition:
    action_event_id: str | None
    direct_changed_event_ids: tuple[str, ...]
    indirect_changed_event_ids: tuple[str, ...]
    candidate_diffs: tuple[CandidateDiff, ...]
    latency_ms: float

@dataclass(frozen=True, slots=True)
class TrailView:
    session_id: str
    result_id: str
    video_id: str
    kis_revision: int
    trail_revision: int
    status: Literal["active", "exhausted"]
    path: tuple[EventCandidate, ...] | None
    last_valid_path: tuple[EventCandidate, ...] | None
    approved_event_ids: tuple[str, ...]
    rejected_counts: dict[str, int]
    window: Interval | None
    submission_selection: SubmissionSelection | None
    transition: TrailTransition | None
```

Compare canonical frame IDs in event order. `action_event_id` is direct context; changed events other than the target are indirect. Approved anchors must never appear as changed under an unrelated action; treat that as an invariant error in service tests.

Add `materialize_snapshot_path(result: SnapshotResult, video: VideoEventScores) -> AlignedPath` in `service.py`: resolve every stored frame ID against the frozen video's frame arrays, rebuild frame indices/timestamps in event order, preserve `result.path_score`, and reject missing or non-unique frame-ID matches with an internal `RuntimeError("snapshot evidence is inconsistent")`. This is an infrastructure correctness failure and is not mapped to a public constraint/domain error code. `open()` uses this helper and never calls the decoder for revision 0.

- [ ] **Step 5: Implement per-session transactional mutation**

Under the per-session lock:

```text
load current session
verify expected revision
make candidate constraint/checkpoint state
decode candidate state
apply action-specific commit rule
append prior checkpoint for committed non-undo actions
increment revision exactly once on commit
build transition diff
replace session in store
```

Do not publish partially mutated session state.

Store each Decline cell as a raw appended cell in `ConstraintSnapshot.rejected_cells`; do not pre-merge the tuple in session state. `build_mask()` may merge intervals for decoding, while `TrailView.rejected_counts[event_id]` remains the count of committed Decline cells/actions.

For event-targeted actions, `TrailTransition.direct_changed_event_ids` contains the target event even when its frame did not move (for example Approve); `candidate_diffs` contains only actual frame changes. Candidate changes on other events are `indirect_changed_event_ids`. Window/Undo have `action_event_id=None`; all actual frame changes are indirect. If a Decline exhausts the video, direct IDs contain only the declined event, indirect IDs are empty, and the transition compares the old candidate to `None` only for the direct event.

For Undo, pop one prior checkpoint, re-decode it, preserve the remaining history, and set `trail_revision = old_revision + 1`.

- [ ] **Step 6: Run service + decoder tests**

```bash
python -m pytest \
  tests/event_trail/test_decoder.py \
  tests/event_trail/test_service.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit session orchestration**

```bash
git add \
  src/hcmai/event_trail/models.py \
  src/hcmai/event_trail/store.py \
  src/hcmai/event_trail/service.py \
  src/hcmai/orchestration/pipeline.py \
  tests/event_trail/test_service.py
git commit -m "feat: add revisioned event trail sessions"
```

---

### Task 7: Expose the EventTrail HTTP Contract

**Files:**
- Create: `src/hcmai/api/contracts/event_trail.py`
- Create: `src/hcmai/api/routers/event_trail.py`
- Modify: `src/hcmai/api/contracts/__init__.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/app.py:20-34,73-82,202-210`
- Create: `tests/api/test_event_trail_routes.py`

**Interfaces:**
- `POST /api/v1/event-trail/open`

```json
{
  "snapshot_id": "es_...",
  "result_id": "r_...",
  "expected_kis_revision": 5
}
```

- `GET /api/v1/event-trail/{session_id}` — read-only recovery; no revision increment.
- `POST /api/v1/event-trail/{session_id}/actions`

```json
{
  "expected_trail_revision": 2,
  "action": {"type": "decline", "event_id": "E2"}
}
```

- Action discriminator values and fields:

```text
approve      -> event_id
use_frame    -> event_id, frame_id
decline      -> event_id
clear_anchor -> event_id
set_window   -> start_ms, end_ms
clear_window -> no extra field
undo         -> no extra field
```

- Public error body uses `detail={"code": <stable-code>, "message": <human-message>}`.
- `src/hcmai/api/contracts/event_trail.py` defines the exact transport union below; every model uses `ConfigDict(extra="forbid")`:

```python
class ApproveAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["approve"] = "approve"
    event_id: str

class UseFrameAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["use_frame"] = "use_frame"
    event_id: str
    frame_id: str

class DeclineAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["decline"] = "decline"
    event_id: str

class ClearAnchorAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["clear_anchor"] = "clear_anchor"
    event_id: str

class SetWindowAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["set_window"] = "set_window"
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must not exceed end_ms")
        return self

class ClearWindowAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["clear_window"] = "clear_window"

class UndoAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["undo"] = "undo"

EventTrailActionContract = Annotated[
    ApproveAction | UseFrameAction | DeclineAction | ClearAnchorAction |
    SetWindowAction | ClearWindowAction | UndoAction,
    Field(discriminator="type"),
]

class EventTrailActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_trail_revision: int = Field(ge=0)
    action: EventTrailActionContract
```

Router code maps these models one-to-one into the Task 6 domain actions. `EventTrailStateResponse.submission_selection` is `EventTrailSubmissionSelection(event_id, frame_id, frame_idx, timestamp_ms) | None`, never a bare frame ID.

- [ ] **Step 1: Write contract validation tests**

Assert extra fields are forbidden, action discriminator selects the correct model, windows require ordered non-negative integers, and `use_frame` requires both event/frame IDs.

- [ ] **Step 2: Write route lifecycle/error tests with a real EventTrailService over fake frozen evidence**

One focused scenario:

```text
open r_1
GET state rev0
approve E1 -> rev1
decline E2 -> rev2 or exhausted
GET same rev2
stale action expected rev1 -> 409 TRAIL_REVISION_CONFLICT
```

Separate small parameterized assertions cover:

```text
SNAPSHOT_NOT_FOUND -> 404
RESULT_NOT_FOUND -> 404
TRAIL_SESSION_NOT_FOUND -> 404
KIS_REVISION_MISMATCH -> 409
CONSTRAINT_CONFLICT -> 409
SNAPSHOT_EXPIRED -> 410
TRAIL_SESSION_EXPIRED -> 410
INVALID_EVENT / INVALID_FRAME / INVALID_WINDOW / NOTHING_TO_UNDO -> 422
```

- [ ] **Step 3: Run API tests and confirm routes are absent**

```bash
python -m pytest tests/api/test_event_trail_routes.py -q
```

- [ ] **Step 4: Implement Pydantic transport models**

`EventTrailStateResponse` mirrors only UI-needed state:

```text
session_id
result_id
video_id
kis_revision
trail_revision
status
path | null
last_valid_path | null
approved_event_ids
rejected_counts
window | null
submission_selection | null
transition | null
```

Do not serialize `VideoEventScores`, NumPy arrays, raw `Conditions`, decoder config, or score values.

- [ ] **Step 5: Implement thin router/error mapping**

Router functions first require `service_container["service"]` and `service.event_trail`; if either is absent, raise the existing service-unavailable HTTP 503 response with message `"EventTrail is unavailable"`. Otherwise call `service.event_trail.open/get/act` in `run_in_threadpool`. A single code-to-status map owns stable EventTrail domain errors:

```python
_STATUS_BY_CODE = {
    "SNAPSHOT_NOT_FOUND": 404,
    "RESULT_NOT_FOUND": 404,
    "TRAIL_SESSION_NOT_FOUND": 404,
    "KIS_REVISION_MISMATCH": 409,
    "TRAIL_REVISION_CONFLICT": 409,
    "CONSTRAINT_CONFLICT": 409,
    "SNAPSHOT_EXPIRED": 410,
    "TRAIL_SESSION_EXPIRED": 410,
    "INVALID_EVENT": 422,
    "INVALID_FRAME": 422,
    "INVALID_WINDOW": 422,
    "NOTHING_TO_UNDO": 422,
    "EVENT_TRAIL_UNAVAILABLE": 503,
}
```

Do not catch arbitrary exceptions and relabel them as constraint conflicts.

- [ ] **Step 6: Register the router without creating another registry**

`SearchService` owns the EventTrail service/stores. `app.py` should only include `create_event_trail_router(service_container)`; do not create a second app-level EventTrail registry.

- [ ] **Step 7: Run API and app import checks**

```bash
python -m pytest tests/api/test_event_trail_routes.py -q
python -m compileall -q src/hcmai
```

Expected: PASS / exit 0.

- [ ] **Step 8: Commit the HTTP surface**

```bash
git add \
  src/hcmai/api/contracts/event_trail.py \
  src/hcmai/api/routers/event_trail.py \
  src/hcmai/api/contracts/__init__.py \
  src/hcmai/api/routers/__init__.py \
  src/hcmai/app.py \
  tests/api/test_event_trail_routes.py
git commit -m "feat: expose event trail api"
```

---

### Task 8: Add EventTrail Interaction and Latency Logging

**Files:**
- Create: `src/hcmai/event_trail/logging.py`
- Modify: `src/hcmai/event_trail/service.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Modify: `src/hcmai/api/contracts/latency.py`
- Create/modify: `tests/event_trail/test_service.py`
- Modify: `tests/event_trail/test_kis_integration.py`

**Interfaces:**
- Emits structured Trail records with this common envelope:

```text
timestamp
search_session_id | null
kis_revision
type
snapshot_id | null
result_id | null
video_id | null
trail_session_id | null
trail_revision | null
event_id | null
payload
```

`timestamp` is an ISO-8601 UTC string from `datetime.now(timezone.utc).isoformat()`. `search_session_id` is nullable in this backend-only phase because the current frontend/task-session owner is not present in `src_v1.2`; the frontend companion plan must supply the stable task/search-session correlation value when its current source is inspected. Do not fabricate cross-query session identity from KIS revision or snapshot ID.

- Trail mutation payload includes path before/after canonical frame IDs, direct/indirect changed event IDs, action latency, and active/exhausted outcome.
- KIS latency includes `snapshot_ms`.
- Trail service measures `constraint_ms`, `dp_ms`, `diff_ms`, `total_ms`; these are logging/transition diagnostics and not raw retrieval scores.

- [ ] **Step 1: Add log-capture tests for one Decline and one exhausting Decline**

Capture the logger and assert the JSON record contains:

```python
assert record["type"] == "trail_decline"
assert record["event_id"] == "E2"
assert record["payload"]["path_before"]
assert "direct_changed_event_ids" in record["payload"]
assert "indirect_changed_event_ids" in record["payload"]
assert record["payload"]["total_ms"] >= 0
```

For exhaustion, assert `outcome == "exhausted"` and the path-after value is null/empty while last-valid information remains in session state rather than the log containing a score matrix.

- [ ] **Step 2: Implement a small structured logger helper**

Use `hcmai.common.utils.logging.get_logger("hcmai.event_trail.interactions")` and emit exactly one compact JSON object as the log message:

```python
logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
```

`record` is the common ET-6 envelope plus payload and contains only JSON-native scalars/lists/dicts. Do not add a database and do not pass NumPy arrays, score matrices, embeddings, or model objects to the logger.

- [ ] **Step 3: Instrument open and committed Trail actions**

Emit `trail_open`, `trail_approve`, `trail_decline`, `trail_use`, `trail_undo`, and `trail_window` from the service. `clear_window` uses `trail_window` with `payload.operation="clear"`; `clear_anchor` emits `trail_clear_anchor`. When a committed action first transitions the session to exhausted, emit a second `trail_exhausted` record after the action record. A successful `UseFrame` also emits `submission_select` with the selected canonical frame metadata. A rejected transactional conflict may be logged at debug/error level but must not masquerade as a committed Trail revision event. `trail_close`, `submission_send`, and `submission_result` are deferred to the frontend/VBS companion integration because only that layer observes Back/close and DRES submission lifecycle.

- [ ] **Step 4: Verify no heavy retrieval dependency is reachable from the action loop**

Add a service integration test that installs sentinels which raise if called after session open:

```python
def fail_rescore(*args, **kwargs):
    raise AssertionError("EventTrail action reran full-corpus scoring")

monkeypatch.setattr(service.decoder.temporal, "score_plan", fail_rescore)
service.act(session_id, revision, DeclineCandidate(event_id="E2"))
```

`EventTrailService.decoder` and `TemporalConstraintDecoder.temporal` are the concrete dependency names established in Tasks 5-6. The action loop may call `decode_video()` only.

- [ ] **Step 5: Run logging/hot-loop tests**

```bash
python -m pytest \
  tests/event_trail/test_service.py \
  tests/event_trail/test_kis_integration.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit evaluation instrumentation**

```bash
git add \
  src/hcmai/event_trail/logging.py \
  src/hcmai/event_trail/service.py \
  src/hcmai/orchestration/pipeline.py \
  src/hcmai/api/contracts/latency.py \
  tests/event_trail/test_service.py \
  tests/event_trail/test_kis_integration.py
git commit -m "feat: log event trail interactions"
```

---

### Task 9: Run the Backend EventTrail Acceptance Gate

**Files:**
- No feature files should be introduced in this task.
- Remove generated `__pycache__`/`.pyc` from the working tree if present.

**Interfaces:**
- Verifies the backend sub-project is ready for the frontend companion plan.
- Does **not** claim full EventTrail v1 completion because current frontend source has not been inspected/migrated and the legacy public TemporalExploration API therefore cannot yet be deleted safely.

- [ ] **Step 1: Run targeted EventTrail/KIS suites**

```bash
python -m pytest \
  tests/api/test_kis_contracts.py \
  tests/api/test_kis_router.py \
  tests/api/test_event_trail_routes.py \
  tests/kis/test_scoped_resolver.py \
  tests/orchestration/test_kis_multimodal_prerequisites.py \
  tests/orchestration/test_retrieval_setup.py \
  tests/orchestration/workflows/test_kis_pipeline.py \
  tests/orchestration/workflows/test_temporal_search_artifact.py \
  tests/event_trail/test_store.py \
  tests/event_trail/test_decoder.py \
  tests/event_trail/test_service.py \
  tests/event_trail/test_kis_integration.py -q
```

Expected: PASS with zero failures.

- [ ] **Step 2: Run the full backend suite available in the implementation repository**

```bash
python -m pytest -q
```

Expected: PASS. If unrelated pre-existing failures exist, record exact failing test names/output before proceeding; do not call the EventTrail backend gate green.

- [ ] **Step 3: Compile and scan production source**

```bash
python -m compileall -q src/hcmai
rg -n "unittest\.mock|isinstance\([^)]*Mock" src/hcmai
rg -n "score_plan\(|search_plan_artifact\(" src/hcmai/event_trail src/hcmai/api/routers/event_trail.py
```

Expected:

- compile exits 0;
- no production mock-identity branch exists;
- EventTrail open/action code contains no full-corpus `score_plan()` call; snapshot capture reaches the temporal search artifact only through KIS search.

- [ ] **Step 4: Verify KIS -> snapshot -> Trail lifecycle with one integration scenario**

Run the integration test with call counters asserting:

```text
KIS search                -> one full-corpus scoring pass
Trail open                -> zero scoring pass
Approve/Decline/Use/Undo  -> zero scoring pass
```

Also assert Trail initial path frame IDs exactly equal the clicked KIS result's stored path frame IDs.

- [ ] **Step 5: Clean generated Python artifacts**

```bash
find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
find src tests -type f -name '*.pyc' -delete
```

- [ ] **Step 6: Commit backend acceptance cleanup**

```bash
git add -A
git commit -m "test: verify event trail backend lifecycle"
```

---

## Companion Frontend Plan Gate

Do not write or execute the frontend/Event Rail migration from the backend-only `src_v1.2.zip`. The approved spec explicitly requires inspecting the actual frontend repository before binding UI concepts to files. Historical paths such as `frontend/src/features/search/components/SearchWorkspace.jsx`, `frontend/src/features/alignment/hooks/useTemporalExploration.js`, and `frontend/src/features/frames/components/ImageModal.jsx` are useful orientation only; they may be stale.

Once the current frontend source is available, create a second implementation plan from the same spec covering:

```text
KIS response snapshot/result IDs
-> EventTrail API client
-> active Trail session state
-> Event Rail + Evidence Inspector
-> player focus/diff review behavior
-> preserved grid scroll/result state
-> explicit Use-based submission selection
-> interaction/session correlation logging
-> migrate all public /api/v1/exploration consumers
-> remove `KISExplorationSeed` / `exploration_seed` from the KIS HTTP response if the consumer scan shows no non-legacy caller remains
-> delete legacy TemporalExploration API/contracts/router
```

Only after that companion plan is implemented and frontend tests/build are green should the repository delete:

```text
src/hcmai/api/contracts/exploration.py
src/hcmai/api/routers/exploration.py
public TemporalExploration product/session orchestration in
src/hcmai/orchestration/workflows/temporal_exploration.py
app-level ExplorationRegistry wiring in src/hcmai/app.py
```

Retain only algorithmic code that has been moved into `event_trail/decoder.py` or another single-purpose internal boundary.

---

## Backend Completion Checklist Against the Spec

This backend phase is green only when:

1. all S0/S1 prerequisite defects covered by Tasks 1-2 are fixed and production `visual_image` weight is verified positive;
2. KIS scoring happens once and produces one `TemporalSearchArtifact` containing ranked paths and exact score objects;
3. each KIS result has an opaque `result_id` and successful searches optionally expose `evidence_snapshot_id`;
4. snapshot storage retains only returned-result video slices and uses fixed TTL/LRU bounds;
5. EventTrail open does not rescore and opens exactly the stored ranked path;
6. open sessions survive source snapshot expiration/eviction;
7. Approve/Use/Decline/window/clear/undo semantics match ET-2 and use only local constrained DP;
8. contradictory hard actions are transactional; exhausting Decline is committed;
9. Trail revisions are independent and monotonic;
10. API errors and action discriminators match ET-5;
11. action responses contain UI-ready direct/indirect diffs but no raw matrices/scores;
12. logging records local feedback behavior and latency without storing embeddings/matrices;
13. targeted and full backend tests pass on the implementation revision;
14. the frontend companion plan remains the only remaining blocker to deleting the legacy public TemporalExploration interface and declaring full EventTrail v1 complete.
