# Segment-Projected ASR Dense Temporal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Dense/Hybrid temporal retrieval use the existing segment-native `artifacts/indexes/asr_segments` index projected onto canonical frames, eliminating the runtime requirement for the nonexistent redundant `artifacts/indexes/asr` frame-native Dense index.

**Architecture:** Keep ASR source-of-truth and the existing segment Dense index unchanged. Add a runtime `SegmentProjectedASRIndex` adapter that mirrors the visual index's canonical frame identity, precomputes segment→frame positions through the existing `SegmentFrameProjector`, scores query BGE vectors against all ASR segment vectors, and scatters segment scores onto canonical frames with max aggregation. `DenseTemporalScorer` continues to consume a frame-shaped `score_subset()` contract, so Dense fusion, Hybrid fusion, BM25, and monotonic DP remain unchanged.

**Tech Stack:** Python 3.11+, NumPy, existing `SegmentDenseIndex`, existing `SegmentFrameProjector`, existing BGE query encoder, Pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-hybrid-dense-bm25-query-preparation-design.md`

**Related plan:** `docs/superpowers/plans/2026-09-02-hybrid-temporal-dense-bm25.md`

## Global Constraints

- `artifacts/enrichment/transcripts/` remains the ASR source-of-truth; do not rewrite or relocate transcript artifacts.
- `artifacts/indexes/asr_segments/` remains the only Dense ASR artifact; do not build `artifacts/indexes/asr/`.
- `artifacts/enrichment/asr/frame_enrichment.parquet` remains a derived frame-aligned compatibility view used by BM25; do not make Dense temporal depend on it.
- Dense temporal remains `Visual + Context + ASR` with the existing configured weights; do not silently drop ASR when Dense is enabled.
- Context and ASR continue to share the already-created BGE query encoder; do not instantiate a second text encoder.z
- Generic segment-ASR retrieval/RRF stays detached and unchanged.
- BM25 document unit, field routing, artifacts, and scoring stay unchanged.
- `src/hcmai/temporal/dp.py` recurrence/ranking stays frozen.
- `src/hcmai/retrieval/evidence/hybrid.py` fusion behavior stays unchanged.
- No Top-K segment shortlist may be used to produce Dense temporal ASR evidence; Dense temporal must score the complete segment index before projection.
- Projection semantics must reuse `SegmentFrameProjector`: frame inside `[start_ms, end_ms)` first; otherwise nearest midpoint within `asr_projection_max_gap_ms`.
- When multiple ASR segments project to one canonical frame, retain the maximum score for that event/frame.
- Frames with no projected ASR evidence must normalize to zero, not become mid-range evidence.
- Use the full repository checkout for implementation/tests; the source archive is used only to lock the current interfaces.

---

## File Map

**Create:**

- `src/hcmai/retrieval/evidence/asr_projected.py` — adapt `SegmentDenseIndex` to the canonical frame-shaped Dense scoring contract.
- `tests/retrieval/evidence/test_asr_projected.py` — projection, scoring, collision, floor, and validation tests.
- `tests/orchestration/test_asr_projected_loading.py` — runtime wiring and real-artifact guard.

**Modify:**

- `src/hcmai/retrieval/evidence/__init__.py` — export `SegmentProjectedASRIndex`.
- `src/hcmai/orchestration/setup.py` — reuse the already-loaded `ASRSegmentRetriever` instead of loading `IndexConfig.asr_path`.
- `src/hcmai/common/config.py` — remove the obsolete frame-native `IndexConfig.asr_path` configuration surface after setup no longer consumes it.
- `tests/retrieval/evidence/test_dense.py` — prove the projected adapter satisfies `DenseTemporalScorer` without changing the scorer contract.
- `tests/orchestration/test_hybrid_health.py` — update readiness expectations to segment-projected ASR.
- `tests/common/test_hybrid_temporal_config.py` — assert `asr_segment_path` remains canonical and the obsolete frame-native `asr_path` is gone.
- `KNOWLEDGE.md` — record the architecture decision and artifact ownership.

**Do not modify:**

- `src/hcmai/temporal/dp.py`
- `src/hcmai/retrieval/evidence/hybrid.py`
- `src/hcmai/retrieval/retriever/segment/projector.py`
- `src/hcmai/retrieval/retriever/segment/retriever.py`
- `offline/indexes/asr_segment.py`
- `offline/indexes/bm25.py`
- transcript generation/materialization code

---

### Task 1: Add the Segment→Canonical-Frame Adapter and Lock Identity Semantics

**Files:**

- Create: `src/hcmai/retrieval/evidence/asr_projected.py`
- Create: `tests/retrieval/evidence/test_asr_projected.py`
- Modify: `src/hcmai/retrieval/evidence/__init__.py`

**Interfaces:**

- Consumes:
  - `SegmentDenseIndex.mapping`, `.vectors`, `.metadata`
  - `SegmentFrameProjector.project(video_id, start_ms=..., end_ms=...)`
  - canonical visual `DenseIndex.frame_ids`, `.video_ids`, `.frame_idx`, `.timestamps`
- Produces:
  - `SegmentProjectedASRIndex`
  - `.frame_ids: np.ndarray`
  - `.video_ids: np.ndarray`
  - `.frame_idx: np.ndarray`
  - `.timestamps: np.ndarray`
  - `.metadata` exposing the segment embedding dimension
  - `.segment_frame_positions: np.ndarray[int64]`, where `-1` means no valid projection
  - `.score_subset(query_vectors, positions, chunk_size) -> np.ndarray`

- [ ] **Step 1: Write the failing identity/projection tests**

Use a tiny canonical frame fixture with two videos and a tiny segment index fixture. Cover all of these cases explicitly:

```python
def test_projected_asr_mirrors_canonical_identity():
    projected = make_projected_asr()
    np.testing.assert_array_equal(projected.frame_ids, VISUAL_FRAME_IDS)
    np.testing.assert_array_equal(projected.video_ids, VISUAL_VIDEO_IDS)
    np.testing.assert_array_equal(projected.frame_idx, VISUAL_FRAME_IDX)
    np.testing.assert_array_equal(projected.timestamps, VISUAL_TIMESTAMPS)


def test_segment_inside_interval_maps_to_existing_canonical_frame():
    projected = make_projected_asr()
    assert projected.segment_frame_positions[0] == 1


def test_segment_without_frame_inside_uses_nearest_midpoint_within_gap():
    projected = make_projected_asr(max_projection_gap_ms=5_000)
    assert projected.segment_frame_positions[1] == EXPECTED_POSITION


def test_segment_outside_projection_gap_is_unmapped():
    projected = make_projected_asr(max_projection_gap_ms=100)
    assert projected.segment_frame_positions[2] == -1
```

The fixture must use real `SegmentFrameProjector` behavior, not a mocked projection function.

- [ ] **Step 2: Run the new test file and verify failure**

Run:

```bash
PYTHONPATH=.:src aic/bin/pytest tests/retrieval/evidence/test_asr_projected.py -q
```

Expected: FAIL because `hcmai.retrieval.evidence.asr_projected` does not exist.

- [ ] **Step 3: Implement the adapter constructor and projection precomputation**

Create the class with this public shape:

```python
class SegmentProjectedASRIndex:
    def __init__(
        self,
        *,
        segment_index: SegmentDenseIndex,
        canonical_index: DenseIndex,
        projector: SegmentFrameProjector,
    ) -> None:
        ...

    def score_subset(
        self,
        query_vectors: np.ndarray,
        positions: np.ndarray,
        chunk_size: int = 65_536,
    ) -> np.ndarray:
        ...
```

In `__init__`:

```python
self.segment_index = segment_index
self.frame_ids = np.asarray(canonical_index.frame_ids)
self.video_ids = np.asarray(canonical_index.video_ids)
self.frame_idx = np.asarray(canonical_index.frame_idx, dtype=np.int64)
self.timestamps = np.asarray(canonical_index.timestamps, dtype=np.int64)
self.metadata = segment_index.metadata
self._segment_vectors = segment_index.vectors
self.segment_frame_positions = self._build_segment_frame_positions(projector)
```

Precompute one canonical position for each segment:

```python
position_by_frame_id = {
    str(frame_id): position
    for position, frame_id in enumerate(self.frame_ids)
}

mapped = np.full(len(self.segment_index.mapping), -1, dtype=np.int64)
for segment_position, row in self.segment_index.mapping.iterrows():
    projection = projector.project(
        str(row["video_id"]),
        start_ms=int(row["start_ms"]),
        end_ms=int(row["end_ms"]),
    )
    if projection is not None:
        mapped[int(segment_position)] = position_by_frame_id[projection.frame_id]
```

Validate before storing:

- canonical identity arrays all have equal non-zero length;
- canonical `frame_id` values are unique;
- segment vector count equals segment mapping row count;
- every non-`-1` projected position is within canonical bounds;
- every projected `frame_id` returned by the projector exists in the canonical index.

Do **not** create new frame IDs.

- [ ] **Step 4: Export the adapter**

Add to `src/hcmai/retrieval/evidence/__init__.py`:

```python
from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex
```

and add it to `__all__`.

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=.:src aic/bin/pytest tests/retrieval/evidence/test_asr_projected.py -q
```

Expected: projection/identity tests PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  src/hcmai/retrieval/evidence/asr_projected.py \
  src/hcmai/retrieval/evidence/__init__.py \
  tests/retrieval/evidence/test_asr_projected.py
git commit -m "feat: adapt segment ASR scores to canonical frames"
```

---

### Task 2: Implement Full-Segment Dense Scoring and Scatter-Max Projection

**Files:**

- Modify: `src/hcmai/retrieval/evidence/asr_projected.py`
- Modify: `tests/retrieval/evidence/test_asr_projected.py`

**Interfaces:**

- Consumes normalized BGE query vectors shaped `[event_count, embedding_dim]`.
- Produces canonical frame scores shaped `[event_count, len(positions)]`.
- Uses every ASR segment vector; no `SegmentDenseIndex.search(top_k=...)` call is allowed.

- [ ] **Step 1: Write exact synthetic scoring tests**

Add tests proving:

```python
def test_score_subset_scores_all_segments_then_scatter_maxes_collisions():
    # Two segments project to the same frame; larger cosine must win.
    scores = projected.score_subset(query_vectors, all_frame_positions, chunk_size=2)
    assert scores.shape == (len(query_vectors), len(all_frame_positions))
    assert scores[0, COLLISION_FRAME] == pytest.approx(EXPECTED_MAX_SCORE)


def test_uncovered_frames_receive_event_floor():
    scores = projected.score_subset(query_vectors, all_frame_positions)
    assert scores[0, NO_ASR_FRAME] == pytest.approx(scores[0].min())


def test_no_valid_projected_segments_returns_constant_zero_row():
    scores = projected_without_valid_segments.score_subset(query_vectors, all_frame_positions)
    np.testing.assert_array_equal(scores, np.zeros_like(scores))


def test_score_subset_honors_requested_canonical_positions():
    subset = np.array([3, 1], dtype=np.int64)
    scores = projected.score_subset(query_vectors, subset)
    assert scores.shape == (len(query_vectors), 2)
```

- [ ] **Step 2: Run and verify the new scoring tests fail**

Run:

```bash
PYTHONPATH=.:src aic/bin/pytest \
  tests/retrieval/evidence/test_asr_projected.py -q
```

Expected: FAIL because `score_subset()` is not implemented yet.

- [ ] **Step 3: Implement complete-segment cosine scoring in bounded chunks**

Requirements:

- coerce queries to contiguous `float32`;
- accept 1-D query input by reshaping to one row;
- reject wrong embedding dimension;
- reject NaN/Inf query values;
- validate `positions` are integral and within canonical bounds;
- score **all** segment vectors, chunking by `chunk_size` only for memory control.

Use exact matrix multiplication because persisted segment vectors are already normalized:

```python
frame_scores = np.full(
    (len(queries), len(self.frame_ids)),
    -np.inf,
    dtype=np.float32,
)

for start in range(0, len(self._segment_vectors), chunk_size):
    stop = min(start + chunk_size, len(self._segment_vectors))
    vectors = np.asarray(self._segment_vectors[start:stop], dtype=np.float32)
    chunk_scores = queries @ vectors.T
    mapped = self.segment_frame_positions[start:stop]
    valid = mapped >= 0
    if not np.any(valid):
        continue
    target_positions = mapped[valid]
    for event_index in range(len(queries)):
        np.maximum.at(
            frame_scores[event_index],
            target_positions,
            chunk_scores[event_index, valid],
        )
```

- [ ] **Step 4: Fill uncovered frames with a per-event floor**

After all segments have been scored:

```python
for event_index in range(len(queries)):
    covered = np.isfinite(frame_scores[event_index])
    if not np.any(covered):
        frame_scores[event_index].fill(0.0)
        continue
    floor = float(frame_scores[event_index, covered].min())
    frame_scores[event_index, ~covered] = floor
```

Why: `DenseTemporalScorer` applies `minmax_rows()` afterward. Filling missing ASR frames with the minimum observed ASR score guarantees missing evidence normalizes to `0.0` without injecting arbitrary positive evidence.

Return only requested positions:

```python
return np.asarray(frame_scores[:, positions], dtype=np.float32)
```

- [ ] **Step 5: Add a guard proving no top-k segment search is used**

Use a fake `SegmentDenseIndex` whose `.search()` raises if called. `score_subset()` must still PASS by reading `.vectors` directly.

- [ ] **Step 6: Run the full adapter tests**

Run:

```bash
PYTHONPATH=.:src aic/bin/pytest tests/retrieval/evidence/test_asr_projected.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  src/hcmai/retrieval/evidence/asr_projected.py \
  tests/retrieval/evidence/test_asr_projected.py
git commit -m "feat: project full-segment ASR dense scores to frames"
```

---

### Task 3: Prove Compatibility With `DenseTemporalScorer`

**Files:**

- Modify: `tests/retrieval/evidence/test_dense.py`
- Do not modify: `src/hcmai/retrieval/evidence/dense.py` unless a test exposes a real contract bug.

**Interfaces:**

- Existing `DenseTemporalScorer` expects visual/context/ASR objects with identical canonical identity arrays and `.score_subset()`.
- `SegmentProjectedASRIndex.metadata.embedding_dim` must equal the Context BGE dimension.

- [ ] **Step 1: Add a projected-ASR integration test**

Build:

- fake frame-native visual index;
- fake frame-native context index;
- real/tiny `SegmentProjectedASRIndex` wrapping a segment fixture;
- counting visual encoder;
- counting shared BGE encoder.

Assert:

```python
scores = scorer.score_events(["event one", "event two"])
assert scores.shape == (2, FRAME_COUNT)
assert visual_encoder.calls == 1
assert text_encoder.calls == 1
```

Also assert Context and projected-ASR receive the **same BGE query vector batch**.

- [ ] **Step 2: Add dimension mismatch coverage**

```python
def test_dense_rejects_projected_asr_embedding_dimension_mismatch():
    with pytest.raises(ValueError, match="Context and ASR Dense index dimensions differ"):
        DenseTemporalScorer(...)
```

- [ ] **Step 3: Run focused Dense tests**

Run:

```bash
PYTHONPATH=.:src aic/bin/pytest \
  tests/retrieval/evidence/test_dense.py \
  tests/retrieval/evidence/test_asr_projected.py -q
```

Expected: PASS without changes to `dense.py`.

- [ ] **Step 4: Verify `dense.py` remains contract-stable**

Run:

```bash
git diff -- src/hcmai/retrieval/evidence/dense.py
```

Expected: no diff. If a change was truly required, it must only generalize typing/validation and must not change Dense weights, normalization, encoder batching, or fusion formula.

- [ ] **Step 5: Commit tests**

```bash
git add tests/retrieval/evidence/test_dense.py
git commit -m "test: validate projected ASR dense compatibility"
```

---

### Task 4: Rewire Runtime Setup to Reuse Existing ASR Segment Retrieval

**Files:**

- Modify: `src/hcmai/orchestration/setup.py`
- Create: `tests/orchestration/test_asr_projected_loading.py`
- Modify: `tests/orchestration/test_hybrid_health.py`

**Interfaces:**

- Existing retrieval construction already produces:
  - `retrieval.source_retriever(RetrievalSource.CONTEXT)` → frame `ContextRetriever`
  - `retrieval.source_retriever(RetrievalSource.ASR)` → `ASRSegmentRetriever`
- `ASRSegmentRetriever` already owns:
  - `.index: SegmentDenseIndex`
  - `.encoder`: shared BGE adapter
  - `.projector: SegmentFrameProjector`

- [ ] **Step 1: Write failing setup tests for the new dependency path**

Test that `_load_dense_temporal()`:

- never calls `RetrievalService.load_index()` for a frame-native ASR path;
- obtains ASR from `retrieval.source_retriever(RetrievalSource.ASR)`;
- builds `SegmentProjectedASRIndex` from `asr_retriever.index`, `visual.index`, and `asr_retriever.projector`;
- reuses `context.encoder` as the one BGE query encoder;
- reports `asr_ready=True` when the ASR segment retriever exists and projection validation succeeds.

Also test these failure cases:

```python
context missing      -> dense=None, context_ready=False
ASR retriever missing -> dense=None, asr_ready=False
projection validation failure -> dense=None, asr_ready=False
segment/context embedding dimension mismatch -> dense=None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=.:src aic/bin/pytest \
  tests/orchestration/test_asr_projected_loading.py \
  tests/orchestration/test_hybrid_health.py -q
```

Expected: FAIL because setup still loads `HCMAI_ASR_INDEX_PATH/settings.index.asr_path`.

- [ ] **Step 3: Replace the frame-ASR loader in `_load_dense_temporal()`**

Change the current logic from:

```python
asr_path = _runtime_path("HCMAI_ASR_INDEX_PATH", settings.index.asr_path)
asr_index = RetrievalService.load_index(asr_path)
```

into reuse of the already-validated ASR retriever:

```python
context = retrieval.source_retriever(RetrievalSource.CONTEXT)
asr_retriever = retrieval.source_retriever(RetrievalSource.ASR)

context_ready = context is not None
asr_ready = asr_retriever is not None
if not context_ready or not asr_ready:
    ...

projected_asr = SegmentProjectedASRIndex(
    segment_index=asr_retriever.index,
    canonical_index=visual.index,
    projector=asr_retriever.projector,
)

scorer = DenseTemporalScorer(
    visual_index=visual.index,
    context_index=context.index,
    asr_index=projected_asr,
    visual_encoder=visual.encoder,
    text_encoder=context.encoder,
    weights=settings.search.hybrid_temporal.dense,
    chunk_size=settings.search.alignment.chunk_size,
)
```

Add the import:

```python
from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex
```

- [ ] **Step 4: Preserve shared-encoder validation**

Before constructing the scorer, validate that Context and segment-ASR are compatible with the same query embedding family:

```python
if context.index.metadata.embedding_dim != asr_retriever.index.metadata.embedding_dim:
    raise ValueError("Context and ASR segment index dimensions differ")
```

Do not instantiate another encoder. The existing fast-track setup already constructs one shared text encoder for Context and ASR.

- [ ] **Step 5: Update startup/readiness messages**

Remove errors referring to nonexistent `artifacts/indexes/asr`.

Use messages that identify the actual capability, for example:

```text
Dense temporal evidence unavailable: ASR segment retriever missing
Dense temporal ASR projection failed: <ExceptionType>: <message>
```

`capabilities.asr_dense` now means "segment-ASR Dense evidence can be projected to canonical frames", not "frame-native ASR index directory exists".

- [ ] **Step 6: Run orchestration tests**

Run:

```bash
PYTHONPATH=.:src aic/bin/pytest \
  tests/orchestration/test_asr_projected_loading.py \
  tests/orchestration/test_hybrid_health.py \
  tests/orchestration/test_hybrid_temporal_search.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  src/hcmai/orchestration/setup.py \
  tests/orchestration/test_asr_projected_loading.py \
  tests/orchestration/test_hybrid_health.py
git commit -m "fix: use segment ASR for dense temporal evidence"
```

---

### Task 5: Remove the Obsolete Frame-Native ASR Index Configuration Contract

**Files:**

- Modify: `src/hcmai/common/config.py`
- Modify: `tests/common/test_hybrid_temporal_config.py`

**Interfaces:**

- Keep:
  - `IndexConfig.asr_segment_path = Path("artifacts/indexes/asr_segments")`
  - `IndexConfig.asr_projection_max_gap_ms = 5_000`
- Remove:
  - `IndexConfig.asr_path = Path("artifacts/indexes/asr")`
  - runtime use/documentation of `HCMAI_ASR_INDEX_PATH`

- [ ] **Step 1: Write the config regression test**

```python
def test_index_config_uses_segment_asr_without_frame_native_asr_path():
    cfg = IndexConfig()
    assert cfg.asr_segment_path == Path("artifacts/indexes/asr_segments")
    assert cfg.asr_projection_max_gap_ms == 5_000
    assert not hasattr(cfg, "asr_path")
```

- [ ] **Step 2: Run the config test and verify failure**

Run:

```bash
PYTHONPATH=.:src aic/bin/pytest tests/common/test_hybrid_temporal_config.py -q
```

Expected: FAIL because `asr_path` still exists.

- [ ] **Step 3: Remove the obsolete field from `IndexConfig`**

Delete only:

```python
asr_path: Path = Path("artifacts/indexes/asr")
```

Do not remove `asr_segment_path`, `asr_segment_embedding_filename`, or `asr_projection_max_gap_ms`.

- [ ] **Step 4: Remove stale explicit config/env declarations**

Run:

```bash
rg -n "HCMAI_ASR_INDEX_PATH|artifacts/indexes/asr\b|\basr_path\b" \
  src configs scripts thundercompute .env* docs \
  --glob '!**/__pycache__/**'
```

Expected after cleanup:

- no runtime/config reference to `HCMAI_ASR_INDEX_PATH`;
- no `artifacts/indexes/asr` frame-native Dense path;
- references to `artifacts/indexes/asr_segments` remain;
- references to `artifacts/enrichment/asr/frame_enrichment.parquet` remain because BM25/materialization still use them.

Be careful not to delete enrichment config fields such as `dataset.enrichment.asr_path` or `frame_enrichment_path`; those refer to the BM25-compatible frame projection, not the removed Dense index.

- [ ] **Step 5: Run config and setup tests**

```bash
PYTHONPATH=.:src aic/bin/pytest \
  tests/common \
  tests/orchestration/test_asr_projected_loading.py \
  tests/orchestration/test_hybrid_health.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  src/hcmai/common/config.py \
  tests/common/test_hybrid_temporal_config.py
git commit -m "refactor: remove redundant frame ASR index config"
```

---

### Task 6: Real-Artifact Smoke Test, Regression Gate, and Architecture Record

**Files:**

- Modify: `tests/orchestration/test_asr_projected_loading.py`
- Modify: `KNOWLEDGE.md`
- Test: existing Dense/Hybrid/temporal suites

**Interfaces:**

- Production artifacts expected:
  - `artifacts/indexes/visual`
  - `artifacts/indexes/context`
  - `artifacts/indexes/asr_segments`
  - `artifacts/indexes/bm25`
- Dense temporal must not require `artifacts/indexes/asr`.

- [ ] **Step 1: Add a real-artifact identity smoke test guarded by artifact existence**

The test should `pytest.skip()` when production artifacts are not mounted. When present, load:

```python
visual = DenseIndex.load("artifacts/indexes/visual")
context = DenseIndex.load("artifacts/indexes/context")
asr_segments = SegmentDenseIndex.load("artifacts/indexes/asr_segments")
```

Construct canonical frames/projector through the same runtime path used by setup, then build `SegmentProjectedASRIndex`.

Assert:

```python
assert len(projected.frame_ids) == len(visual.frame_ids)
np.testing.assert_array_equal(projected.frame_ids, visual.frame_ids)
assert projected.metadata.embedding_dim == context.metadata.embedding_dim
assert np.any(projected.segment_frame_positions >= 0)
assert np.all(
    (projected.segment_frame_positions == -1)
    | (
        (projected.segment_frame_positions >= 0)
        & (projected.segment_frame_positions < len(visual.frame_ids))
    )
)
```

Do not change projection logic to make this test pass. If it fails, fix artifact lineage/configuration.

- [ ] **Step 2: Run focused evidence/orchestration suites**

```bash
PYTHONPATH=.:src aic/bin/pytest \
  tests/retrieval/evidence \
  tests/orchestration/test_asr_projected_loading.py \
  tests/orchestration/test_hybrid_temporal_search.py \
  tests/orchestration/test_hybrid_health.py \
  tests/temporal -q
```

Expected: PASS; real-artifact test may SKIP only when artifacts are absent.

- [ ] **Step 3: Run the broader KIS/TRAKE/API regression slice**

```bash
PYTHONPATH=.:src aic/bin/pytest \
  tests/orchestration/test_kis_pipeline.py \
  tests/orchestration/test_trake_pipeline.py \
  tests/api/test_hybrid_search_contracts.py \
  tests/api/test_router_inventory.py \
  tests/test_api.py \
  tests/test_api_contracts.py -q
```

Expected: PASS.

- [ ] **Step 4: Compile changed production modules**

```bash
PYTHONPATH=.:src aic/bin/python -m compileall -q \
  src/hcmai/retrieval/evidence \
  src/hcmai/orchestration \
  src/hcmai/common
```

Expected: no errors.

- [ ] **Step 5: Verify forbidden diffs are absent**

```bash
git diff -- \
  src/hcmai/temporal/dp.py \
  src/hcmai/retrieval/evidence/hybrid.py \
  src/hcmai/retrieval/retriever/segment/projector.py \
  src/hcmai/retrieval/retriever/segment/retriever.py \
  offline/indexes/bm25.py \
  offline/indexes/asr_segment.py
```

Expected: no output.

- [ ] **Step 6: Verify the obsolete artifact is no longer part of the runtime contract**

```bash
rg -n "HCMAI_ASR_INDEX_PATH|artifacts/indexes/asr\b|settings\.index\.asr_path" \
  src configs scripts docs .env* \
  --glob '!**/__pycache__/**'
```

Expected: no frame-native Dense-ASR references. `asr_segments` and `enrichment/asr/frame_enrichment.parquet` are allowed and expected.

- [ ] **Step 7: Runtime health smoke test**

Start/load the normal runtime with the existing artifacts and inspect health.

Expected capability state when Visual, Context, ASR segments, BM25, and query preparation are available:

```json
{
  "visual_dense": true,
  "context_dense": true,
  "asr_dense": true,
  "dense_temporal": true,
  "bm25": true,
  "hybrid_temporal": true
}
```

Expected startup messages: no warning about missing `artifacts/indexes/asr`.

- [ ] **Step 8: Record the decision in `KNOWLEDGE.md`**

Append:

```markdown
## Segment-projected ASR Dense temporal

**Date:** 2026-09-02

**Problem:** Dense temporal was wired to a frame-native
`artifacts/indexes/asr` artifact that is not part of the production artifact
pipeline. Production ASR already exists as timestamped transcript segments and a
segment-native Dense index at `artifacts/indexes/asr_segments`.

**Decision:** Reuse the existing `SegmentDenseIndex` and
`SegmentFrameProjector` at runtime. Score each event against all ASR segments,
project segment scores onto canonical visual frames, max-aggregate collisions,
and assign uncovered frames the event floor before the existing per-event
min-max normalization.

**Preserved contracts:** Transcript artifacts, segment-ASR generic retrieval,
BM25 frame-ASR projection, Dense weights, Hybrid fusion, and monotonic DP remain
unchanged.

**Artifact contract:** No frame-native Dense ASR index is required. Dense ASR
uses `artifacts/indexes/asr_segments`; BM25 ASR may continue to use
`artifacts/enrichment/asr/frame_enrichment.parquet`.
```

- [ ] **Step 9: Final commit**

```bash
git add KNOWLEDGE.md tests/orchestration/test_asr_projected_loading.py
git commit -m "docs: lock segment-projected ASR temporal architecture"
```

---

## Acceptance Criteria

The fix is complete only when all of the following are true:

- [ ] `artifacts/indexes/asr` is no longer required, loaded, documented, or checked by runtime health.
- [ ] `artifacts/indexes/asr_segments` is loaded exactly once through the existing fast-track retrieval setup.
- [ ] Dense temporal reuses the existing ASR segment retriever/index/projector rather than constructing a duplicate ASR pipeline.
- [ ] One BGE query encoding batch is reused by Context and ASR Dense scoring for each request.
- [ ] Every ASR segment is considered for Dense temporal scoring; no top-k pre-pruning is introduced before DP.
- [ ] Segment scores are projected only to existing canonical frames using current `SegmentFrameProjector` semantics.
- [ ] Multiple segments projected to one frame use max aggregation.
- [ ] Frames without ASR evidence normalize to zero.
- [ ] Dense output remains `[event_count, canonical_frame_count]`.
- [ ] Dense fusion remains configured Visual/Context/ASR weighting.
- [ ] Hybrid fusion remains Dense/BM25 weighting with no behavioral change.
- [ ] BM25 still consumes frame-aligned ASR text independently from Dense ASR.
- [ ] Generic RRF segment-ASR retrieval remains unchanged.
- [ ] `src/hcmai/temporal/dp.py` has no diff.
- [ ] Runtime health shows `asr_dense=true`, `dense_temporal=true`, and `hybrid_temporal=true` when the existing production artifacts are present.
- [ ] No startup warning mentions missing `artifacts/indexes/asr`.
- [ ] Focused and regression test suites pass.

---

## Implementation Order

Execute strictly in this order:

1. `SegmentProjectedASRIndex` identity/projection contract.
2. Full-segment score + scatter-max + coverage floor.
3. `DenseTemporalScorer` compatibility tests.
4. Runtime setup rewiring to existing `ASRSegmentRetriever`.
5. Remove obsolete frame-native ASR config/path.
6. Real-artifact smoke test + regressions + documentation.

Do **not** start by changing setup/config. The adapter and its deterministic scoring tests must exist first so the runtime wiring has a stable target contract.

---

## Self-Review

- **Spec coverage:** Dense still uses Visual + Context + ASR; segment-ASR generic retrieval stays detached; BM25 remains frame-document-based; DP remains frozen.
- **Artifact compatibility:** Existing `transcripts`, `asr_segments`, BM25, Visual, and Context artifacts are reused. No re-embedding or new production artifact is introduced.
- **Placeholder scan:** No implementation step depends on an undefined future function or artifact.
- **Type consistency:** `SegmentProjectedASRIndex.score_subset(query_vectors, positions, chunk_size)` matches the contract consumed by `DenseTemporalScorer`. Canonical identity names match `DenseIndex`: `frame_ids`, `video_ids`, `frame_idx`, `timestamps`.
- **Risk containment:** All new logic is isolated in one evidence adapter; setup is the only runtime integration point. DP, Hybrid, BM25, segment retrieval, and offline ASR builders remain frozen.

## Experiment Note

This plan fixes an artifact/architecture mismatch; it does **not** claim an accuracy improvement. After correctness is established, benchmark at least:

- Dense Visual + Context, ASR weight disabled.
- Dense Visual + Context + segment-projected ASR.
- Hybrid with segment-projected ASR.

Compare retrieval quality and latency before making any research claim about ASR contribution.
