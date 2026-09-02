# Hybrid Temporal Dense + BM25 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace visual-only temporal evidence with selectable Dense (Visual + Context + frame-ASR), BM25, or 0.5/0.5 hybrid full-corpus evidence while leaving the monotonic DP decoder unchanged.

**Architecture:** Build fielded BM25 offline over canonical frame documents and load it read-only at runtime. A new `TemporalEvidenceScorer` owns Dense source scoring, per-event normalization, language-routed BM25 scoring, and final fusion; `TemporalSearchService` receives its `VideoEventScores` and continues to call the existing `rank_paths()` unchanged.

**Tech Stack:** Python 3.11+, NumPy, Pandas, SciPy sparse matrices, existing FAISS DenseIndex/BGE/SigLIP2 adapters, Pydantic, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-hybrid-dense-bm25-query-preparation-design.md`

## Global Constraints

- Canonical frame is the only BM25 document unit.
- BM25 fields: title, caption, OCR, ASR; objects are excluded from this baseline.
- Corpus artifacts are not translated or rewritten offline.
- BM25 title/OCR/ASR query uses original Vietnamese events; caption uses selected English candidate or literal English translation.
- Dense block requires Visual SigLIP2 + Context BGE + frame-ASR BGE; no selected Dense source is silently omitted.
- Segment-ASR generic retriever remains detached and unchanged.
- All source rows use per-event min-max normalization; constant rows become all zeros.
- Dense internal weights default to 1/3 each and sum to 1.0.
- Hybrid weights default Dense=0.5, BM25=0.5 and sum to 1.0 when both are enabled.
- No RRF, rank conversion, Top-K shortlist, filter, or candidate-video pruning before DP.
- `hcmai/temporal/dp.py` recurrence and ranking behavior are frozen.
- At least one of `use_dense` / `use_bm25` must be true.
- Execution prerequisite: use the full repository checkout containing the existing `offline/` package. The reviewed `src_hcmai_v6.zip` contains runtime source only; do not create a second competing offline root if the real repository already has one.

---

## File Map

**Create:**
- `src/hcmai/retrieval/evidence/__init__.py`
- `src/hcmai/retrieval/evidence/normalization.py`
- `src/hcmai/retrieval/evidence/dense.py`
- `src/hcmai/retrieval/evidence/bm25.py`
- `src/hcmai/retrieval/evidence/hybrid.py`
- `offline/indexes/bm25.py`
- `tests/retrieval/evidence/test_normalization.py`
- `tests/retrieval/evidence/test_dense.py`
- `tests/retrieval/evidence/test_bm25.py`
- `tests/retrieval/evidence/test_hybrid.py`
- `tests/offline/indexes/test_bm25_builder.py`
- `tests/orchestration/test_hybrid_temporal_search.py`
- `tests/api/test_hybrid_search_contracts.py`

**Modify:**
- `src/hcmai/common/config.py`
- `src/hcmai/orchestration/setup.py`
- `src/hcmai/orchestration/pipeline.py`
- `src/hcmai/orchestration/temporal_search.py`
- `src/hcmai/orchestration/workflows/kis.py`
- `src/hcmai/orchestration/workflows/trake.py`
- `src/hcmai/api/contracts/search.py`
- `src/hcmai/api/contracts/trake.py`
- `src/hcmai/api/routers/filter.py` — delete.
- `src/hcmai/api/routers/__init__.py`
- `src/hcmai/app.py`
- `pyproject.toml` — add SciPy if it is not already a direct runtime/offline dependency.

---

### Task 1: Lock Hybrid/BM25 Configuration and Validation

**Files:**
- Modify: `src/hcmai/common/config.py`
- Test: `tests/common/test_hybrid_temporal_config.py`

**Interfaces:**
- Produces `DenseTemporalWeights`, `BM25FieldWeights`, `HybridTemporalConfig` and `IndexConfig.bm25_path`.

- [ ] **Step 1: Write validation tests**

```python
import pytest
from pydantic import ValidationError
from hcmai.common.config import HybridTemporalConfig


def test_hybrid_defaults():
    cfg = HybridTemporalConfig()
    assert cfg.dense.visual_weight == pytest.approx(1 / 3)
    assert cfg.dense.context_weight == pytest.approx(1 / 3)
    assert cfg.dense.asr_weight == pytest.approx(1 / 3)
    assert cfg.dense_weight == pytest.approx(0.5)
    assert cfg.bm25_weight == pytest.approx(0.5)


def test_dense_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        HybridTemporalConfig(dense={"visual_weight": 1, "context_weight": 1, "asr_weight": 1})
```

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONPATH=src pytest tests/common/test_hybrid_temporal_config.py -v`

Expected: FAIL with missing config.

- [ ] **Step 3: Implement config models**

Use Pydantic `model_validator(mode="after")` with `math.isclose(total, 1.0, abs_tol=1e-6)`. Add:

```python
bm25_path: Path = Path("artifacts/indexes/bm25")
```

to `IndexConfig`, and add `hybrid_temporal: HybridTemporalConfig` beside the existing `alignment` config; do not reuse `FusionConfig`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/common/test_hybrid_temporal_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/common/config.py tests/common/test_hybrid_temporal_config.py
git commit -m "feat: configure hybrid temporal evidence"
```

---

### Task 2: Implement Deterministic BM25 Tokenization and Offline Artifact Builder

**Files:**
- Create: `offline/indexes/bm25.py`
- Modify: `pyproject.toml` if `scipy` is not already declared.
- Test: `tests/offline/indexes/test_bm25_builder.py`

**Interfaces:**
- Produces artifact directory at `IndexConfig.bm25_path` with:
  - `frame_mapping.parquet`
  - `metadata.json`
  - `title_vocab.json`, `title_weights.npz`
  - `caption_vocab.json`, `caption_weights.npz`
  - `ocr_vocab.json`, `ocr_weights.npz`
  - `asr_vocab.json`, `asr_weights.npz`
- Sparse matrices are CSR, shape `[document_count, vocabulary_size]`, storing precomputed BM25 document-term contributions.
- `offline.indexes.bm25` exposes a module CLI with `--frames`, `--caption`, `--ocr`, `--asr`, `--media-info`, `--output`, and `--dataset-version`; default production values remain the current artifact paths.

- [ ] **Step 1: Write tokenizer tests**

Use examples proving Unicode/lowercase/punctuation behavior and preservation of alphanumeric tokens such as `HTV`, `VTV24`, `X`, `60`.

```python
def test_tokenize_keeps_alphanumeric_tokens():
    assert tokenize("HTV: 60 Giây Sáng, X!") == ["htv", "60", "giây", "sáng", "x"]
```

- [ ] **Step 2: Write tiny-corpus artifact test**

Construct three canonical frames with mixed fields and assert `frame_mapping.parquet` preserves `frame_id/video_id/frame_idx/timestamp_ms`, missing evidence becomes empty text, and all four sparse matrices have exactly three rows.

- [ ] **Step 3: Run and verify failure**

Run: `PYTHONPATH=.:src pytest tests/offline/indexes/test_bm25_builder.py -v`

Expected: FAIL because builder is absent.

- [ ] **Step 4: Implement tokenizer**

Use Unicode NFKC normalization, lowercase, replace non-word/non-alphanumeric punctuation boundaries with spaces, split whitespace, and drop empty tokens. Do not stem or invoke a Vietnamese segmenter.

- [ ] **Step 5: Implement BM25 document-term weighting**

For each field independently use baseline constants `k1=1.5`, `b=0.75` and:

```text
idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
weight(d,t) = idf(t) * tf(d,t)*(k1+1) / (tf(d,t) + k1*(1-b+b*dl(d)/avgdl))
```

Store the constants and tokenizer/schema versions in metadata. Query-time scoring can then sum the sparse columns for query terms without rescanning documents.

- [ ] **Step 6: Join only existing artifacts**

Build documents from canonical frames plus current title/caption/OCR/frame-ASR sources. Never write translated caption/VI artifacts. Missing source values become `""`.

- [ ] **Step 7: Publish atomically using the repository's existing directory-publication helper**

Build in a sibling staging directory, write metadata last, then use `publish_directory()` so runtime never sees a partial bundle.

- [ ] **Step 8: Add and test the module CLI**

The parser must call the same builder function tested above; it must not duplicate build logic. Verify help first:

```bash
PYTHONPATH=.:src python -m offline.indexes.bm25 --help
```

Then run: `PYTHONPATH=.:src pytest tests/offline/indexes/test_bm25_builder.py -v`

Expected: help exits 0 and tests PASS.

- [ ] **Step 9: Commit**

```bash
git add offline/indexes/bm25.py pyproject.toml tests/offline/indexes/test_bm25_builder.py
git commit -m "feat: build canonical frame bm25 artifact"
```

---

### Task 3: Load and Score the BM25 Artifact at Runtime

**Files:**
- Create: `src/hcmai/retrieval/evidence/bm25.py`
- Create: `src/hcmai/retrieval/evidence/__init__.py`
- Test: `tests/retrieval/evidence/test_bm25.py`

**Interfaces:**
- Produces `BM25TemporalScorer.score_events(original_events, caption_events) -> np.ndarray[event_count, canonical_frame_count]`.
- Loader validates/reorders artifact mapping once against canonical visual-index mapping.

- [ ] **Step 1: Write language-routing and weighting tests**

Tiny corpus must prove:
- `"htv"` in Vietnamese original influences OCR/title/ASR fields.
- English `"white apron"` influences caption only.
- an English caption token passed only in Vietnamese fields does not accidentally score caption.
- missing fields contribute 0.

- [ ] **Step 2: Write identity mismatch test**

Change one artifact `frame_idx` or `timestamp_ms` while keeping `frame_id`; loading must raise an artifact identity error before search.

- [ ] **Step 3: Run and verify failure**

Run: `PYTHONPATH=src pytest tests/retrieval/evidence/test_bm25.py -v`

Expected: FAIL with missing module.

- [ ] **Step 4: Implement read-only loader**

Load mapping, metadata, vocabularies and CSR matrices. Build one reorder array from artifact positions to canonical visual positions and verify all four identity columns exactly.

- [ ] **Step 5: Implement full-corpus field scoring**

For each event, tokenize original VI once and caption EN once. Sum matching sparse term columns, multiply field scores by configured field weights, and return `float32` full-corpus rows.

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=src pytest tests/retrieval/evidence/test_bm25.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hcmai/retrieval/evidence tests/retrieval/evidence/test_bm25.py
git commit -m "feat: score fielded bm25 temporal evidence"
```

---

### Task 4: Add Per-Event Normalization

**Files:**
- Create: `src/hcmai/retrieval/evidence/normalization.py`
- Test: `tests/retrieval/evidence/test_normalization.py`

**Interfaces:**
- Produces `minmax_rows(scores: np.ndarray) -> np.ndarray` preserving shape/dtype-compatible float values.

- [ ] **Step 1: Write exact synthetic tests**

```python
def test_minmax_rows_normalizes_each_event_independently():
    actual = minmax_rows(np.array([[2., 4., 6.], [10., 20., 30.]], dtype=np.float32))
    np.testing.assert_allclose(actual, [[0., .5, 1.], [0., .5, 1.]])


def test_constant_row_becomes_zero():
    np.testing.assert_array_equal(minmax_rows(np.array([[7., 7.]], dtype=np.float32)), [[0., 0.]])
```

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONPATH=src pytest tests/retrieval/evidence/test_normalization.py -v`

Expected: FAIL.

- [ ] **Step 3: Implement vectorized normalization without NaNs**

Compute row min/max with `keepdims=True`; divide only where span > 0 and leave constant rows zero.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=src pytest tests/retrieval/evidence/test_normalization.py -v
git add src/hcmai/retrieval/evidence/normalization.py tests/retrieval/evidence/test_normalization.py
git commit -m "feat: normalize temporal evidence rows"
```

---

### Task 5: Implement Full-Corpus Dense Temporal Scoring With Shared Encodings

**Files:**
- Create: `src/hcmai/retrieval/evidence/dense.py`
- Modify: `src/hcmai/orchestration/setup.py`
- Test: `tests/retrieval/evidence/test_dense.py`

**Interfaces:**
- `DenseTemporalScorer.score_events(retrieval_events) -> np.ndarray[event_count, canonical_frame_count]`.
- Owns references to visual, context and frame-ASR DenseIndexes plus existing visual/text query encoders; it does not construct duplicate encoder instances.

- [ ] **Step 1: Write a test that counts encoder calls**

Use fake encoders/indexes and assert one SigLIP encoding batch and one BGE encoding batch per request, with the BGE vector batch reused for both context and ASR indexes.

- [ ] **Step 2: Write mapping/capability tests**

All three indexes must have identical canonical `frame_id/video_id/frame_idx/timestamp_ms` order. Missing context or frame-ASR makes `dense_temporal` unavailable instead of falling back to visual-only.

- [ ] **Step 3: Run and verify failure**

Run: `PYTHONPATH=src pytest tests/retrieval/evidence/test_dense.py -v`

Expected: FAIL.

- [ ] **Step 4: Implement scorer using existing `DenseIndex.score_subset()`**

Score `positions = np.arange(frame_count)` in configured chunks. Normalize each source with `minmax_rows`, then compute:

```python
D = (
    cfg.visual_weight * visual_norm
    + cfg.context_weight * context_norm
    + cfg.asr_weight * asr_norm
)
```

- [ ] **Step 5: Extend setup to load `IndexConfig.asr_path` as a frame-native DenseIndex**

Do not replace or remove existing `asr_segment_path`; generic RRF keeps using segment-ASR. Reuse the already-created evidence BGE adapter for Context + frame-ASR queries.

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=src pytest tests/retrieval/evidence/test_dense.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/hcmai/retrieval/evidence/dense.py src/hcmai/orchestration/setup.py tests/retrieval/evidence/test_dense.py
git commit -m "feat: score multimodal dense temporal evidence"
```

---

### Task 6: Implement Hybrid Evidence Routing and Fusion

**Files:**
- Create: `src/hcmai/retrieval/evidence/hybrid.py`
- Test: `tests/retrieval/evidence/test_hybrid.py`

**Interfaces:**
- Produces `TemporalEvidenceScorer.score_events(original_events, retrieval_events, *, caption_events, use_dense, use_bm25) -> list[VideoEventScores]`.
- `caption_events` is already resolved to English by orchestration; scorer does not call Qwen.

- [ ] **Step 1: Write three-mode fusion tests**

For synthetic matrices `D` and `B`, assert Dense-only=`D`, BM25-only=`minmax_rows(B_raw)`, both=`0.5*D + 0.5*B_norm` under defaults.

- [ ] **Step 2: Write both-off and event-count mismatch tests**

Both toggles false and mismatched original/retrieval/caption event lengths must raise explicit validation errors before scoring.

- [ ] **Step 3: Run and verify failure**

Run: `PYTHONPATH=src pytest tests/retrieval/evidence/test_hybrid.py -v`

Expected: FAIL.

- [ ] **Step 4: Implement scorer and split fused canonical matrix into `VideoEventScores`**

Use the canonical visual mapping to create per-video arrays; preserve the same sorted-video and frame ordering contract currently produced by `score_all_videos()`.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONPATH=src pytest tests/retrieval/evidence/test_hybrid.py -v
git add src/hcmai/retrieval/evidence/hybrid.py tests/retrieval/evidence/test_hybrid.py
git commit -m "feat: fuse dense and bm25 temporal evidence"
```

---

### Task 7: Route Original/Candidate Queries Through TemporalSearchService Without Touching DP

**Files:**
- Modify: `src/hcmai/orchestration/temporal_search.py`
- Test: `tests/orchestration/test_hybrid_temporal_search.py`
- Test: existing/new `tests/temporal/test_dp.py`

**Interfaces:**
- New search signature:

```python
def search(
    self,
    original_events: Sequence[str],
    *,
    retrieval_events: Sequence[str] | None,
    caption_events: Sequence[str] | None,
    use_dense: bool,
    use_bm25: bool,
    top_k: int,
) -> TemporalSearchResult: ...
```

- `retrieval_events=None` resolves to original events for Dense.
- `caption_events` must be present whenever BM25 is on by the time this method calls the evidence scorer.

- [ ] **Step 1: Write orchestration tests with a fake evidence scorer**

Assert exact routed event arrays reach the scorer and the returned fused `VideoEventScores` reach `rank_paths()` unchanged.

- [ ] **Step 2: Characterize DP before editing orchestration**

Run existing monotonic ordering, gap penalty, full alignment, multiple-same-video path, and level-wise ranking tests. Save output in the implementation notes.

- [ ] **Step 3: Replace direct `RetrievalService.score_event_videos()` dependency with `TemporalEvidenceScorer`**

Do not edit recurrence code in `src/hcmai/temporal/dp.py`.

- [ ] **Step 4: Keep existing canonical identity validation/materialization**

Fused scores must still be checked against Corpus frame identity before `AlignedPath` creation.

- [ ] **Step 5: Run orchestration + DP tests**

```bash
PYTHONPATH=src pytest tests/orchestration/test_hybrid_temporal_search.py tests/temporal -v
```

Expected: PASS with unchanged DP characterization.

- [ ] **Step 6: Commit**

```bash
git add src/hcmai/orchestration/temporal_search.py tests/orchestration/test_hybrid_temporal_search.py tests/temporal
git commit -m "refactor: feed hybrid evidence into temporal dp"
```

---

### Task 8: Extend KIS/TRAKE Public Contracts and Resolve English Caption Events Lazily

**Files:**
- Modify: `src/hcmai/api/contracts/search.py`
- Modify: `src/hcmai/api/contracts/trake.py`
- Modify: `src/hcmai/orchestration/workflows/kis.py`
- Modify: `src/hcmai/orchestration/workflows/trake.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Test: `tests/api/test_hybrid_search_contracts.py`
- Test: `tests/orchestration/test_hybrid_query_routing.py`

**Interfaces:**
- KIS request: `query`, optional `retrieval_events`, `use_dense=True`, `use_bm25=True`, `top_k`.
- TRAKE request: original `events`, optional `retrieval_events`, same toggles/top_k.
- Responses add `dense_events`, `bm25_caption_events`, `use_dense`, `use_bm25` while preserving existing result/path schemas.

- [ ] **Step 1: Write validation tests**

Reject both toggles OFF and candidate event-count mismatch. Keep existing Top-K constraints.

- [ ] **Step 2: Write lazy-translation routing tests**

Cover all required cases:

```text
Original + Dense only: no QueryPreparationService call; dense_events=VI; bm25_caption_events=None
Original + BM25: translate_literal(VI); BM25 VI fields=VI; caption=literal EN
Candidate + Dense only: Dense=candidate EN; no translation
Candidate + BM25: BM25 VI fields=original VI; caption=candidate EN; no translation
```

- [ ] **Step 3: Run and verify failures**

Run: `PYTHONPATH=src pytest tests/api/test_hybrid_search_contracts.py tests/orchestration/test_hybrid_query_routing.py -v`

Expected: FAIL against old request/workflow signatures.

- [ ] **Step 4: Extend request/response contracts**

Use request model validators to enforce both-off and event-count invariants. Do not add candidate IDs.

- [ ] **Step 5: Resolve caption events in orchestration, not inside BM25 scorer**

Only call `QueryPreparationService.translate_literal()` if `use_bm25=True` and `retrieval_events is None`. If the service is unavailable in this exact case, raise `SearchServiceUnavailableError` naming query preparation; do not disable caption silently.

- [ ] **Step 6: Update KIS and TRAKE workflows to pass both original and selected representations**

KIS still uses `split_query_events(query)` for original events. TRAKE preserves its explicit event list.

- [ ] **Step 7: Run tests**

Run: `PYTHONPATH=src pytest tests/api/test_hybrid_search_contracts.py tests/orchestration/test_hybrid_query_routing.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/hcmai/api/contracts src/hcmai/orchestration tests/api/test_hybrid_search_contracts.py tests/orchestration/test_hybrid_query_routing.py
git commit -m "feat: expose hybrid temporal retrieval modes"
```

---

### Task 9: Wire Runtime Capabilities, Remove Backend Filter Placeholder, and Preserve Detached RRF

**Files:**
- Modify: `src/hcmai/orchestration/setup.py`
- Modify: `src/hcmai/orchestration/pipeline.py`
- Delete: `src/hcmai/api/routers/filter.py`
- Modify: `src/hcmai/api/routers/__init__.py`
- Modify: `src/hcmai/app.py`
- Test: `tests/orchestration/test_hybrid_health.py`
- Test: `tests/api/test_router_inventory.py`

**Interfaces:**
- Health adds `bm25`, `visual_dense`, `context_dense`, `asr_dense`, `dense_temporal`, `hybrid_temporal`.
- `/api/v1/filter` no longer exists.
- Existing generic `RetrievalService`/RRF construction remains available for detached experiments.

- [ ] **Step 1: Write health capability tests for missing combinations**

Dense ready only when all three Dense source indexes are ready. Hybrid ready only when Dense and BM25 are ready. Query-preparation health remains independent from Plan A.

- [ ] **Step 2: Write route inventory test**

Assert `/api/v1/filter` is absent while `/api/v1/search`, `/api/v1/trake`, and `/api/v1/query-candidates` remain.

- [ ] **Step 3: Run and verify failures**

Run: `PYTHONPATH=src pytest tests/orchestration/test_hybrid_health.py tests/api/test_router_inventory.py -v`

Expected: FAIL until wiring/removal is complete.

- [ ] **Step 4: Load BM25 and DenseTemporalScorer independently of generic RRF**

A missing BM25 artifact disables BM25/hybrid capability but does not prevent Dense-only startup. Missing one Dense source disables Dense/hybrid but does not prevent BM25-only startup.

- [ ] **Step 5: Delete backend filter router and mount**

Remove imports/registration, not just hide the route.

- [ ] **Step 6: Verify generic RRF code remains**

Run:

```bash
test -f src/hcmai/retrieval/retriever/fusion/rrf.py
rg -n "RRFFusionRetriever|rrf" src/hcmai/retrieval
```

Expected: detached RRF implementation still exists.

- [ ] **Step 7: Run tests and commit**

```bash
PYTHONPATH=src pytest tests/orchestration/test_hybrid_health.py tests/api/test_router_inventory.py -v
git add -A src/hcmai tests/orchestration/test_hybrid_health.py tests/api/test_router_inventory.py
git commit -m "feat: wire hybrid evidence capabilities"
```

---

### Task 10: Hybrid Temporal Acceptance and Ablation Gate

**Files:**
- No production changes unless verification finds a defect.

- [ ] **Step 1: Compile runtime/offline code**

```bash
PYTHONPATH=.:src python -m compileall -q src/hcmai/retrieval/evidence src/hcmai/orchestration offline/indexes
```

Expected: exit 0.

- [ ] **Step 2: Run full new backend slice plus temporal regression**

```bash
PYTHONPATH=.:src pytest   tests/retrieval/evidence   tests/offline/indexes/test_bm25_builder.py   tests/orchestration/test_hybrid_temporal_search.py   tests/orchestration/test_hybrid_query_routing.py   tests/orchestration/test_hybrid_health.py   tests/api/test_hybrid_search_contracts.py   tests/api/test_router_inventory.py   tests/temporal -v
```

Expected: all PASS.

- [ ] **Step 3: Build BM25 against one existing corpus and reload it**

Run the concrete offline builder:

```bash
PYTHONPATH=.:src python -m offline.indexes.bm25 \
  --frames artifacts/frame_store/frames.parquet \
  --caption artifacts/corpus/caption.parquet \
  --ocr artifacts/corpus/ocr_frames.parquet \
  --asr artifacts/enrichment/asr/frame_enrichment.parquet \
  --media-info data/media-info \
  --output artifacts/indexes/bm25 \
  --dataset-version hcmai2026_v1
```

After build, start runtime and verify health reports `bm25=true` and no frame identity mismatch.

- [ ] **Step 4: Run one fixed query through all six ablations**

Record result paths/scores for:

```text
A Dense original VI
B BM25 original VI + literal EN caption
C Hybrid original
D Dense candidate #k
E BM25 candidate #k
F Hybrid candidate #k
```

This is a functionality/observability gate, not a claim that hybrid is more accurate.

- [ ] **Step 5: Verify DP source was not modified**

Run:

```bash
BASE_SHA="$(git merge-base origin/main HEAD)"
git diff "$BASE_SHA"..HEAD -- src/hcmai/temporal/dp.py
```

Expected: empty diff. If the feature branch is not based on `origin/main`, substitute the actual branch base ref in the `git merge-base` command before running it.
