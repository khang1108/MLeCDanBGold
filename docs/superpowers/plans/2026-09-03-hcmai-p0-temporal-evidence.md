# HCM-AI P0 Temporal Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v9's early fixed-weight/noise-amplifying temporal evidence fusion with inspectable, robustly calibrated, availability-aware multimodal emission scoring while keeping the existing DP recurrence and all corpus/index artifacts unchanged.

**Architecture:** P0 turns each retrieval source into an independent `[event, frame]` component, calibrates each row without forcing weak evidence to look confident, attaches modality coverage/reliability, and fuses components with lightweight event-dependent priors. The output remains the existing `list[VideoEventScores]`, so `TemporalSearchService` and `src/hcmai/temporal/dp.py` remain unchanged. A `legacy` mode must numerically reproduce the current v9 behavior for rollback and ablation.

**Tech Stack:** Python 3, NumPy, SciPy sparse matrices, pandas, Pydantic config models, existing SigLIP2/BGE query encoders, existing Dense/BM25/ASR artifacts, pytest for new regression tests.

**Spec:** `docs/superpowers/specs/2026-09-03-hcmai-p0-temporal-evidence-design.md`

## Global Constraints

- Query preparation is frozen; P0 consumes already prepared dynamic events `E1..EN` and must never assume a fixed event count.
- Do not regenerate Caption/OCR/Object/ASR/Context artifacts.
- Do not rebuild FAISS/BM25 indexes.
- Do not change artifact paths, filenames, schemas, or manifests.
- Do not reintroduce runtime Qwen candidate generation.
- Do not change KIS/TRAKE public request/response contracts.
- Do not modify the recurrence or semantics of `src/hcmai/temporal/dp.py` in P0.
- Preserve canonical visual-index identity and the existing `VideoEventScores` contract.
- `fusion_mode="legacy"` must reproduce current v9 scoring under `np.allclose(actual_scores, expected_v9_scores, rtol=1e-6, atol=1e-6)` when Visual, Context, ASR, and BM25 are all available.
- Every adaptive feature must be independently disableable so A/B ablations can isolate its effect.
- No learned gating model, no new encoder, no full-corpus VLM reranker, and no DP soft-order work belong in P0.

---

## File Structure Before Implementation

### New files

- `src/hcmai/retrieval/evidence/components.py`
  - Defines immutable score-component and score-bundle contracts.
- `src/hcmai/retrieval/evidence/calibration.py`
  - Owns robust row calibration and deterministic reliability estimation.
- `src/hcmai/retrieval/evidence/fusion.py`
  - Owns event cue routing and adaptive availability-aware fusion.
- `src/hcmai/retrieval/evidence/diagnostics.py`
  - Converts component/calibration/fusion state into compact debug records without changing public API responses.
- `tests/retrieval/evidence/fakes.py`
  - Small fake indexes/encoders for deterministic scorer tests.
- `tests/retrieval/evidence/test_legacy_characterization.py`
- `tests/retrieval/evidence/test_components.py`
- `tests/retrieval/evidence/test_calibration.py`
- `tests/retrieval/evidence/test_asr_interval.py`
- `tests/retrieval/evidence/test_fusion.py`
- `tests/orchestration/test_temporal_evidence_setup.py`
- `scripts/debug_temporal_evidence.py`
  - Offline diagnostic entry point for one prepared query and selected video IDs.

### Existing files modified

- `src/hcmai/retrieval/evidence/dense.py:17-80`
  - Add independent raw component scoring; retain legacy wrapper.
- `src/hcmai/retrieval/evidence/bm25.py:32-143`
  - Expose title/caption/OCR/ASR scores separately; retain legacy weighted sum.
- `src/hcmai/retrieval/evidence/asr_projected.py:24-233`
  - Replace one-frame-only ASR semantics with interval coverage plus deterministic fallback.
- `src/hcmai/retrieval/evidence/hybrid.py:19-102`
  - Add `legacy` vs `adaptive_p0` execution path; keep final `VideoEventScores` output.
- `src/hcmai/retrieval/evidence/__init__.py`
  - Export new contracts needed by orchestration/tests.
- `src/hcmai/orchestration/setup.py:88-231`
  - Make Context and ASR independently optional instead of disabling Dense entirely.
- `src/hcmai/common/config.py:316-362`
  - Add P0 calibration/fusion config while retaining all legacy weight fields.

### Files intentionally not modified

- `src/hcmai/temporal/dp.py`
- `src/hcmai/orchestration/temporal_search.py`
- `src/hcmai/orchestration/workflows/kis.py`
- `src/hcmai/orchestration/workflows/trake.py`

If implementation appears to require changing these four files, stop and review the boundary before proceeding. P0 is designed so they do not need behavioral changes.

---

## P0 Data Flow and Mathematical Contract

For `N` prepared events and `F` canonical corpus frames, each available source produces raw scores:

```text
R_visual       in R^(N x F)
R_context      in R^(N x F)
R_asr_dense    in R^(N x F)
R_bm25_title   in R^(N x F)
R_bm25_caption in R^(N x F)
R_bm25_ocr     in R^(N x F)
R_bm25_asr     in R^(N x F)
```

For each component `m`, robust calibration produces:

```text
C_m[e, f] in [0, 1]
reliability_m[e] in [0, 1]
coverage_m[f] in {False, True} or None for full coverage
```

The adaptive P0 fusion computes event-specific requested weights from a configured base weight and cue multipliers:

```text
requested[e, m] = base[m] * cue_multiplier(event_e, m)
```

Then per frame:

```text
effective[e, m, f]
  = requested[e, m]
  * reliability_m[e]          # if confidence gating enabled, else 1
  * coverage_m[f]             # if coverage exists, else 1
```

Final emission:

```text
S[e, f]
  = sum_m effective[e,m,f] * C_m[e,f]
    / sum_m effective[e,m,f]
```

If the denominator is zero at one `(event, frame)`, use calibrated Visual evidence when Visual exists; otherwise return `0.0`. Visual is expected to exist at runtime because canonical identity comes from the visual index.

The existing `_split_videos()` then returns the same `VideoEventScores` objects consumed by DP.

---

### Task 1: Add a regression test harness and freeze current v9 behavior

**Why this task exists:** Current v9 has no repository test suite around temporal evidence. Before changing scoring, capture the exact legacy behavior so later refactors can prove rollback equivalence rather than merely "look similar".

**Files:**
- Create: `tests/retrieval/evidence/fakes.py`
- Create: `tests/retrieval/evidence/test_legacy_characterization.py`
- Create: `tests/orchestration/test_temporal_evidence_setup.py`
- Read only: `src/hcmai/retrieval/evidence/normalization.py:8-22`
- Read only: `src/hcmai/retrieval/evidence/dense.py:17-80`
- Read only: `src/hcmai/retrieval/evidence/hybrid.py:19-102`
- Read only: `src/hcmai/orchestration/setup.py:121-209`

**Interfaces:**
- Consumes: existing `DenseTemporalScorer.score_events()`, `TemporalEvidenceScorer.score_events()`, `minmax_rows()`.
- Produces: deterministic fake indexes/encoders reused by every later task; numerical regression assertions for legacy mode.

- [ ] **Step 1: Create deterministic fake index/encoder helpers**

Add `tests/retrieval/evidence/fakes.py` with focused fakes matching only attributes the production scorers actually use:

```python
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np


@dataclass
class FakeIndex:
    scores: np.ndarray
    embedding_dim: int = 2

    def __post_init__(self) -> None:
        frame_count = self.scores.shape[1]
        self.frame_ids = np.asarray([f"f{i}" for i in range(frame_count)])
        self.video_ids = np.asarray(["v1"] * frame_count)
        self.frame_idx = np.arange(frame_count, dtype=np.int64)
        self.timestamps = np.arange(frame_count, dtype=np.int64) * 1000
        self.metadata = SimpleNamespace(embedding_dim=self.embedding_dim)

    def score_subset(
        self,
        query_vectors: np.ndarray,
        positions: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray:
        del query_vectors, chunk_size
        return np.asarray(self.scores[:, positions], dtype=np.float32)

    def video_positions(self, video_id: str) -> np.ndarray:
        return np.flatnonzero(self.video_ids == video_id)


class FakeEncoder:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.calls: list[tuple[str, ...]] = []

    def encode_text(self, events: list[str]) -> np.ndarray:
        self.calls.append(tuple(events))
        return self.vectors[: len(events)]
```

- [ ] **Step 2: Write characterization tests for `minmax_rows()`**

Create `tests/retrieval/evidence/test_legacy_characterization.py` and assert current semantics explicitly:

```python
import numpy as np

from hcmai.retrieval.evidence.normalization import minmax_rows


def test_minmax_rows_stretches_each_nonconstant_event_independently() -> None:
    raw = np.asarray([[0.20, 0.21, 0.22], [10.0, 20.0, 30.0]], dtype=np.float32)

    actual = minmax_rows(raw)

    np.testing.assert_allclose(
        actual,
        np.asarray([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32),
    )


def test_minmax_rows_turns_constant_event_into_zero() -> None:
    actual = minmax_rows(np.asarray([[0.3, 0.3, 0.3]], dtype=np.float32))
    np.testing.assert_array_equal(actual, np.zeros((1, 3), dtype=np.float32))
```

- [ ] **Step 3: Run the normalization characterization tests**

Run:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/retrieval/evidence/test_legacy_characterization.py
```

Expected: PASS against untouched v9.

- [ ] **Step 4: Add a Dense legacy-equivalence test**

Use raw rows where every modality has a different scale so the test proves that v9 normalizes first and weights second:

```python
import numpy as np
from hcmai.common.config import DenseTemporalWeights
from hcmai.retrieval.evidence.dense import DenseTemporalScorer
from tests.retrieval.evidence.fakes import FakeEncoder, FakeIndex


def test_dense_legacy_normalizes_each_modality_then_averages() -> None:
    visual = FakeIndex(np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32))
    context = FakeIndex(np.asarray([[10.0, 10.0, 12.0]], dtype=np.float32))
    asr = FakeIndex(np.asarray([[0.70, 0.71, 0.72]], dtype=np.float32))
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    scorer = DenseTemporalScorer(
        visual_index=visual,
        context_index=context,
        asr_index=asr,
        visual_encoder=encoder,
        text_encoder=encoder,
        weights=DenseTemporalWeights(),
        chunk_size=8,
    )

    actual = scorer.score_events(["event"])

    expected = np.asarray([[0.0, 1.0 / 6.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
```

- [ ] **Step 5: Add a hybrid legacy-equivalence test**

Create a tiny fake Dense/BM25 pair and verify current `dense_weight=0.5`, `bm25_weight=0.5` semantics. The expected value must be computed explicitly, not by calling production normalization in the test:

```python
class StaticDense:
    def score_events(self, events):
        del events
        return np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)


class StaticBM25:
    def score_events(self, original, caption):
        del original, caption
        return np.asarray([[2.0, 3.0, 6.0]], dtype=np.float32)


def test_hybrid_legacy_minmaxes_bm25_then_uses_half_half_weights():
    visual = FakeIndex(np.zeros((1, 3), dtype=np.float32))
    scorer = TemporalEvidenceScorer(
        visual_index=visual,
        dense=StaticDense(),
        bm25=StaticBM25(),
        config=HybridTemporalConfig(),
    )

    videos = scorer.score_events(
        ["vi"],
        ["en"],
        caption_events=["vi caption"],
        use_dense=True,
        use_bm25=True,
    )

    expected_bm25 = np.asarray([[0.0, 0.25, 1.0]], dtype=np.float32)
    expected = 0.5 * np.asarray([[0.0, 0.5, 1.0]]) + 0.5 * expected_bm25
    np.testing.assert_allclose(videos[0].scores, expected, rtol=1e-6, atol=1e-6)
```

- [ ] **Step 6: Characterize the current all-or-nothing Dense loader**

Create `tests/orchestration/test_temporal_evidence_setup.py` with explicit source/retrieval fakes:

```python
from types import SimpleNamespace
import numpy as np

from hcmai.common.config import AppConfig
from hcmai.orchestration.setup import _load_dense_temporal
from hcmai.retrieval.models import RetrievalSource
from tests.retrieval.evidence.fakes import FakeEncoder, FakeIndex


class FakeSourceRetriever:
    def __init__(self, source, index, encoder) -> None:
        self.source = source
        self.index = index
        self.encoder = encoder


class FakeRetrievalService:
    def __init__(self, retrievers) -> None:
        self.retrievers = {retriever.source: retriever for retriever in retrievers}

    def source_retriever(self, source):
        return self.retrievers.get(source)


def test_v9_dense_loader_requires_context_and_asr() -> None:
    scores = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    visual = FakeSourceRetriever(RetrievalSource.VISUAL, FakeIndex(scores), encoder)
    context = FakeSourceRetriever(RetrievalSource.CONTEXT, FakeIndex(scores), encoder)
    retrieval = FakeRetrievalService([visual, context])

    scorer, context_ready, asr_ready = _load_dense_temporal(
        AppConfig(),
        retrieval,
        visual,
        messages := [],
    )

    assert scorer is None
    assert context_ready is True
    assert asr_ready is False
    assert any("ASR segment retriever missing" in message for message in messages)
```

This test is intentionally expected to change in Task 3.

- [ ] **Step 7: Run the full characterization group**

Run:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/retrieval/evidence/test_legacy_characterization.py \
  tests/orchestration/test_temporal_evidence_setup.py
```

Expected: PASS.

- [ ] **Step 8: Commit the baseline tests**

```bash
git add tests/retrieval/evidence tests/orchestration/test_temporal_evidence_setup.py
git commit -m "test(temporal): characterize v9 evidence scoring"
```

---

### Task 2: Introduce first-class raw score components without changing legacy output

**Why this task exists:** v9 destroys modality identity too early. Dense collapses Visual/Context/ASR inside `DenseTemporalScorer`; BM25 collapses title/caption/OCR/ASR inside `BM25TemporalScorer`. Adaptive fusion cannot reason about source quality after these sums are taken.

**Files:**
- Create: `src/hcmai/retrieval/evidence/components.py`
- Create: `tests/retrieval/evidence/test_components.py`
- Modify: `src/hcmai/retrieval/evidence/dense.py:17-80`
- Modify: `src/hcmai/retrieval/evidence/bm25.py:32-143`
- Modify: `src/hcmai/retrieval/evidence/__init__.py`

**Interfaces:**
- Produces: `TemporalScoreComponent`, `TemporalScoreBundle`.
- Produces: `DenseTemporalScorer.score_components(retrieval_events)`.
- Produces: `BM25TemporalScorer.score_components(original_events, caption_events)`.
- Preserves: current `DenseTemporalScorer.score_events()` and `BM25TemporalScorer.score_events()` numerical behavior.

- [ ] **Step 1: Write failing tests for component contracts**

Create `tests/retrieval/evidence/test_components.py`:

```python
import numpy as np
import pytest

from hcmai.retrieval.evidence.components import TemporalScoreBundle, TemporalScoreComponent


def test_component_requires_finite_two_dimensional_scores() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        TemporalScoreComponent("visual_dense", np.asarray([1.0], dtype=np.float32))

    with pytest.raises(ValueError, match="finite"):
        TemporalScoreComponent(
            "visual_dense",
            np.asarray([[0.0, np.inf]], dtype=np.float32),
        )


def test_bundle_requires_same_event_and_frame_shape() -> None:
    with pytest.raises(ValueError, match="same score shape"):
        TemporalScoreBundle(
            {
                "visual_dense": TemporalScoreComponent(
                    "visual_dense", np.zeros((2, 3), dtype=np.float32)
                ),
                "context_dense": TemporalScoreComponent(
                    "context_dense", np.zeros((2, 4), dtype=np.float32)
                ),
            }
        )
```

- [ ] **Step 2: Run component tests and verify RED**

Run:

```bash
PYTHONPATH=src python -m pytest -q tests/retrieval/evidence/test_components.py
```

Expected: FAIL because `components.py` does not exist.

- [ ] **Step 3: Implement immutable component contracts**

Create `src/hcmai/retrieval/evidence/components.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalScoreComponent:
    name: str
    raw_scores: np.ndarray
    coverage: np.ndarray | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.raw_scores, dtype=np.float32)
        if scores.ndim != 2:
            raise ValueError("component scores must be two-dimensional")
        if not np.all(np.isfinite(scores)):
            raise ValueError("component scores must contain only finite values")
        if self.coverage is not None:
            coverage = np.asarray(self.coverage, dtype=bool)
            if coverage.shape != (scores.shape[1],):
                raise ValueError("component coverage must match frame count")
            object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "raw_scores", scores)


@dataclass(frozen=True, slots=True)
class TemporalScoreBundle:
    components: Mapping[str, TemporalScoreComponent]

    def __post_init__(self) -> None:
        copied = dict(self.components)
        if not copied:
            raise ValueError("temporal score bundle must contain at least one component")
        shapes = {component.raw_scores.shape for component in copied.values()}
        if len(shapes) != 1:
            raise ValueError("all temporal components must have the same score shape")
        for key, component in copied.items():
            if key != component.name:
                raise ValueError("component mapping key must match component name")
        object.__setattr__(self, "components", MappingProxyType(copied))

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.components.values())).raw_scores.shape
```

- [ ] **Step 4: Add failing Dense component tests**

Extend `test_components.py`:

```python
def test_dense_score_components_preserves_raw_modality_scores() -> None:
    visual = FakeIndex(np.asarray([[0.20, 0.21, 0.22]], dtype=np.float32))
    context = FakeIndex(np.asarray([[2.0, 4.0, 8.0]], dtype=np.float32))
    asr = FakeIndex(np.asarray([[-0.1, 0.0, 0.1]], dtype=np.float32))
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    scorer = DenseTemporalScorer(
        visual_index=visual,
        context_index=context,
        asr_index=asr,
        visual_encoder=encoder,
        text_encoder=encoder,
        weights=DenseTemporalWeights(),
    )

    bundle = scorer.score_components(["event"])

    assert set(bundle.components) == {"visual_dense", "context_dense", "asr_dense"}
    np.testing.assert_array_equal(
        bundle.components["visual_dense"].raw_scores,
        visual.scores,
    )
```

- [ ] **Step 5: Implement Dense raw component scoring and preserve the legacy wrapper**

In `src/hcmai/retrieval/evidence/dense.py`, add `score_components()` that encodes the batch once and returns raw matrices. Rewrite `score_events()` only as a compatibility wrapper:

```python
def score_components(self, retrieval_events: Sequence[str]) -> TemporalScoreBundle:
    events = [" ".join(event.split()) for event in retrieval_events]
    if not events or any(not event for event in events):
        raise ValueError("retrieval events must contain non-empty strings")

    visual_vectors = np.asarray(self.visual_encoder.encode_text(events), dtype=np.float32)
    text_vectors = np.asarray(self.text_encoder.encode_text(events), dtype=np.float32)
    positions = np.arange(len(self.visual_index.frame_ids), dtype=np.int64)

    return TemporalScoreBundle(
        {
            "visual_dense": TemporalScoreComponent(
                "visual_dense",
                self.visual_index.score_subset(visual_vectors, positions, self.chunk_size),
            ),
            "context_dense": TemporalScoreComponent(
                "context_dense",
                self.context_index.score_subset(text_vectors, positions, self.chunk_size),
            ),
            "asr_dense": TemporalScoreComponent(
                "asr_dense",
                self.asr_index.score_subset(text_vectors, positions, self.chunk_size),
            ),
        }
    )


def score_events(self, retrieval_events: Sequence[str]) -> np.ndarray:
    bundle = self.score_components(retrieval_events)
    visual = minmax_rows(bundle.components["visual_dense"].raw_scores)
    context = minmax_rows(bundle.components["context_dense"].raw_scores)
    asr = minmax_rows(bundle.components["asr_dense"].raw_scores)
    return np.asarray(
        self.weights.visual_weight * visual
        + self.weights.context_weight * context
        + self.weights.asr_weight * asr,
        dtype=np.float32,
    )
```

- [ ] **Step 6: Add failing BM25 field-separation test**

Use a minimal scorer instance with tiny sparse matrices and assert exact field names:

```python
def test_bm25_score_components_keeps_fields_separate() -> None:
    bundle = scorer.score_components(["mot nguoi"], ["mot nguoi"])

    assert set(bundle.components) == {
        "bm25_title",
        "bm25_caption",
        "bm25_ocr",
        "bm25_asr",
    }
```

- [ ] **Step 7: Implement BM25 component scoring and keep old `score_events()`**

Add to `BM25TemporalScorer`:

```python
def score_components(
    self,
    original_events: Sequence[str],
    caption_events: Sequence[str],
) -> TemporalScoreBundle:
    if len(original_events) != len(caption_events) or not original_events:
        raise ValueError("original and caption events must have equal non-zero lengths")

    rows = {
        "bm25_title": np.zeros((len(original_events), len(self._reorder)), dtype=np.float32),
        "bm25_caption": np.zeros((len(original_events), len(self._reorder)), dtype=np.float32),
        "bm25_ocr": np.zeros((len(original_events), len(self._reorder)), dtype=np.float32),
        "bm25_asr": np.zeros((len(original_events), len(self._reorder)), dtype=np.float32),
    }
    for event_index, (original, caption) in enumerate(
        zip(original_events, caption_events, strict=True)
    ):
        original_tokens = _tokenize(original)
        rows["bm25_title"][event_index] = self._score_field("title", original_tokens)[self._reorder]
        rows["bm25_ocr"][event_index] = self._score_field("ocr", original_tokens)[self._reorder]
        rows["bm25_asr"][event_index] = self._score_field("asr", original_tokens)[self._reorder]
        rows["bm25_caption"][event_index] = self._score_field(
            "caption", _tokenize(caption)
        )[self._reorder]

    return TemporalScoreBundle(
        {name: TemporalScoreComponent(name, scores) for name, scores in rows.items()}
    )
```

Then implement `score_events()` by taking the component rows and applying the current configured field weights exactly once. Do not move the legacy field weights into the new adaptive path yet.

- [ ] **Step 8: Run component and legacy regression tests**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/retrieval/evidence/test_components.py \
  tests/retrieval/evidence/test_legacy_characterization.py
```

Expected: PASS, including all numerical legacy assertions.

- [ ] **Step 9: Commit component separation**

```bash
git add src/hcmai/retrieval/evidence tests/retrieval/evidence
git commit -m "refactor(temporal): expose raw evidence components"
```

---

### Task 3: Make Visual/Context/ASR Dense experts independently available

**Why this task exists:** `_load_dense_temporal()` currently returns `None` whenever either Context or ASR is missing. That makes a partial artifact outage disable useful Visual evidence and prevents adaptive fusion from renormalizing over the sources that actually exist.

**Files:**
- Modify: `src/hcmai/retrieval/evidence/dense.py:20-74`
- Modify: `src/hcmai/orchestration/setup.py:121-209`
- Modify: `tests/orchestration/test_temporal_evidence_setup.py`
- Modify: `tests/retrieval/evidence/test_components.py`

**Interfaces:**
- `DenseTemporalScorer.__init__(*, visual_index: Any, context_index: Any | None, asr_index: Any | None, visual_encoder: Any, text_encoder: Any | None, weights: DenseTemporalWeights, chunk_size: int = 65_536)`.
- `DenseTemporalScorer.score_components()` always returns `visual_dense`; returns Context/ASR components only when those experts are configured.
- `_load_dense_temporal()` returns a scorer whenever Visual exists, plus accurate `context_ready`/`asr_ready` booleans.

- [ ] **Step 1: Replace the Task 1 characterization with desired partial-loader tests**

Replace the former all-or-nothing assertion with a Visual+Context test using the Task 1 helpers:

```python
def test_dense_loader_keeps_visual_context_when_asr_is_missing() -> None:
    scores = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    visual = FakeSourceRetriever(RetrievalSource.VISUAL, FakeIndex(scores), encoder)
    context = FakeSourceRetriever(RetrievalSource.CONTEXT, FakeIndex(scores), encoder)
    retrieval = FakeRetrievalService([visual, context])

    scorer, context_ready, asr_ready = _load_dense_temporal(
        AppConfig(), retrieval, visual, messages := []
    )

    assert scorer is not None
    assert context_ready is True
    assert asr_ready is False
    assert any("ASR segment retriever missing" in message for message in messages)
```

Add minimal ASR segment/projector fakes for the Context-missing case:

```python
import pandas as pd


class FakeSegmentIndex:
    def __init__(self) -> None:
        self.mapping = pd.DataFrame(
            [{"video_id": "v1", "start_ms": 0, "end_ms": 1000}]
        )
        self.vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
        self.metadata = SimpleNamespace(embedding_dim=2)


class FakeProjection:
    video_id = "v1"
    frame_id = "f0"
    frame_idx = 0
    timestamp_ms = 0


class FakeProjector:
    def project(self, video_id: str, *, start_ms: int, end_ms: int):
        del video_id, start_ms, end_ms
        return FakeProjection()


class FakeASRRetriever(FakeSourceRetriever):
    def __init__(self, encoder) -> None:
        super().__init__(RetrievalSource.ASR, FakeSegmentIndex(), encoder)
        self.projector = FakeProjector()


def test_dense_loader_keeps_visual_asr_when_context_is_missing() -> None:
    scores = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    visual = FakeSourceRetriever(RetrievalSource.VISUAL, FakeIndex(scores), encoder)
    asr = FakeASRRetriever(encoder)
    retrieval = FakeRetrievalService([visual, asr])

    scorer, context_ready, asr_ready = _load_dense_temporal(
        AppConfig(), retrieval, visual, messages := []
    )

    assert scorer is not None
    assert context_ready is False
    assert asr_ready is True
```

- [ ] **Step 2: Run only the loader tests and verify RED**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/orchestration/test_temporal_evidence_setup.py
```

Expected: FAIL because v9 still returns `None` if either text expert is missing.

- [ ] **Step 3: Make `DenseTemporalScorer` optional-expert aware**

Change constructor types and validation:

```python
def __init__(
    self,
    *,
    visual_index: Any,
    context_index: Any | None,
    asr_index: Any | None,
    visual_encoder: Any,
    text_encoder: Any | None,
    weights: DenseTemporalWeights,
    chunk_size: int = 65_536,
) -> None:
    _validate_optional_indexes(visual_index, context_index, asr_index)
    if (context_index is not None or asr_index is not None) and text_encoder is None:
        raise ValueError("text_encoder is required for Context or ASR Dense scoring")
    self.visual_index = visual_index
    self.context_index = context_index
    self.asr_index = asr_index
    self.visual_encoder = visual_encoder
    self.text_encoder = text_encoder
    self.weights = weights
    self.chunk_size = chunk_size
```

`score_components()` must encode BGE text only when at least one text expert exists:

```python
text_vectors = None
if self.context_index is not None or self.asr_index is not None:
    assert self.text_encoder is not None
    text_vectors = np.asarray(self.text_encoder.encode_text(events), dtype=np.float32)
```

Always create `visual_dense`. Add `context_dense`/`asr_dense` conditionally.

- [ ] **Step 4: Preserve legacy fixed-weight behavior when all three components exist**

In legacy `score_events()`, require all three components because the old fixed `DenseTemporalWeights` contract sums exactly three normalized experts:

```python
required = {"visual_dense", "context_dense", "asr_dense"}
if set(bundle.components) != required:
    raise RuntimeError("legacy Dense temporal fusion requires Visual, Context, and ASR")
```

This avoids silently changing what `legacy` means. Adaptive mode will be responsible for partial-source operation.

- [ ] **Step 5: Rewrite `_load_dense_temporal()` as independent capability loading**

Use Visual as mandatory. Context and ASR are validated separately. Choose the shared BGE encoder from whichever text retriever is available:

```python
text_encoder = None
if context is not None:
    text_encoder = context.encoder
elif asr_retriever is not None:
    text_encoder = asr_retriever.encoder
```

Only build `SegmentProjectedASRIndex` if ASR exists and validates. A Context failure must not set `asr_ready=False`; an ASR failure must not set `context_ready=False`.

Construct the scorer with any surviving optional expert:

```python
scorer = DenseTemporalScorer(
    visual_index=visual.index,
    context_index=context.index if context_ready and context is not None else None,
    asr_index=projected_asr if asr_ready else None,
    visual_encoder=visual.encoder,
    text_encoder=text_encoder,
    weights=settings.search.hybrid_temporal.dense,
    chunk_size=settings.search.alignment.chunk_size,
)
```

- [ ] **Step 6: Add four component-availability tests**

Cover these exact combinations:

```text
Visual only               -> {visual_dense}
Visual + Context          -> {visual_dense, context_dense}
Visual + ASR              -> {visual_dense, asr_dense}
Visual + Context + ASR    -> all three
```

Also assert the text encoder is called zero times for Visual-only and exactly once for each multi-event request when Context and/or ASR exists.

- [ ] **Step 7: Run focused tests**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/retrieval/evidence/test_components.py \
  tests/orchestration/test_temporal_evidence_setup.py \
  tests/retrieval/evidence/test_legacy_characterization.py
```

Expected: PASS.

- [ ] **Step 8: Commit optional expert support**

```bash
git add src/hcmai/retrieval/evidence/dense.py src/hcmai/orchestration/setup.py tests
git commit -m "refactor(temporal): allow partial dense evidence experts"
```

---

### Task 4: Make ASR evidence interval-aware and distinguish coverage from score

**Why this task exists:** `SegmentProjectedASRIndex` currently maps each ASR segment to one frame and then fills every uncovered frame with the weakest covered score. That creates two errors: it discards the temporal extent of speech, and it makes "no speech here" look like a low-confidence speech match rather than no evidence.

**Files:**
- Modify: `src/hcmai/retrieval/evidence/asr_projected.py:24-233`
- Create: `tests/retrieval/evidence/test_asr_interval.py`
- Modify: `src/hcmai/retrieval/evidence/dense.py`

**Interfaces:**
- `SegmentProjectedASRIndex.coverage_mask: np.ndarray` shaped `[frame_count]`.
- `SegmentProjectedASRIndex.score_subset()` returns finite scores and leaves uncovered frames as `0.0`; coverage tells fusion whether the zeros are meaningful.
- `DenseTemporalScorer.score_components()` attaches ASR coverage to the `asr_dense` component.

- [ ] **Step 1: Add explicit ASR test fakes**

At the top of `tests/retrieval/evidence/test_asr_interval.py`, define the minimal segment/canonical/projector contracts used by `SegmentProjectedASRIndex`:

```python
from types import SimpleNamespace
import numpy as np
import pandas as pd


class FakeSegmentIndex:
    def __init__(self, rows: list[dict[str, object]], vectors: list[list[float]]) -> None:
        self.mapping = pd.DataFrame(rows)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.metadata = SimpleNamespace(embedding_dim=self.vectors.shape[1])


class FakeCanonicalIndex:
    def __init__(self, timestamps: list[int]) -> None:
        count = len(timestamps)
        self.frame_ids = np.asarray([f"f{i}" for i in range(count)])
        self.video_ids = np.asarray(["v1"] * count)
        self.frame_idx = np.arange(count, dtype=np.int64)
        self.timestamps = np.asarray(timestamps, dtype=np.int64)

    def video_positions(self, video_id: str) -> np.ndarray:
        return np.flatnonzero(self.video_ids == video_id)


class FakeProjection:
    def __init__(self, position: int, timestamps: list[int]) -> None:
        self.video_id = "v1"
        self.frame_id = f"f{position}"
        self.frame_idx = position
        self.timestamp_ms = timestamps[position]


class FakeProjector:
    def __init__(self, timestamps: list[int], fallback_position: int) -> None:
        self.timestamps = timestamps
        self.fallback_position = fallback_position

    def project(self, video_id: str, *, start_ms: int, end_ms: int):
        del video_id, start_ms, end_ms
        return FakeProjection(self.fallback_position, self.timestamps)
```

- [ ] **Step 2: Write interval, overlap, and fallback tests**

Use one video with canonical timestamps `[0, 1000, 2000, 3000, 4000]`. A segment `[900, 3100]` must cover positions 1, 2, 3. Two overlapping segments must use max aggregation. A segment `[2100, 2900]` over sparse timestamps `[0, 2000, 4000]` contains no canonical frame and must fall back to exactly one projected frame.

```python
def test_asr_segment_covers_every_canonical_frame_inside_interval() -> None:
    timestamps = [0, 1000, 2000, 3000, 4000]
    index = SegmentProjectedASRIndex(
        segment_index=FakeSegmentIndex(
            [{"video_id": "v1", "start_ms": 900, "end_ms": 3100}],
            [[1.0, 0.0]],
        ),
        canonical_index=FakeCanonicalIndex(timestamps),
        projector=FakeProjector(timestamps, fallback_position=2),
    )

    scores = index.score_subset(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.arange(5, dtype=np.int64),
    )

    np.testing.assert_array_equal(
        index.coverage_mask,
        np.asarray([False, True, True, True, False]),
    )
    np.testing.assert_allclose(scores, [[0.0, 1.0, 1.0, 1.0, 0.0]])


def test_overlapping_asr_segments_use_max_similarity() -> None:
    timestamps = [0, 1000, 2000, 3000]
    index = SegmentProjectedASRIndex(
        segment_index=FakeSegmentIndex(
            [
                {"video_id": "v1", "start_ms": 500, "end_ms": 2200},
                {"video_id": "v1", "start_ms": 1500, "end_ms": 3200},
            ],
            [[0.8, 0.6], [0.9, 0.4358899]],
        ),
        canonical_index=FakeCanonicalIndex(timestamps),
        projector=FakeProjector(timestamps, fallback_position=1),
    )
    scores = index.score_subset(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.arange(4, dtype=np.int64),
    )

    np.testing.assert_allclose(scores[0, 2], 0.9, atol=1e-6)


def test_asr_interval_without_sampled_frame_uses_projector_fallback() -> None:
    timestamps = [0, 2000, 4000]
    index = SegmentProjectedASRIndex(
        segment_index=FakeSegmentIndex(
            [{"video_id": "v1", "start_ms": 2100, "end_ms": 2900}],
            [[1.0, 0.0]],
        ),
        canonical_index=FakeCanonicalIndex(timestamps),
        projector=FakeProjector(timestamps, fallback_position=1),
    )

    assert index.coverage_mask.tolist() == [False, True, False]
```

- [ ] **Step 3: Run ASR interval tests and verify RED**

```bash
PYTHONPATH=src python -m pytest -q tests/retrieval/evidence/test_asr_interval.py
```

Expected: FAIL because current class exposes only `segment_frame_positions` and floor-fills uncovered frames.

- [ ] **Step 4: Precompute canonical interval ranges per segment**

Keep the existing class name to minimize runtime churn. Do not assume one video's canonical positions are globally contiguous. During `__init__`, build a CSR-like segment-to-frame mapping:

```python
self.segment_coverage_offsets: np.ndarray    # int64 [segment_count + 1]
self.segment_coverage_positions: np.ndarray  # flattened canonical positions
self.segment_frame_positions: np.ndarray     # existing one-point fallback positions
self.coverage_mask: np.ndarray               # bool [F]
```

For each segment, obtain the video's canonical positions from `canonical_index.video_positions(video_id)`, then inspect the timestamps at those positions:

```python
video_positions = canonical_index.video_positions(video_id)
video_timestamps = self.timestamps[video_positions]
inside = video_positions[(video_timestamps >= start_ms) & (video_timestamps <= end_ms)]
```

If `inside` is non-empty, append those canonical positions to the flattened coverage array and advance the offsets. If `inside` is empty, call the existing projector and append exactly its canonical fallback position. `coverage_mask` is `True` at every position present in the flattened coverage array.

Do not mutate `SegmentDenseIndex`, its mapping, or any artifact.

- [ ] **Step 5: Rewrite ASR score scatter to use interval ranges**

Keep chunked matrix multiplication. For each segment in a chunk, fetch its explicit canonical coverage positions from the CSR-style arrays and apply max aggregation:

```python
for local_segment, global_segment in enumerate(range(start, stop)):
    offset_start = self.segment_coverage_offsets[global_segment]
    offset_stop = self.segment_coverage_offsets[global_segment + 1]
    target_positions = self.segment_coverage_positions[offset_start:offset_stop]
    if not len(target_positions):
        continue
    for event_index in range(len(queries)):
        np.maximum.at(
            frame_scores[event_index],
            target_positions,
            chunk_scores[event_index, local_segment],
        )
```

Initialize `frame_scores` with `0.0`, not `-inf`. Coverage is now a separate boolean signal, so do not perform the current lines 222-228 floor fill.

- [ ] **Step 6: Attach ASR coverage to the component bundle**

In Dense component construction:

```python
"asr_dense": TemporalScoreComponent(
    "asr_dense",
    self.asr_index.score_subset(text_vectors, positions, self.chunk_size),
    coverage=np.asarray(self.asr_index.coverage_mask, dtype=bool),
)
```

- [ ] **Step 7: Run focused tests including legacy regression**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/retrieval/evidence/test_asr_interval.py \
  tests/retrieval/evidence/test_components.py \
  tests/retrieval/evidence/test_legacy_characterization.py
```

Expected: interval tests PASS. Legacy Dense numerical tests that use fake ASR indexes still PASS because their fake indexes have no coverage semantics and legacy behavior remains isolated.

- [ ] **Step 8: Commit interval ASR**

```bash
git add src/hcmai/retrieval/evidence/asr_projected.py src/hcmai/retrieval/evidence/dense.py tests/retrieval/evidence
git commit -m "feat(temporal): spread ASR scores across segment intervals"
```

---

### Task 5: Add robust row calibration and deterministic reliability

**Why this task exists:** v9's `minmax_rows()` guarantees that every nonconstant row has a maximum of `1.0`, even when the raw difference is tiny. Example: `[0.201, 0.203, 0.209]` becomes `[0, 0.25, 1]`, which can turn noise into apparently strong evidence. Adaptive P0 must preserve relative ranking without treating weak dynamic range as confidence.

**Files:**
- Create: `src/hcmai/retrieval/evidence/calibration.py`
- Create: `tests/retrieval/evidence/test_calibration.py`
- Modify: `src/hcmai/common/config.py:316-362`

**Interfaces:**
- `RobustCalibrationConfig`.
- `CalibratedComponent(scores: np.ndarray, reliability: np.ndarray)`.
- `calibrate_component(raw_scores, config) -> CalibratedComponent`.
- Legacy `minmax_rows()` remains untouched in `normalization.py`.

- [ ] **Step 1: Add calibration config with explicit validated defaults**

In `src/hcmai/common/config.py` add:

```python
class RobustCalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q_low: float = Field(default=0.05, ge=0.0, lt=1.0)
    q_high: float = Field(default=0.95, gt=0.0, le=1.0)
    top_fraction: float = Field(default=0.01, gt=0.0, le=0.25)
    eps: float = Field(default=1e-6, gt=0.0)

    @model_validator(mode="after")
    def validate_quantiles(self) -> "RobustCalibrationConfig":
        if self.q_low >= self.q_high:
            raise ValueError("q_low must be less than q_high")
        return self
```

Do not modify existing `DenseTemporalWeights`, `BM25FieldWeights`, or legacy hybrid weights in this task.

- [ ] **Step 2: Write RED tests for weak, strong, constant, and outlier rows**

Create `tests/retrieval/evidence/test_calibration.py`:

```python
import numpy as np

from hcmai.common.config import RobustCalibrationConfig
from hcmai.retrieval.evidence.calibration import calibrate_component


def test_constant_row_has_zero_scores_and_zero_reliability() -> None:
    result = calibrate_component(
        np.asarray([[0.3, 0.3, 0.3, 0.3]], dtype=np.float32),
        RobustCalibrationConfig(),
    )
    np.testing.assert_array_equal(result.scores, np.zeros((1, 4), dtype=np.float32))
    np.testing.assert_array_equal(result.reliability, np.asarray([0.0], dtype=np.float32))


def test_tiny_dynamic_range_is_ranked_but_not_fully_trusted() -> None:
    result = calibrate_component(
        np.asarray([[0.201, 0.203, 0.204, 0.209]], dtype=np.float32),
        RobustCalibrationConfig(),
    )
    assert result.scores[0, -1] > result.scores[0, 0]
    assert 0.0 < result.reliability[0] < 1.0


def test_large_outlier_is_clipped_by_quantiles() -> None:
    raw = np.asarray([[0.0, 0.1, 0.2, 0.3, 100.0]], dtype=np.float32)
    result = calibrate_component(raw, RobustCalibrationConfig(q_high=0.8))
    assert result.scores.max() == 1.0
    assert np.isfinite(result.scores).all()
```

- [ ] **Step 3: Run calibration tests and verify RED**

```bash
PYTHONPATH=src python -m pytest -q tests/retrieval/evidence/test_calibration.py
```

Expected: FAIL because `calibration.py` does not exist.

- [ ] **Step 4: Implement robust calibration**

Create `src/hcmai/retrieval/evidence/calibration.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from hcmai.common.config import RobustCalibrationConfig


@dataclass(frozen=True, slots=True)
class CalibratedComponent:
    scores: np.ndarray       # [E, F], in [0, 1]
    reliability: np.ndarray  # [E], in [0, 1]


def calibrate_component(
    raw_scores: np.ndarray,
    config: RobustCalibrationConfig,
) -> CalibratedComponent:
    values = np.asarray(raw_scores, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("raw_scores must be two-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("raw_scores must contain only finite values")
    if values.shape[1] == 0:
        return CalibratedComponent(
            np.zeros_like(values),
            np.zeros(values.shape[0], dtype=np.float32),
        )

    low = np.quantile(values, config.q_low, axis=1, keepdims=True)
    high = np.quantile(values, config.q_high, axis=1, keepdims=True)
    span = high - low

    clipped = np.clip(values, low, high)
    calibrated = np.zeros_like(values, dtype=np.float32)
    np.divide(clipped - low, span, out=calibrated, where=span > config.eps)

    median = np.median(values, axis=1)
    q25 = np.quantile(values, 0.25, axis=1)
    q75 = np.quantile(values, 0.75, axis=1)
    iqr = q75 - q25
    top_k = max(1, int(np.ceil(values.shape[1] * config.top_fraction)))
    top_mean = np.mean(np.partition(values, -top_k, axis=1)[:, -top_k:], axis=1)
    robust_z = np.maximum(top_mean - median, 0.0) / (iqr + config.eps)
    reliability = robust_z / (1.0 + robust_z)
    reliability = np.where(span[:, 0] > config.eps, reliability, 0.0)

    return CalibratedComponent(
        np.asarray(calibrated, dtype=np.float32),
        np.asarray(np.clip(reliability, 0.0, 1.0), dtype=np.float32),
    )
```

- [ ] **Step 5: Add invariance tests**

Verify affine scale invariance of reliability and calibration ordering:

```python
def test_positive_affine_rescaling_preserves_calibration() -> None:
    raw = np.asarray([[0.2, 0.4, 0.9, 1.2]], dtype=np.float32)
    a = calibrate_component(raw, RobustCalibrationConfig())
    b = calibrate_component(raw * 100.0 + 7.0, RobustCalibrationConfig())

    np.testing.assert_allclose(a.scores, b.scores, atol=1e-6)
    np.testing.assert_allclose(a.reliability, b.reliability, atol=1e-6)
```

- [ ] **Step 6: Run all calibration tests**

```bash
PYTHONPATH=src python -m pytest -q tests/retrieval/evidence/test_calibration.py
```

Expected: PASS with no warnings/NaN/Inf.

- [ ] **Step 7: Commit calibration**

```bash
git add src/hcmai/common/config.py src/hcmai/retrieval/evidence/calibration.py tests/retrieval/evidence/test_calibration.py
git commit -m "feat(temporal): add robust component calibration"
```

---

### Task 6: Add event-adaptive, coverage-aware multimodal fusion behind a legacy switch

**Why this task exists:** P0 becomes useful only when independent components are combined without giving equal influence to irrelevant/missing evidence. This task introduces the actual adaptive emission function while preserving the same final matrix contract.

**Files:**
- Create: `src/hcmai/retrieval/evidence/fusion.py`
- Create: `tests/retrieval/evidence/test_fusion.py`
- Modify: `src/hcmai/common/config.py:346-362`
- Modify: `src/hcmai/retrieval/evidence/hybrid.py:19-102`

**Interfaces:**
- `AdaptiveTemporalFusionConfig`.
- `EventModalityRouter.multipliers(original_event, retrieval_event) -> dict[str, float]`.
- `TemporalFusionScorer.fuse(original_events, retrieval_events, bundle) -> np.ndarray`.
- `HybridTemporalConfig.fusion_mode: Literal["legacy", "adaptive_p0"]` with default `legacy`.

- [ ] **Step 1: Add explicit adaptive config**

Add to `src/hcmai/common/config.py`:

```python
class AdaptiveTemporalFusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calibration: RobustCalibrationConfig = Field(default_factory=RobustCalibrationConfig)
    confidence_gating: bool = True
    event_routing: bool = True
    base_component_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "visual_dense": 0.35,
            "context_dense": 0.35,
            "asr_dense": 0.08,
            "bm25_title": 0.02,
            "bm25_caption": 0.10,
            "bm25_ocr": 0.04,
            "bm25_asr": 0.06,
        }
    )
    visual_boost: float = Field(default=1.4, ge=1.0)
    speech_boost: float = Field(default=5.0, ge=1.0)
    ocr_boost: float = Field(default=3.0, ge=1.0)

    @model_validator(mode="after")
    def validate_component_weights(self) -> "AdaptiveTemporalFusionConfig":
        if any(weight < 0.0 for weight in self.base_component_weights.values()):
            raise ValueError("adaptive component weights must be non-negative")
        if sum(self.base_component_weights.values()) <= 0.0:
            raise ValueError("adaptive component weights must contain positive mass")
        return self
```

Extend `HybridTemporalConfig`:

```python
fusion_mode: Literal["legacy", "adaptive_p0"] = "legacy"
adaptive: AdaptiveTemporalFusionConfig = Field(default_factory=AdaptiveTemporalFusionConfig)
```

Keep `dense`, `bm25_fields`, `dense_weight`, and `bm25_weight` unchanged for legacy mode. In `TemporalEvidenceScorer.__init__`, instantiate the adaptive scorer once:

```python
self._adaptive_fusion = TemporalFusionScorer(config.adaptive)
```

- [ ] **Step 2: Write router tests before implementing it**

Create `tests/retrieval/evidence/test_fusion.py`:

```python
def test_speech_event_boosts_asr_components() -> None:
    router = EventModalityRouter(AdaptiveTemporalFusionConfig())
    weights = router.multipliers(
        "Cô gái nói chuyện với người đối diện về món ăn.",
        "The woman talks with a person seated opposite her about the dish.",
    )
    assert weights["asr_dense"] > weights["visual_dense"]
    assert weights["bm25_asr"] > weights["bm25_caption"]


def test_visible_text_event_boosts_ocr() -> None:
    router = EventModalityRouter(AdaptiveTemporalFusionConfig())
    weights = router.multipliers(
        'Màn hình hiển thị dòng chữ "TP.HCM".',
        'The screen displays the text "TP.HCM".',
    )
    assert weights["bm25_ocr"] > weights["bm25_asr"]
```

- [ ] **Step 3: Implement deterministic cue routing**

Use normalized lowercase strings and small cue tables, not an LLM:

```python
SPEECH_CUES = (
    "nói", "hỏi", "trả lời", "đối thoại", "phỏng vấn", "cho biết", "giới thiệu",
    "says", "asks", "answers", "talks", "speaks", "interview", "announces",
)
OCR_CUES = (
    "dòng chữ", "chữ", "biển hiệu", "màn hình hiển thị", "logo", "nhãn",
    "text", "sign", "screen displays", "logo", "label",
)
VISUAL_CUES = (
    "mặc", "cầm", "đặt", "đứng", "ngồi", "chạy", "xe", "đĩa", "màu",
    "wearing", "holds", "places", "stands", "sits", "runs", "plate", "color",
)
```

`EventModalityRouter.multipliers()` starts from `base_component_weights`, multiplies ASR components by `speech_boost` if a speech cue is present, OCR by `ocr_boost` if an OCR cue is present, and Visual+Context by `visual_boost` if visual cues are present. Return the unnormalized positive weights; fusion handles final normalization after reliability/coverage.

- [ ] **Step 4: Write fusion tests for coverage and reliability**

Use synthetic components where ASR covers only the middle frame:

```python
def test_fusion_renormalizes_when_asr_has_no_frame_coverage() -> None:
    bundle = TemporalScoreBundle(
        {
            "visual_dense": TemporalScoreComponent(
                "visual_dense", np.asarray([[0.2, 0.4, 0.6]], dtype=np.float32)
            ),
            "asr_dense": TemporalScoreComponent(
                "asr_dense",
                np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
                coverage=np.asarray([False, True, False]),
            ),
        }
    )
    config = AdaptiveTemporalFusionConfig(
        confidence_gating=False,
        event_routing=False,
        base_component_weights={"visual_dense": 0.5, "asr_dense": 0.5},
    )
    scorer = TemporalFusionScorer(config)

    actual = scorer.fuse(
        original_events=["người phụ nữ nói"],
        retrieval_events=["the woman speaks"],
        bundle=bundle,
    )
    visual_only = scorer.fuse(
        original_events=["người phụ nữ nói"],
        retrieval_events=["the woman speaks"],
        bundle=TemporalScoreBundle({"visual_dense": bundle.components["visual_dense"]}),
    )

    assert actual.shape == (1, 3)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[0, [0, 2]], visual_only[0, [0, 2]], atol=1e-6)
```

Also add a constant-noise component test: a constant BM25 OCR row must receive reliability `0` and contribute no weight when confidence gating is enabled.

- [ ] **Step 5: Implement `TemporalFusionScorer`**

Create `src/hcmai/retrieval/evidence/fusion.py` with an explicit constructor and router dependency:

```python
class TemporalFusionScorer:
    def __init__(self, config: AdaptiveTemporalFusionConfig) -> None:
        self.config = config
        self.router = EventModalityRouter(config)

    def fuse(
    self,
    *,
    original_events: Sequence[str],
    retrieval_events: Sequence[str],
    bundle: TemporalScoreBundle,
) -> np.ndarray:
    if len(original_events) != len(retrieval_events):
        raise ValueError("original and retrieval event counts must match")
    if bundle.shape[0] != len(original_events):
        raise ValueError("component event count must match query event count")

    calibrated = {
        name: calibrate_component(component.raw_scores, self.config.calibration)
        for name, component in bundle.components.items()
    }
    result = np.zeros(bundle.shape, dtype=np.float32)

    for event_index, (original, retrieval) in enumerate(
        zip(original_events, retrieval_events, strict=True)
    ):
        requested = self.router.multipliers(original, retrieval)
        numerator = np.zeros(bundle.shape[1], dtype=np.float32)
        denominator = np.zeros(bundle.shape[1], dtype=np.float32)

        for name, component in bundle.components.items():
            base = float(requested.get(name, 0.0))
            if base <= 0.0:
                continue
            confidence = (
                float(calibrated[name].reliability[event_index])
                if self.config.confidence_gating
                else 1.0
            )
            weight = base * confidence
            if weight <= 0.0:
                continue
            coverage = (
                np.ones(bundle.shape[1], dtype=np.float32)
                if component.coverage is None
                else component.coverage.astype(np.float32)
            )
            effective = weight * coverage
            numerator += effective * calibrated[name].scores[event_index]
            denominator += effective

        np.divide(numerator, denominator, out=result[event_index], where=denominator > 0.0)
        missing = denominator <= 0.0
        if np.any(missing) and "visual_dense" in calibrated:
            result[event_index, missing] = calibrated["visual_dense"].scores[event_index, missing]

    return result
```

- [ ] **Step 6: Add `legacy`/`adaptive_p0` branch in `TemporalEvidenceScorer`**

In `hybrid.py`, move the current body into `_score_legacy(original_events, retrieval_events, *, caption_events, use_dense, use_bm25) -> np.ndarray`. Add `_score_components(original_events, retrieval_events, *, caption_events, use_dense, use_bm25) -> TemporalScoreBundle` to merge Dense and BM25 bundles without summing fields:

```python
components: dict[str, TemporalScoreComponent] = {}
if use_dense:
    components.update(self.dense.score_components(retrieval_events).components)
if use_bm25:
    components.update(self.bm25.score_components(original_events, caption_events).components)
bundle = TemporalScoreBundle(components)
```

Then:

```python
if self.config.fusion_mode == "legacy":
    scores = self._score_legacy(
        original_events,
        retrieval_events,
        caption_events=caption_events,
        use_dense=use_dense,
        use_bm25=use_bm25,
    )
else:
    scores = self._adaptive_fusion.fuse(
        original_events=original_events,
        retrieval_events=retrieval_events,
        bundle=bundle,
    )
```

Both branches must end at the existing shape validation and `_split_videos()`.

- [ ] **Step 7: Handle partial Dense experts only in adaptive mode**

If `fusion_mode="legacy"` and the caller requests Dense while the Dense scorer lacks Context/ASR, raise the explicit legacy error from Task 3. If `fusion_mode="adaptive_p0"`, use every available component and renormalize automatically.

- [ ] **Step 8: Run fusion, component, calibration, and legacy tests**

```bash
PYTHONPATH=src python -m pytest -q \
  tests/retrieval/evidence/test_fusion.py \
  tests/retrieval/evidence/test_calibration.py \
  tests/retrieval/evidence/test_components.py \
  tests/retrieval/evidence/test_legacy_characterization.py \
  tests/orchestration/test_temporal_evidence_setup.py
```

Expected: PASS. The legacy characterization suite is the rollback gate.

- [ ] **Step 9: Commit adaptive fusion**

```bash
git add src/hcmai/common/config.py src/hcmai/retrieval/evidence tests
git commit -m "feat(temporal): add adaptive multimodal emission fusion"
```

---

### Task 7: Add component-level diagnostics and a reproducible L26_V254 debug workflow

**Why this task exists:** P0 must answer "why did this video rank here?" without guessing from UI thumbnails. The known `L26_V254` case should be inspectable by event and modality so P1 DP work starts from measured evidence rather than intuition.

**Files:**
- Create: `src/hcmai/retrieval/evidence/diagnostics.py`
- Create: `tests/retrieval/evidence/test_diagnostics.py`
- Create: `scripts/debug_temporal_evidence.py`
- Modify: `src/hcmai/retrieval/evidence/hybrid.py`

**Interfaces:**
- `TemporalEvidenceScorer.debug_score_events(original_events: Sequence[str], retrieval_events: Sequence[str], *, caption_events: Sequence[str] | None, use_dense: bool, use_bm25: bool, top_positions: int = 10) -> TemporalEvidenceDebugResult` for local/offline diagnostics only.
- Existing public `score_events()` remains unchanged.
- Debug output includes raw peak, calibrated peak, reliability, coverage ratio, and top canonical positions for each event/component.

- [ ] **Step 1: Define debug dataclasses and tests**

Create immutable records:

```python
@dataclass(frozen=True, slots=True)
class ComponentEventDebug:
    component: str
    event_index: int
    raw_max: float
    raw_median: float
    calibrated_max: float
    reliability: float
    coverage_ratio: float
    top_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TemporalEvidenceDebugResult:
    fused_scores: np.ndarray
    rows: tuple[ComponentEventDebug, ...]
```

Test that a component with coverage `[False, True, False]` reports `coverage_ratio == 1/3` and top positions are deterministic.

- [ ] **Step 2: Implement debug summarization without duplicating scoring logic**

The diagnostics layer must consume the same `TemporalScoreBundle` and calibration results used by adaptive fusion. It must not rescore embeddings or indexes. Extract shared component/calibration preparation into a required private method `_prepare_adaptive_components(original_events, retrieval_events, *, caption_events, use_dense, use_bm25) -> TemporalScoreBundle` and call that method from both `score_events()` and `debug_score_events()` so they cannot drift.

For top positions use:

```python
positions = np.argsort(-calibrated_scores[event_index], kind="stable")[:top_k]
```

- [ ] **Step 3: Add a CLI script for one prepared query**

`scripts/debug_temporal_evidence.py` accepts:

```text
--run B3                  adaptive B-series stage to inspect (B1-B6)
--query-file PATH          YAML containing `query` and `retrieval_events`
--video-id L26_V254        repeatable
--top-frames 10
--use-bm25
```

It loads the same runtime services as the app, calls debug scoring, filters canonical positions to requested video IDs, and prints a table per event/component:

```text
Event  Component      RawMax  Reliability  Coverage  TopFrameIdx
E1     visual_dense   0.612   0.71         100%      325,300,475
E1     context_dense  0.688   0.77         100%      325,350,300
E1     asr_dense      0.204   0.08          42%      750,775,800
```

Do not add this data to KIS/TRAKE HTTP responses in P0.

- [ ] **Step 4: Add a golden diagnostic query fixture**

Create `tests/fixtures/l26_v254_query.yaml` with the frozen prepared event bundle for the cooking query:

```yaml
query: |
  Một cô gái mặc tạp dề trắng đứng cạnh một lọ hoa riềng tía.
  Cô gái mặc tạp dề trắng đặt bốn nguyên liệu X chưa xác định lên một đĩa trắng.
  Cùng cô gái mặc tạp dề trắng cầm hai nguyên liệu X cùng loại lên.
  Cô gái mặc tạp dề trắng nói chuyện với một người ngồi đối diện về món ăn sẽ nấu.
retrieval_events:
  - "A woman wearing a white apron stands beside a vase of purple galangal flowers."
  - "A woman wearing a white apron places four unidentified ingredients X on a white plate."
  - "The same woman wearing a white apron holds two of the same unidentified ingredients X."
  - "The woman wearing a white apron talks with a person seated opposite her about the dish they will cook."
```

This fixture is not a general parser test; query preparation is frozen.

- [ ] **Step 5: Run unit tests**

```bash
PYTHONPATH=src python -m pytest -q tests/retrieval/evidence/test_diagnostics.py
```

Expected: PASS.

- [ ] **Step 6: Run the real-artifact diagnostic on the target video**

In the competition environment where artifacts are mounted:

```bash
PYTHONPATH=src python scripts/debug_temporal_evidence.py \
  --run B3 \
  --query-file tests/fixtures/l26_v254_query.yaml \
  --video-id L26_V254 \
  --top-frames 12 \
  --use-bm25
```

Record for each event:
- Visual top frame indices;
- Context top frame indices;
- ASR top covered frame indices;
- BM25 field peaks;
- reliability per component;
- final adaptive fused top frame indices.

Run the same command for B1, B3, and B5 to isolate flat fusion, robust
calibration, and event routing without mutating the loaded baseline scorer.

Expected qualitative check from known artifacts: shell/plate evidence should appear around frame indices ~300-525 and dialogue evidence later around ~550-950. Do not require exact score values in a unit test because the deployed model/index artifacts are external to the source archive.

- [ ] **Step 7: Commit diagnostics**

```bash
git add src/hcmai/retrieval/evidence/diagnostics.py scripts/debug_temporal_evidence.py tests
git commit -m "chore(temporal): add component evidence diagnostics"
```

---

### Task 8: Run P0 ablations and choose whether adaptive scoring is safe to enable

**Why this task exists:** P0 changes several interacting pieces. The goal is not merely "tests pass"; we need to determine whether cleaner emissions improve ranking and whether any specific feature causes regressions before P1 changes DP.

**Files:**
- Create: `scripts/evaluate_temporal_p0.py`
- Create: `tests/retrieval/evidence/test_p0_ablation_config.py`
- Modify: `src/hcmai/common/config.py` only if an independent flag is missing.

**Interfaces:**
- The evaluation script runs the same query set under named performance configurations B0-B6.
- It must not change query text between ablations.
- It records result ranks and timing, not just top-1 screenshots.

- [ ] **Step 1: Make adaptive subfeatures independently switchable**

Ensure `AdaptiveTemporalFusionConfig` includes:

```python
robust_calibration: bool = True
confidence_gating: bool = True
event_routing: bool = True
```

In `TemporalFusionScorer`, centralize component calibration in `_calibrate(raw_scores)`. When `robust_calibration=True`, call `calibrate_component()`. When false, return `minmax_rows(raw_scores)` with reliability equal to `1.0` for nonconstant rows and `0.0` for constant rows. This creates the B1 flat-component baseline without changing DP or component extraction.

ASR interval projection is a data semantics change inside the ASR adapter. For ablation, add a runtime constructor flag `interval_projection: bool = True` to `SegmentProjectedASRIndex`; when false, execute the original one-point projection without the old floor-fill. This isolates interval spread from coverage semantics without restoring the misleading floor behavior.

- [ ] **Step 2: Define the exact ablation matrix**

Exact componentized-legacy recombination is an A1 regression test and must be
numerically equal to legacy scoring. It is not a runtime performance condition.

`scripts/evaluate_temporal_p0.py` must support these named runs:

```text
B0 legacy_v9
   legacy minmax + fixed Dense + fixed BM25 hybrid

B1 flat_components
   separated components with flat minmax-style fusion; intentionally a new equation

B2 asr_interval
   B1 + interval ASR coverage

B3 robust_calibration
   B2 + robust quantile calibration, confidence gating OFF, event routing OFF

B4 confidence_gating
   B3 + reliability gating

B5 adaptive_p0
   B4 + event cue routing

B6 dense_only
   B5 without BM25 components
```

Do not alter DP settings between B0-B6. Use the `TemporalSearchService` alignment
configuration loaded by application setup; run definitions must not own a fresh
default `AlignmentConfig`.

- [ ] **Step 3: Write an ablation-config regression test**

Create a test that constructs all six named configs and asserts:
- `fusion_mode` is correct;
- only the intended feature differs from the prior stage;
- the evaluator never replaces the loaded alignment configuration.

This prevents accidental "P0 gain" from a hidden DP parameter change.

- [ ] **Step 4: Implement evaluation output**

For each query/run, write one JSONL row:

```json
{
  "run": "B5_adaptive_p0",
  "query_id": "q17",
  "target_video_id": "L26_V254",
  "target_rank": 4,
  "top_video_id": "L26_V254",
  "top_score": 2.31,
  "retrieval_ms": 182.4,
  "alignment_ms": 11.8
}
```

If a query has no known target label, omit `target_rank` and still record top video IDs/scores/timings for manual review.

- [ ] **Step 5: Evaluate the known `L26_V254` case first**

Run B0-B6 with the frozen query bundle and record:
- rank of `L26_V254`;
- top 10 video IDs;
- score gap between target and rank 1;
- top aligned frame indices for each event;
- retrieval/alignment latency.

Stop and inspect if any stage makes the target materially worse before running the full query set. This is a diagnostic gate, not a hard-coded requirement that the target must become rank 1.

- [ ] **Step 6: Evaluate the available query set with fixed prepared queries**

For the 50 organizer queries without complete labels:
- prepare/freeze each query once using the ChatGPT skill;
- never regenerate it between B0-B6;
- manually tag each query as `visual`, `speech`, `ocr`, or `mixed` for analysis only;
- compare top-result relevance and path coherence side by side.

Where ground truth becomes known, compute:

```text
Recall@1, Recall@5, Recall@20
median target-video rank
p50 retrieval latency
p95 retrieval latency
p50 alignment latency
p95 alignment latency
```

- [ ] **Step 7: Define the rollout decision rule**

Keep `fusion_mode="legacy"` as default until all of these are true:

1. Unit/regression suite passes.
2. No NaN/Inf appears in adaptive scoring.
3. `L26_V254` diagnostics show the relevant shell/plate/dialogue evidence rather than unrelated modality peaks dominating fusion.
4. Adaptive P0 does not materially increase p95 retrieval latency beyond the competition budget.
5. On the manually reviewed query set, adaptive P0 reduces obvious unrelated top results without creating a new systematic failure class.

If these hold, switch deployment config to `adaptive_p0`. Do not delete legacy code before the competition.

- [ ] **Step 8: Run the complete P0 test suite**

```bash
PYTHONPATH=src python -m pytest -q tests
```

Expected: PASS.

- [ ] **Step 9: Commit evaluation tooling**

```bash
git add scripts/evaluate_temporal_p0.py tests src/hcmai/common/config.py
git commit -m "chore(temporal): add P0 ablation evaluation"
```

---

## P0 Completion Gate

Do not start P1 soft-monotonic DP work until every item below is answered with evidence:

- [ ] Legacy v9 numerical equivalence is proven by tests.
- [ ] Every active source can be inspected as an independent raw score matrix.
- [ ] Context/ASR outages no longer disable surviving Dense experts in adaptive mode.
- [ ] ASR has explicit interval coverage and uncovered frames are no longer represented by a fake floor score.
- [ ] Robust calibration never returns NaN/Inf and constant rows have zero reliability.
- [ ] Adaptive fusion renormalizes over only available/covered/reliable evidence.
- [ ] Event routing is deterministic and can be disabled independently.
- [ ] `TemporalSearchService` still receives the same `list[VideoEventScores]` contract.
- [ ] `src/hcmai/temporal/dp.py` is byte-for-byte behaviorally unchanged.
- [ ] The `L26_V254` diagnostic explains which component supports each event and whether the target is lost before or during DP.
- [ ] A1 componentized-legacy recombination equals legacy scoring numerically.
- [ ] B0-B6 experiments are recorded with identical prepared queries and the loaded DP settings.

## Expected Outcome of P0

P0 is successful even if `L26_V254` is not yet rank 1. The key success is that the emission matrix becomes trustworthy enough that we can distinguish two cases:

```text
Case A: Correct evidence is weak/noisy before DP
        -> continue improving emission/fusion.

Case B: Correct evidence is strong at the right local regions,
        but strict chronology still suppresses the target
        -> P1 soft-monotonic DP is justified by measurement.
```

That separation is the main purpose of P0. P1 should change transition/alignment logic only after P0 makes the input evidence interpretable.

## Execution Notes

Recommended implementation order is exactly Task 1 -> Task 8. Do not parallelize Tasks 2-6 because their contracts build on each other. Task 7 diagnostics can begin only after Task 6 adaptive fusion is stable. Task 8 is evaluation/rollout, not another scoring refactor.

At execution time, create an isolated worktree first using `superpowers:using-git-worktrees`, then implement each task with TDD and review each commit before continuing.
