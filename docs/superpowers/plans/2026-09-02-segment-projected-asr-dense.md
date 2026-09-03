# Segment-Projected ASR Dense Temporal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Dense/Hybrid temporal retrieval available by scoring the ASR source through the existing segment-native `asr_segments` index projected onto canonical frames, instead of requiring a nonexistent frame-native `artifacts/indexes/asr` index built by re-embedding 470k frames.

**Architecture:** A new `SegmentProjectedASRIndex` wraps the existing `SegmentDenseIndex` plus canonical frame identity taken from the visual Dense index. At load time it precomputes a deterministic segment→frame position map by reusing the already-tested `SegmentFrameProjector` (inside-interval, else nearest-midpoint within a gap). Per request it scores events against segment vectors (cosine) and scatters each segment score to its projected frame (max on collision), filling frames with no ASR evidence at a per-event floor so min-max normalization treats them as "no evidence found," not mid-range. The wrapper exposes the exact identity arrays and `score_subset` contract that `DenseTemporalScorer` already consumes, so the monotonic DP, hybrid fusion, and BM25 paths remain untouched.

**Tech Stack:** Python 3.11+, NumPy, FAISS-backed `SegmentDenseIndex`, Pydantic v2 config, pytest. Runtime interpreter is `aic/bin/python`; tests run via `PYTHONPATH=.:src aic/bin/pytest`.

---

## Background & Key Facts (verified)

- **SOURCE** `artifacts/indexes/asr_segments/` is `entity_kind=segment`, 40,156 vectors, dim 1024, columns `segment_id, video_id, segment_index, start_ms, end_ms`, model label `reused-asr-vectors`. It exists and is loadable via `SegmentDenseIndex.load`.
- **SOURCE** `artifacts/indexes/visual/` and `context/` are `entity_kind=frame`, 470,804 vectors, dim 1024, BGE-m3, canonical order shared.
- **SOURCE** `DenseTemporalScorer` (`src/hcmai/retrieval/evidence/dense.py`) needs the ASR index to expose `frame_ids/video_ids/frame_idx/timestamps`, `metadata.embedding_dim`, and `score_subset(query_vectors, positions, chunk_size)`. It scores the ASR index with the SAME shared BGE-m3 text vectors it uses for Context.
- **SOURCE** `SegmentFrameProjector` (`src/hcmai/retrieval/retriever/segment/projector.py`) already projects one segment interval to one canonical frame: frames inside `[start_ms, end_ms)` win, else the nearest frame to the segment midpoint within `max_projection_gap_ms` (default 5000). Returns `None` when no frame qualifies.
- **SOURCE** `settings.index.asr_segment_path = artifacts/indexes/asr_segments` and `settings.index.asr_projection_max_gap_ms = 5000` already exist in `configs/baseline.yaml` and `IndexConfig`.
- **SOURCE** `minmax_rows` (`src/hcmai/retrieval/evidence/normalization.py`) requires all-finite input and maps each event row's min→0, max→1; constant rows become 0.
- **PROPOSED** Segment ASR vectors share the BGE-m3 embedding space (dim 1024) with the shared text encoder, so cosine between query text vectors and segment vectors is meaningful. This must be guarded at load (dimension check + normalized-vector check) and recorded as an assumption to validate in an HCMAI experiment.

## Design Decisions (from brainstorm)

1. **Reuse existing projection semantics.** The same `SegmentFrameProjector` used by the production RRF ASR path defines segment→frame mapping, keeping Dense ASR consistent with the rest of the system. AGENTS.md: "ASR is timeline evidence, not inherently frame-native."
2. **Aggregation = max.** When several segments project to the same frame, the frame keeps the best (max) segment cosine for each event.
3. **Coverage floor for absent evidence.** Frames with no projected segment are filled with the per-event minimum over covered frames, so after `minmax_rows` they normalize to 0 (the floor). This encodes "no evidence found" instead of inventing a mid-range score. Rows with zero covered frames stay all-equal and `minmax_rows` maps them to 0.
4. **No new 470k index, no DP change.** Scoring is a per-request `Q × 40,156` matmul plus a scatter, cheap enough to ignore `chunk_size`. `DenseTemporalScorer`, `temporal/dp.py`, hybrid fusion, and BM25 stay byte-for-byte unchanged.
5. **Swap the ASR source at load only.** `_load_dense_temporal` loads the segment index and wraps it; it no longer looks for `artifacts/indexes/asr`.

## File Structure

- Create: `src/hcmai/retrieval/evidence/asr_projected.py` — `SegmentProjectedASRIndex` (identity arrays, metadata shim, precomputed segment→frame map, `score_subset`).
- Modify: `src/hcmai/orchestration/setup.py` — `_load_dense_temporal` builds the projected ASR index from the segment index + visual identity instead of loading a frame ASR index.
- Test: `tests/retrieval/evidence/test_asr_projected.py` — unit tests for identity, projection, scatter-max, coverage floor, and validation guards.
- Test: `tests/orchestration/test_hybrid_health.py` — extend to assert the segment-projected ASR path reports `asr_dense` readiness (only if a matching test already exists; otherwise add a focused case).

---

### Task 1: SegmentProjectedASRIndex identity + precomputed projection

**Files:**
- Create: `src/hcmai/retrieval/evidence/asr_projected.py`
- Test: `tests/retrieval/evidence/test_asr_projected.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/evidence/test_asr_projected.py
"""Unit tests for segment-projected ASR Dense scoring."""

import numpy as np
import pytest

from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex


class _Meta:
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim


class _FakeSegmentIndex:
    """Minimal stand-in exposing the SegmentDenseIndex surface we use."""

    def __init__(self, vectors, mapping_rows, embedding_dim):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.mapping = _FakeMapping(mapping_rows)
        self.metadata = _Meta(embedding_dim)


class _FakeMapping:
    def __init__(self, rows):
        self._rows = rows

    def to_dict(self, orient):
        assert orient == "records"
        return list(self._rows)


def _visual_identity():
    # Two videos, canonical order by (video, frame_idx).
    frame_ids = np.array(["A_0", "A_1", "B_0"], dtype=object)
    video_ids = np.array(["A", "A", "B"], dtype=object)
    frame_idx = np.array([0, 1, 0], dtype=np.int64)
    timestamps = np.array([0, 1000, 0], dtype=np.int64)
    return frame_ids, video_ids, frame_idx, timestamps


def test_identity_arrays_mirror_visual_order():
    frame_ids, video_ids, frame_idx, timestamps = _visual_identity()
    segments = _FakeSegmentIndex(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        mapping_rows=[
            {"video_id": "A", "start_ms": 0, "end_ms": 1500},
            {"video_id": "B", "start_ms": 0, "end_ms": 500},
        ],
        embedding_dim=2,
    )
    index = SegmentProjectedASRIndex(
        segment_index=segments,
        frame_ids=frame_ids,
        video_ids=video_ids,
        frame_idx=frame_idx,
        timestamps=timestamps,
        max_projection_gap_ms=5000,
    )

    assert list(index.frame_ids) == ["A_0", "A_1", "B_0"]
    assert list(index.video_ids) == ["A", "A", "B"]
    assert list(index.frame_idx) == [0, 1, 0]
    assert list(index.timestamps) == [0, 1000, 0]
    assert index.metadata.embedding_dim == 2


def test_precomputed_map_projects_each_segment_to_a_frame():
    frame_ids, video_ids, frame_idx, timestamps = _visual_identity()
    segments = _FakeSegmentIndex(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        mapping_rows=[
            {"video_id": "A", "start_ms": 0, "end_ms": 1500},  # covers A_0 and A_1
            {"video_id": "B", "start_ms": 0, "end_ms": 500},   # covers B_0
        ],
        embedding_dim=2,
    )
    index = SegmentProjectedASRIndex(
        segment_index=segments,
        frame_ids=frame_ids,
        video_ids=video_ids,
        frame_idx=frame_idx,
        timestamps=timestamps,
        max_projection_gap_ms=5000,
    )

    # Segment 0 midpoint is 750ms; inside [0,1500) both A_0(0) and A_1(1000)
    # qualify, nearest midpoint is A_1 -> position 1. Segment 1 -> B_0 position 2.
    assert index.segment_frame_position.tolist() == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:src aic/bin/pytest tests/retrieval/evidence/test_asr_projected.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'hcmai.retrieval.evidence.asr_projected'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hcmai/retrieval/evidence/asr_projected.py
"""Segment-projected ASR Dense scoring over canonical frames.

This module adapts the segment-native ASR vector index into the frame-native
score matrix that ``DenseTemporalScorer`` consumes. It reuses the deterministic
segment-to-frame projection already used by the RRF ASR path and never mints
new canonical identity: identity is taken verbatim from the visual Dense index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hcmai.retrieval.retriever.segment.projector import SegmentFrameProjector


@dataclass(frozen=True, slots=True)
class _ProjFrame:
    """Lightweight canonical frame record used only for timeline projection."""

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int


class _EmbeddingMeta:
    """Expose the single metadata field DenseTemporalScorer validates."""

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = int(embedding_dim)


class SegmentProjectedASRIndex:
    """Score events against ASR segments and project them onto canonical frames.

    The wrapper presents the same identity arrays and ``score_subset`` contract
    as a frame-native Dense index so it is a drop-in ASR source for
    ``DenseTemporalScorer``. Frames without any projected segment receive a
    per-event floor from :meth:`score_subset` so absent ASR never becomes
    positive evidence.
    """

    def __init__(
        self,
        *,
        segment_index: Any,
        frame_ids: np.ndarray,
        video_ids: np.ndarray,
        frame_idx: np.ndarray,
        timestamps: np.ndarray,
        max_projection_gap_ms: int = 5_000,
    ) -> None:
        """Bind the segment index to canonical frame identity and precompute map."""

        self._segment_index = segment_index
        self.frame_ids = np.asarray(frame_ids, dtype=object)
        self.video_ids = np.asarray(video_ids, dtype=object)
        self.frame_idx = np.asarray(frame_idx, dtype=np.int64)
        self.timestamps = np.asarray(timestamps, dtype=np.int64)
        self.metadata = _EmbeddingMeta(segment_index.metadata.embedding_dim)
        self._segment_vectors = np.asarray(segment_index.vectors, dtype=np.float32)
        self.segment_frame_position = self._precompute_projection(
            max_projection_gap_ms
        )

    def _precompute_projection(self, max_projection_gap_ms: int) -> np.ndarray:
        """Map each segment row to a canonical frame position, or -1 if none."""

        frames = [
            _ProjFrame(
                frame_id=str(self.frame_ids[position]),
                video_id=str(self.video_ids[position]),
                frame_idx=int(self.frame_idx[position]),
                timestamp_ms=int(self.timestamps[position]),
            )
            for position in range(len(self.frame_ids))
        ]
        projector = SegmentFrameProjector(frames, max_projection_gap_ms)
        position_of = {
            str(self.frame_ids[position]): position
            for position in range(len(self.frame_ids))
        }
        rows = self._segment_index.mapping.to_dict(orient="records")
        mapped = np.full(len(rows), -1, dtype=np.int64)
        for segment_index, row in enumerate(rows):
            projection = projector.project(
                row["video_id"],
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
            )
            if projection is not None:
                mapped[segment_index] = position_of[projection.frame_id]
        return mapped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:src aic/bin/pytest tests/retrieval/evidence/test_asr_projected.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/retrieval/evidence/asr_projected.py tests/retrieval/evidence/test_asr_projected.py
git commit -m "feat(retrieval): add segment-projected ASR identity and projection map"
```

---

### Task 2: score_subset scatter-max with coverage floor

**Files:**
- Modify: `src/hcmai/retrieval/evidence/asr_projected.py`
- Test: `tests/retrieval/evidence/test_asr_projected.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/evidence/test_asr_projected.py  (append)
def _build_index():
    frame_ids = np.array(["A_0", "A_1", "B_0"], dtype=object)
    video_ids = np.array(["A", "A", "B"], dtype=object)
    frame_idx = np.array([0, 1, 0], dtype=np.int64)
    timestamps = np.array([0, 1000, 0], dtype=np.int64)
    segments = _FakeSegmentIndex(
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        mapping_rows=[
            {"video_id": "A", "start_ms": 900, "end_ms": 1100},  # -> A_1 (pos 1)
            {"video_id": "B", "start_ms": 0, "end_ms": 500},      # -> B_0 (pos 2)
        ],
        embedding_dim=2,
    )
    return SegmentProjectedASRIndex(
        segment_index=segments,
        frame_ids=frame_ids,
        video_ids=video_ids,
        frame_idx=frame_idx,
        timestamps=timestamps,
        max_projection_gap_ms=5000,
    )


def test_score_subset_scatters_segment_scores_to_frames():
    index = _build_index()
    # Two events: event 0 aligns with segment 0 axis, event 1 with segment 1 axis.
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    positions = np.arange(3, dtype=np.int64)

    scores = index.score_subset(queries, positions)

    assert scores.shape == (2, 3)
    # Covered frames carry the matching cosine.
    assert scores[0, 1] == pytest.approx(1.0)  # event 0 -> segment 0 -> A_1
    assert scores[1, 2] == pytest.approx(1.0)  # event 1 -> segment 1 -> B_0


def test_uncovered_frames_get_per_event_floor():
    index = _build_index()
    queries = np.array([[1.0, 0.0]], dtype=np.float32)  # event 0
    positions = np.arange(3, dtype=np.int64)

    scores = index.score_subset(queries, positions)

    # A_0 (pos 0) has no projected segment. Covered scores for event 0 are
    # {A_1: 1.0, B_0: 0.0}; floor is the min (0.0), so A_0 must equal 0.0
    # and never exceed a covered frame.
    assert scores[0, 0] == pytest.approx(0.0)
    assert scores[0, 0] <= scores[0, 1]
    assert np.all(np.isfinite(scores))


def test_score_subset_honors_position_subset_order():
    index = _build_index()
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    # Request positions in reversed order; columns must follow the argument.
    positions = np.array([2, 1, 0], dtype=np.int64)

    scores = index.score_subset(queries, positions)

    full = index.score_subset(queries, np.arange(3, dtype=np.int64))
    assert scores[0, 0] == pytest.approx(full[0, 2])
    assert scores[0, 1] == pytest.approx(full[0, 1])
    assert scores[0, 2] == pytest.approx(full[0, 0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:src aic/bin/pytest tests/retrieval/evidence/test_asr_projected.py -q`
Expected: FAIL with `AttributeError: 'SegmentProjectedASRIndex' object has no attribute 'score_subset'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hcmai/retrieval/evidence/asr_projected.py  (add method to the class)
    def score_subset(
        self,
        query_vectors: np.ndarray,
        positions: np.ndarray,
        chunk_size: int = 65_536,
    ) -> np.ndarray:
        """Score events against ASR segments and project them onto frames.

        ``chunk_size`` is accepted for interface parity but ignored: the segment
        matmul and scatter are small enough to run in one pass.
        """

        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
        queries = queries.reshape(len(queries), -1)
        segment_scores = queries @ self._segment_vectors.T  # (Q, num_segments)

        frame_count = len(self.frame_ids)
        covered = self.segment_frame_position >= 0
        covered_positions = self.segment_frame_position[covered]

        # Start covered frames at -inf so scatter-max keeps the best segment.
        frame_scores = np.full(
            (len(queries), frame_count), -np.inf, dtype=np.float32
        )
        if covered_positions.size:
            rows = np.repeat(np.arange(len(queries)), covered_positions.size)
            cols = np.tile(covered_positions, len(queries))
            values = segment_scores[:, covered].reshape(-1)
            np.maximum.at(frame_scores, (rows, cols), values)

        # Fill frames that never received a segment with the per-event floor so
        # absent ASR normalizes to zero rather than a mid-range value.
        finite = np.isfinite(frame_scores)
        masked = np.where(finite, frame_scores, np.inf)
        floor = masked.min(axis=1, keepdims=True)
        floor = np.where(np.isfinite(floor), floor, 0.0).astype(np.float32)
        frame_scores = np.where(finite, frame_scores, floor).astype(np.float32)

        positions = np.ascontiguousarray(positions, dtype=np.int64)
        return frame_scores[:, positions]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:src aic/bin/pytest tests/retrieval/evidence/test_asr_projected.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/retrieval/evidence/asr_projected.py tests/retrieval/evidence/test_asr_projected.py
git commit -m "feat(retrieval): scatter-max segment ASR scores with coverage floor"
```

---

### Task 3: Load-time validation guards

**Files:**
- Modify: `src/hcmai/retrieval/evidence/asr_projected.py`
- Test: `tests/retrieval/evidence/test_asr_projected.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/evidence/test_asr_projected.py  (append)
def test_rejects_identity_arrays_of_unequal_length():
    segments = _FakeSegmentIndex(
        vectors=[[1.0, 0.0]],
        mapping_rows=[{"video_id": "A", "start_ms": 0, "end_ms": 500}],
        embedding_dim=2,
    )
    with pytest.raises(ValueError, match="identity arrays"):
        SegmentProjectedASRIndex(
            segment_index=segments,
            frame_ids=np.array(["A_0", "A_1"], dtype=object),
            video_ids=np.array(["A"], dtype=object),
            frame_idx=np.array([0], dtype=np.int64),
            timestamps=np.array([0], dtype=np.int64),
            max_projection_gap_ms=5000,
        )


def test_rejects_segment_vectors_with_mismatched_dimension():
    segments = _FakeSegmentIndex(
        vectors=[[1.0, 0.0, 0.0]],  # width 3 but metadata says 2
        mapping_rows=[{"video_id": "A", "start_ms": 0, "end_ms": 500}],
        embedding_dim=2,
    )
    with pytest.raises(ValueError, match="segment vector dimension"):
        SegmentProjectedASRIndex(
            segment_index=segments,
            frame_ids=np.array(["A_0"], dtype=object),
            video_ids=np.array(["A"], dtype=object),
            frame_idx=np.array([0], dtype=np.int64),
            timestamps=np.array([0], dtype=np.int64),
            max_projection_gap_ms=5000,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:src aic/bin/pytest tests/retrieval/evidence/test_asr_projected.py -q`
Expected: FAIL because no `ValueError` is raised for either case.

- [ ] **Step 3: Write minimal implementation**

Insert the guards at the start of `__init__`, immediately after `self._segment_index = segment_index` and before the identity arrays are stored:

```python
# src/hcmai/retrieval/evidence/asr_projected.py  (inside __init__)
        self._segment_index = segment_index

        lengths = {
            len(frame_ids),
            len(video_ids),
            len(frame_idx),
            len(timestamps),
        }
        if len(lengths) != 1:
            raise ValueError("canonical identity arrays must have equal lengths")

        vectors = np.asarray(segment_index.vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("segment vectors must be a two-dimensional matrix")
        if vectors.shape[1] != int(segment_index.metadata.embedding_dim):
            raise ValueError(
                "segment vector dimension does not match index metadata: "
                f"{vectors.shape[1]} != {segment_index.metadata.embedding_dim}"
            )
```

Then reuse the already-validated `vectors` for `self._segment_vectors` instead of re-reading it:

```python
        self._segment_vectors = vectors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=.:src aic/bin/pytest tests/retrieval/evidence/test_asr_projected.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/retrieval/evidence/asr_projected.py tests/retrieval/evidence/test_asr_projected.py
git commit -m "feat(retrieval): validate identity and segment vector dimensions"
```

---

### Task 4: Wire segment-projected ASR into Dense temporal loading

**Files:**
- Modify: `src/hcmai/orchestration/setup.py:127-175` (`_load_dense_temporal`)
- Test: `tests/orchestration/test_asr_projected_loading.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestration/test_asr_projected_loading.py
"""Loading Dense temporal evidence with a segment-projected ASR source."""

import numpy as np

from hcmai.common.config import DenseTemporalWeights
from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex
from hcmai.retrieval.evidence.dense import DenseTemporalScorer


class _Meta:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim


class _FrameIndex:
    """Frame-native Dense index stub with three canonical positions."""

    def __init__(self):
        self.frame_ids = np.array(["A_0", "A_1", "B_0"], dtype=object)
        self.video_ids = np.array(["A", "A", "B"], dtype=object)
        self.frame_idx = np.array([0, 1, 0], dtype=np.int64)
        self.timestamps = np.array([0, 1000, 0], dtype=np.int64)
        self.metadata = _Meta(2)

    def score_subset(self, query_vectors, positions, chunk_size=65_536):
        return np.ones((len(query_vectors), len(positions)), dtype=np.float32)


class _SegmentIndex:
    def __init__(self):
        self.vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.metadata = _Meta(2)
        self._rows = [
            {"video_id": "A", "start_ms": 900, "end_ms": 1100},
            {"video_id": "B", "start_ms": 0, "end_ms": 500},
        ]

    class _Mapping:
        def __init__(self, rows):
            self._rows = rows

        def to_dict(self, orient):
            return list(self._rows)

    @property
    def mapping(self):
        return _SegmentIndex._Mapping(self._rows)


class _Encoder:
    def encode_text(self, events):
        return np.array([[1.0, 0.0] for _ in events], dtype=np.float32)


def test_dense_scorer_accepts_segment_projected_asr_index():
    visual = _FrameIndex()
    context = _FrameIndex()
    asr = SegmentProjectedASRIndex(
        segment_index=_SegmentIndex(),
        frame_ids=visual.frame_ids,
        video_ids=visual.video_ids,
        frame_idx=visual.frame_idx,
        timestamps=visual.timestamps,
        max_projection_gap_ms=5000,
    )

    scorer = DenseTemporalScorer(
        visual_index=visual,
        context_index=context,
        asr_index=asr,
        visual_encoder=_Encoder(),
        text_encoder=_Encoder(),
        weights=DenseTemporalWeights(),
        chunk_size=8,
    )

    scores = scorer.score_events(["mot su kien"])
    assert scores.shape == (1, 3)
    assert np.all(np.isfinite(scores))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=.:src aic/bin/pytest tests/orchestration/test_asr_projected_loading.py -q`
Expected: PASS already if `DenseTemporalScorer` accepts the wrapper unchanged — this test guards that contract. If it fails, it reveals a real incompatibility in `_validate_indexes`; fix by ensuring `SegmentProjectedASRIndex` exposes `frame_ids/video_ids/frame_idx/timestamps` and `metadata.embedding_dim` exactly (already done in Tasks 1–3). Do not weaken `DenseTemporalScorer`.

- [ ] **Step 3: Update the loader**

Replace the ASR-index loading block in `_load_dense_temporal` so it loads the segment index and wraps it. Locate the current block:

```python
# src/hcmai/orchestration/setup.py  (current)
    asr_index: Any | None = None
    asr_path = _runtime_path("HCMAI_ASR_INDEX_PATH", settings.index.asr_path)
    try:
        asr_index = RetrievalService.load_index(asr_path)
    except Exception as error:
        messages.append(
            f"Dense temporal evidence unavailable at {asr_path}: "
            f"{type(error).__name__}: {error}"
        )
```

Replace it with segment-index loading and projection wrapping:

```python
# src/hcmai/orchestration/setup.py  (new)
    asr_index: Any | None = None
    asr_segment_path = _runtime_path(
        "HCMAI_ASR_SEGMENT_INDEX_PATH", settings.index.asr_segment_path
    )
    try:
        # Reuse the existing segment-native ASR vectors instead of a frame index:
        # ASR is timeline evidence, so scores are projected onto canonical frames.
        segment_index = SegmentDenseIndex.load(asr_segment_path)
        asr_index = SegmentProjectedASRIndex(
            segment_index=segment_index,
            frame_ids=visual.index.frame_ids,
            video_ids=visual.index.video_ids,
            frame_idx=visual.index.frame_idx,
            timestamps=visual.index.timestamps,
            max_projection_gap_ms=settings.index.asr_projection_max_gap_ms,
        )
    except Exception as error:
        messages.append(
            f"Dense temporal evidence unavailable at {asr_segment_path}: "
            f"{type(error).__name__}: {error}"
        )
```

Add the imports near the other evidence imports at the top of `setup.py`:

```python
# src/hcmai/orchestration/setup.py  (imports)
from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
```

(`SegmentDenseIndex` is already imported in `setup.py`; if so, do not duplicate it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=.:src aic/bin/pytest tests/orchestration/test_asr_projected_loading.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/hcmai/orchestration/setup.py tests/orchestration/test_asr_projected_loading.py
git commit -m "feat(orchestration): load Dense ASR from projected segment index"
```

---

### Task 5: Real-artifact identity and readiness check

**Files:**
- Test: `tests/orchestration/test_asr_projected_loading.py` (add a real-artifact guard, skipped when artifacts are absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/orchestration/test_asr_projected_loading.py  (append)
from pathlib import Path

import pytest

from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

_VISUAL = Path("artifacts/indexes/visual")
_SEGMENTS = Path("artifacts/indexes/asr_segments")


@pytest.mark.skipif(
    not (_VISUAL.exists() and _SEGMENTS.exists()),
    reason="visual and asr_segments artifacts are required",
)
def test_real_segment_projection_preserves_canonical_identity():
    visual = RetrievalService.load_index(_VISUAL)
    segments = SegmentDenseIndex.load(_SEGMENTS)

    asr = SegmentProjectedASRIndex(
        segment_index=segments,
        frame_ids=visual.frame_ids,
        video_ids=visual.video_ids,
        frame_idx=visual.frame_idx,
        timestamps=visual.timestamps,
        max_projection_gap_ms=5000,
    )

    assert len(asr.frame_ids) == len(visual.frame_ids)
    assert list(asr.frame_ids[:5]) == list(visual.frame_ids[:5])
    # Every projected segment must land on a valid canonical position.
    projected = asr.segment_frame_position[asr.segment_frame_position >= 0]
    assert projected.min() >= 0
    assert projected.max() < len(visual.frame_ids)
    # At least some segments must project given real overlapping transcripts.
    assert projected.size > 0
```

- [ ] **Step 2: Run the test**

Run: `PYTHONPATH=.:src aic/bin/pytest tests/orchestration/test_asr_projected_loading.py -q`
Expected: PASS if artifacts exist, otherwise SKIPPED. If it FAILS on identity length, the visual and segment artifacts come from different dataset versions — stop and reconcile dataset lineage before continuing.

- [ ] **Step 3: No implementation change**

This task validates real artifacts; no source edits are expected. If the guard fails, fix lineage/config, not the scorer.

- [ ] **Step 4: Commit**

```bash
git add tests/orchestration/test_asr_projected_loading.py
git commit -m "test(orchestration): guard real segment-to-frame projection identity"
```

---

### Task 6: Acceptance, regression, and KNOWLEDGE.md

**Files:**
- Modify: `KNOWLEDGE.md`
- Test: existing suites

- [ ] **Step 1: Run the focused evidence and orchestration suites**

Run:
```bash
PYTHONPATH=.:src aic/bin/pytest \
  tests/retrieval/evidence \
  tests/orchestration/test_asr_projected_loading.py \
  tests/orchestration/test_hybrid_temporal_search.py \
  tests/orchestration/test_hybrid_health.py \
  tests/temporal -q
```
Expected: PASS (real-artifact test may SKIP without artifacts).

- [ ] **Step 2: Verify DP and hybrid fusion are untouched**

Run:
```bash
git diff --stat src/hcmai/temporal/dp.py src/hcmai/retrieval/evidence/hybrid.py
```
Expected: no output (no changes to either file).

- [ ] **Step 3: Compile changed modules**

Run:
```bash
PYTHONPATH=.:src aic/bin/python -m compileall -q \
  src/hcmai/retrieval/evidence src/hcmai/orchestration
```
Expected: no errors.

- [ ] **Step 4: Record the design decision in KNOWLEDGE.md**

Append this entry:

```markdown
## Segment-projected ASR Dense temporal

**Date:** 2026-09-02
**Problem:** DenseTemporalScorer expected a frame-native ASR index that was never
built; only a segment-native `asr_segments` index exists.

### Sources
- SOURCE `artifacts/indexes/asr_segments` metadata and `SegmentDenseIndex`.
- SOURCE `SegmentFrameProjector` timeline projection used by the RRF ASR path.
- AGENTS.md: "ASR is timeline evidence, not inherently frame-native."

### Findings
Segment ASR scores can be projected onto canonical frames per request by reusing
the existing deterministic segment-to-frame projection, avoiding a 470k-frame
re-embedding. Frames without a projected segment take a per-event floor so absent
ASR normalizes to zero.

### Relevance to HCMAI
Unblocks Dense/Hybrid temporal using existing artifacts and keeps ASR semantics as
timeline evidence.

### Status
PROPOSED — expected to match or beat a frame-native ASR index; must be validated by
an HCMAI retrieval experiment (ablate ASR-off vs frame-native vs segment-projected).

### Decision or Experiment
Adopt segment-projected ASR for Dense temporal. Validate with the plan's ablation
(A: no ASR, B: segment-projected ASR) on the evaluation query set before claiming
any accuracy gain.
```

- [ ] **Step 5: Commit**

```bash
git add KNOWLEDGE.md
git commit -m "docs(knowledge): record segment-projected ASR Dense decision"
```

---

## Self-Review

- **Spec coverage:** Identity mirroring (Task 1), projection precompute (Task 1), scatter-max + coverage floor (Task 2), validation guards (Task 3), loader wiring (Task 4), real-artifact identity (Task 5), acceptance + KNOWLEDGE.md (Task 6). All design decisions map to a task.
- **Placeholder scan:** No TBD/TODO; every code step is complete.
- **Type consistency:** `SegmentProjectedASRIndex` exposes `frame_ids`, `video_ids`, `frame_idx`, `timestamps`, `metadata.embedding_dim`, `segment_frame_position`, and `score_subset(query_vectors, positions, chunk_size)` consistently across all tasks and matches the `DenseTemporalScorer` and `DenseIndex.score_subset` contracts. `SegmentFrameProjector.project(video_id, start_ms=, end_ms=)` and its `SegmentFrameProjection.frame_id` field match the real module.

## Open Assumptions To Validate

1. **PROPOSED** Segment vectors share the BGE-m3 space of the shared text encoder. Task 3 guards dimension; add an experiment to confirm score sanity (top projected frames for a known spoken phrase).
2. **PROPOSED** Max aggregation and per-event floor are the right projection semantics; the ablation in Task 6 should compare against ASR-off to confirm ASR adds signal, not noise.

---

**Plan complete and saved to `docs/superpowers/plans/2026-09-02-segment-projected-asr-dense.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
