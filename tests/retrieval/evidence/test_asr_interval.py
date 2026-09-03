"""Tests for interval-aware ASR projection, legacy point floor, and coverage masking."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
import numpy as np
import pandas as pd
import pytest

from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex
from tests.retrieval.evidence.test_legacy_characterization import reference_v9_asr


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
    def __init__(self, timestamps: list[int], fallback_positions: list[int] | int) -> None:
        self.timestamps = timestamps
        self.fallback_positions = (
            fallback_positions if isinstance(fallback_positions, list) else [fallback_positions]
        )
        self._call_count = 0

    def project(self, video_id: str, *, start_ms: int, end_ms: int):
        del video_id, start_ms, end_ms
        pos = self.fallback_positions[self._call_count % len(self.fallback_positions)]
        self._call_count += 1
        return FakeProjection(pos, self.timestamps)


def _make_index(
    segment_index: Any,
    canonical_index: Any,
    projector: Any,
) -> SegmentProjectedASRIndex:
    return SegmentProjectedASRIndex(
        segment_index=cast(Any, segment_index),
        canonical_index=cast(Any, canonical_index),
        projector=cast(Any, projector),
    )


def test_asr_segment_covers_every_canonical_frame_inside_interval() -> None:
    timestamps = [0, 1000, 2000, 3000, 4000]
    index = _make_index(
        segment_index=FakeSegmentIndex(
            [{"video_id": "v1", "start_ms": 900, "end_ms": 3100}],
            [[1.0, 0.0]],
        ),
        canonical_index=FakeCanonicalIndex(timestamps),
        projector=FakeProjector(timestamps, fallback_positions=2),
    )

    scores, coverage = index.score_subset_masked(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.arange(5, dtype=np.int64),
        interval_projection=True,
    )

    np.testing.assert_array_equal(
        coverage,
        np.asarray([False, True, True, True, False]),
    )
    np.testing.assert_allclose(scores, [[0.0, 1.0, 1.0, 1.0, 0.0]])


def test_overlapping_asr_segments_use_max_similarity() -> None:
    timestamps = [0, 1000, 2000, 3000]
    index = _make_index(
        segment_index=FakeSegmentIndex(
            [
                {"video_id": "v1", "start_ms": 500, "end_ms": 2200},
                {"video_id": "v1", "start_ms": 1500, "end_ms": 3200},
            ],
            [[0.8, 0.6], [0.9, 0.4358899]],
        ),
        canonical_index=FakeCanonicalIndex(timestamps),
        projector=FakeProjector(timestamps, fallback_positions=1),
    )
    scores, coverage = index.score_subset_masked(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.arange(4, dtype=np.int64),
        interval_projection=True,
    )

    np.testing.assert_allclose(scores[0, 2], 0.9, atol=1e-6)
    assert coverage[2] is True or coverage[2] == 1


def test_asr_interval_without_sampled_frame_uses_projector_fallback() -> None:
    timestamps = [0, 2000, 4000]
    index = _make_index(
        segment_index=FakeSegmentIndex(
            [{"video_id": "v1", "start_ms": 2100, "end_ms": 2900}],
            [[1.0, 0.0]],
        ),
        canonical_index=FakeCanonicalIndex(timestamps),
        projector=FakeProjector(timestamps, fallback_positions=1),
    )
    scores, coverage = index.score_subset_masked(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.arange(3, dtype=np.int64),
        interval_projection=True,
    )
    np.testing.assert_array_equal(coverage, [False, True, False])
    np.testing.assert_allclose(scores, [[0.0, 1.0, 0.0]])


def test_negative_covered_similarity_remains_negative() -> None:
    timestamps = [0, 1000, 2000]
    index = _make_index(
        segment_index=FakeSegmentIndex(
            [{"video_id": "v1", "start_ms": 900, "end_ms": 1100}],
            [[-0.8, 0.6]],
        ),
        canonical_index=FakeCanonicalIndex(timestamps),
        projector=FakeProjector(timestamps, fallback_positions=1),
    )
    # query dot vector = 1.0 * (-0.8) + 0.0 = -0.8
    scores, coverage = index.score_subset_masked(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.arange(3, dtype=np.int64),
        interval_projection=True,
    )
    assert coverage[1] is True or coverage[1] == 1
    assert coverage[0] is False or coverage[0] == 0
    assert scores[0, 1] == pytest.approx(-0.8, abs=1e-5)
    assert scores[0, 0] == 0.0  # uncovered is 0.0


def test_uncovered_score_is_zero_and_coverage_is_false() -> None:
    timestamps = [0, 1000, 2000]
    index = _make_index(
        segment_index=FakeSegmentIndex(
            [{"video_id": "v1", "start_ms": 0, "end_ms": 100}],
            [[0.5, 0.5]],
        ),
        canonical_index=FakeCanonicalIndex(timestamps),
        projector=FakeProjector(timestamps, fallback_positions=0),
    )
    scores, coverage = index.score_subset_masked(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.arange(3, dtype=np.int64),
        interval_projection=True,
    )
    assert coverage[2] is False or coverage[2] == 0
    assert scores[0, 2] == 0.0


def test_legacy_point_floor_matches_reference_v9() -> None:
    timestamps = [0, 1000, 2000, 3000]
    segment_vectors = [[0.8, 0.6], [0.4, 0.9]]
    mapping = [
        {"video_id": "v1", "start_ms": 500, "end_ms": 1500},
        {"video_id": "v1", "start_ms": 2500, "end_ms": 3500},
    ]
    # Fallback positions: segment 0 -> frame 1, segment 1 -> frame 3
    index = _make_index(
        segment_index=FakeSegmentIndex(mapping, segment_vectors),
        canonical_index=FakeCanonicalIndex(timestamps),
        projector=FakeProjector(timestamps, fallback_positions=[1, 3]),
    )
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    scores_legacy = index.score_subset_legacy(query, np.arange(4, dtype=np.int64))

    # Segment similarities for query:
    # seg 0: 0.8, seg 1: 0.4
    seg_sims = np.asarray([[0.8, 0.4]], dtype=np.float32)
    ref = reference_v9_asr(seg_sims, np.asarray([1, 3]), frame_count=4)
    np.testing.assert_allclose(scores_legacy, ref, rtol=1e-5, atol=1e-5)


def test_point_masked_and_interval_masked_differ_on_multiframe_segment() -> None:
    timestamps = [0, 1000, 2000, 3000]
    index = _make_index(
        segment_index=FakeSegmentIndex(
            [{"video_id": "v1", "start_ms": 900, "end_ms": 2100}],  # covers frames 1 and 2
            [[1.0, 0.0]],
        ),
        canonical_index=FakeCanonicalIndex(timestamps),
        projector=FakeProjector(timestamps, fallback_positions=1),  # fallback maps only to frame 1
    )
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    positions = np.arange(4, dtype=np.int64)

    scores_point, cov_point = index.score_subset_masked(
        query, positions, interval_projection=False
    )
    scores_interval, cov_interval = index.score_subset_masked(
        query, positions, interval_projection=True
    )

    # Point masked only covers frame 1
    np.testing.assert_array_equal(cov_point, [False, True, False, False])
    np.testing.assert_allclose(scores_point, [[0.0, 1.0, 0.0, 0.0]])

    # Interval masked covers frame 1 and 2
    np.testing.assert_array_equal(cov_interval, [False, True, True, False])
    np.testing.assert_allclose(scores_interval, [[0.0, 1.0, 1.0, 0.0]])
