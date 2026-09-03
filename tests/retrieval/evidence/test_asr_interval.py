"""Tests for interval-aware ASR projection and coverage masking."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pandas as pd

from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex


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
