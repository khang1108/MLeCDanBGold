"""Tests for full-corpus visual event scoring before temporal DP decoding."""

from __future__ import annotations

import numpy as np

from hcmai.retrieval.retriever.video_scores import score_all_videos


class FakeIndex:
    """Small visual index that records the dense scoring window it receives."""

    def __init__(self) -> None:
        self.frame_ids = np.asarray(["a0", "a1", "b0", "b1", "c0"])
        self.frame_idx = np.asarray([0, 1, 0, 1, 0])
        self.timestamps = np.asarray([0, 1_000, 0, 1_000, 0])
        self.video_ids = np.asarray(["a", "a", "b", "b", "c"])

    def video_positions(self, video_id: str) -> np.ndarray:
        """Return positions owned by one canonical video."""

        return np.flatnonzero(self.video_ids == video_id)

    def score_subset(
        self,
        query_vectors: np.ndarray,
        positions: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray:
        """Assert that temporal search scores every visual-index frame."""

        assert positions.tolist() == [0, 1, 2, 3, 4]
        return np.asarray([[1, 2, 3, 4, 5]], dtype=np.float32)


def test_score_all_videos_scores_every_index_position() -> None:
    """Split one full-corpus score matrix back into ordered video rows."""

    rows = score_all_videos(FakeIndex(), np.asarray([[1.0]], dtype=np.float32))

    assert [row.video_id for row in rows] == ["a", "b", "c"]
    assert [row.frame_ids.tolist() for row in rows] == [
        ["a0", "a1"],
        ["b0", "b1"],
        ["c0"],
    ]
