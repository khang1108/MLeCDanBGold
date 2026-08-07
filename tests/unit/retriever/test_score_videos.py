"""Shortlisting by coverage then RRF must rescore only the kept videos' frames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcmai.retriever.video_scores import score_videos

_MAPPING = pd.DataFrame(
    {
        "embedding_index": [0, 1, 2, 3, 4],
        "frame_id": ["v1_b", "v2_a", "v1_a", "v3_a", "v1_c"],
        "video_id": ["v1", "v2", "v1", "v3", "v1"],
        "frame_idx": [10, 7, 5, 3, 20],
        "timestamp_ms": [400, 280, 200, 120, 800],
    }
)
# Both events rank a v1 frame first and v3_a second; v2_a is last for both.
_SCORES = np.array(
    [[0.9, 0.1, 0.5, 0.7, 0.3], [0.4, 0.2, 0.8, 0.6, 0.5]], dtype=np.float32
)


class _FakeIndex:
    """Exact index returning every frame in descending score order."""

    def __init__(self, mapping: pd.DataFrame, scores: np.ndarray) -> None:
        self._scores = scores
        self.video_ids = mapping["video_id"].to_numpy()
        self.frame_ids = mapping["frame_id"].to_numpy()
        self.frame_idx = mapping["frame_idx"].to_numpy()
        self.timestamps_ms = mapping["timestamp_ms"].to_numpy(dtype=np.float64)
        ordered = mapping.sort_values(["video_id", "frame_idx"])
        self.video_positions = {
            str(video_id): group["embedding_index"].to_numpy()
            for video_id, group in ordered.groupby("video_id", sort=False)
        }
        self.scored_positions: np.ndarray | None = None

    def search(self, query_vectors: np.ndarray, top_k: int):
        del query_vectors
        order = np.argsort(-self._scores, axis=1)[:, :top_k]
        return np.take_along_axis(self._scores, order, axis=1), order

    def score_subset(
        self, query_vectors: np.ndarray, positions: np.ndarray, chunk_size: int
    ) -> np.ndarray:
        del query_vectors, chunk_size
        self.scored_positions = positions
        return self._scores[:, positions]


def test_only_shortlisted_videos_are_rescored_in_canonical_frame_order() -> None:
    index = _FakeIndex(_MAPPING, _SCORES)
    results = score_videos(index, np.zeros((2, 4), dtype=np.float32), max_videos=2)

    assert [item.video_id for item in results] == ["v1", "v3"]
    # v2 never reaches the matmul, which is the whole point of shortlisting.
    assert index.scored_positions is not None
    assert 1 not in index.scored_positions.tolist()

    assert list(results[0].frame_ids) == ["v1_a", "v1_b", "v1_c"]
    assert list(results[0].frame_idx) == [5, 10, 20]
    assert np.array_equal(results[0].timestamps_ms, [200.0, 400.0, 800.0])
    assert np.array_equal(results[0].scores, _SCORES[:, [2, 0, 4]])
    assert np.array_equal(results[1].scores, _SCORES[:, [3]])


_COVERAGE_MAPPING = pd.DataFrame(
    {
        "embedding_index": [0, 1, 2, 3, 4],
        "frame_id": ["v1_a", "v3_a", "v3_b", "v2_a", "v2_b"],
        "video_id": ["v1", "v3", "v3", "v2", "v2"],
        "frame_idx": [1, 1, 2, 1, 2],
        "timestamp_ms": [100, 100, 200, 100, 200],
    }
)
# Event 1's top 3 is v1_a, v3_a, v2_a; event 2's is v3_a, v3_b, v2_b. So v1 has
# the single best frame of either event but no evidence at all for event 2.
_COVERAGE_SCORES = np.array(
    [[0.9, 0.8, 0.1, 0.7, 0.05], [0.1, 0.9, 0.8, 0.05, 0.7]], dtype=np.float32
)


def test_full_coverage_outranks_a_single_stronger_event() -> None:
    index = _FakeIndex(_COVERAGE_MAPPING, _COVERAGE_SCORES)
    results = score_videos(
        index,
        np.zeros((2, 4), dtype=np.float32),
        top_k=3,
        max_videos=2,
        rrf_k=1,
    )

    # RRF alone would keep v1 (1.0) over v2 (1/3 + 1/3), but v1 covers one event
    # and v2 covers both, so v2 takes the second slot.
    assert [item.video_id for item in results] == ["v2", "v3"]


def test_top_k_zero_matches_shortlists_nothing() -> None:
    index = _FakeIndex(_MAPPING, _SCORES)
    assert score_videos(index, np.zeros((2, 4), dtype=np.float32), top_k=0) == []
