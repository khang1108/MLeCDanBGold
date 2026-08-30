"""Shortlisting by coverage then RRF must rescore only the kept videos' frames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcmai.common.schemas import SearchFilters
from hcmai.retrieval.retriever.video_scores import score_videos

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
        self.timestamps = mapping["timestamp_ms"].to_numpy(dtype=np.int64)
        self.scored_positions: np.ndarray | None = None

    def search(self, query_vectors: np.ndarray, top_k: int):
        del query_vectors
        order = np.argsort(-self._scores, axis=1)[:, :top_k]
        return np.take_along_axis(self._scores, order, axis=1), order

    def search_filtered(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        filters: SearchFilters | None,
    ):
        """Match the unrestricted DenseIndex protocol used by legacy tests."""

        assert filters is None
        return self.search(query_vectors, top_k)

    def filtered_positions(self, filters: SearchFilters | None):
        """Report that this fixture has no restricted candidate subset."""

        assert filters is None
        return None

    def video_positions(self, video_id: str) -> np.ndarray:
        positions = np.flatnonzero(self.video_ids == video_id)
        return positions[np.argsort(self.frame_idx[positions], kind="stable")]

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
    assert np.array_equal(results[0].timestamps_ms, [200, 400, 800])
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


class _FilteredFakeIndex:
    """Small index fixture proving the shortlist and rescore obey one filter."""

    video_ids = np.array(["V01", "V01", "V02", "V02"], dtype=object)
    frame_ids = np.array(["a", "b", "c", "d"], dtype=object)
    frame_idx = np.array([0, 1, 0, 1])
    timestamps = np.array([0, 1_000, 0, 1_000])

    def search_filtered(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        filters: SearchFilters | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return only V02 positions after asserting caller filter propagation."""

        del query_vectors, top_k
        assert filters is not None
        assert filters.video_ids == ["V02"]
        return np.array([[0.9, 0.8]]), np.array([[2, 3]])

    def filtered_positions(self, filters: SearchFilters | None) -> np.ndarray:
        """Allow only the timestamp-qualified V02 frame to be rescored."""

        assert filters is not None
        assert filters.start_time_ms == 1_000
        return np.array([3], dtype=np.int64)

    def video_positions(self, video_id: str) -> np.ndarray:
        """Return the canonical positions belonging to one fixture video."""

        assert video_id == "V02"
        return np.array([2, 3], dtype=np.int64)

    def score_subset(
        self,
        query_vectors: np.ndarray,
        positions: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray:
        """Return deterministic exact scores for the allowed frame subset."""

        del chunk_size
        assert positions.tolist() == [3]
        return np.full(
            (len(query_vectors), len(positions)),
            0.75,
            dtype=np.float32,
        )


def test_score_videos_respects_video_and_time_filters() -> None:
    """Filter both shortlist evidence and full-video rescoring positions."""

    results = score_videos(
        _FilteredFakeIndex(),
        np.array([[1.0, 0.0]], dtype=np.float32),
        top_k=10,
        max_videos=10,
        filters=SearchFilters(video_ids=["V02"], start_time_ms=1_000),
    )

    assert [result.video_id for result in results] == ["V02"]
    assert results[0].frame_ids.tolist() == ["d"]
