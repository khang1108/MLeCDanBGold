"""Smoke test for TRAKE video shortlisting and per-video rescoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcmai.common.schemas.retrieval import RetrievalCandidate
from hcmai.agents.trake import event_video_scores

_MAPPING = pd.DataFrame(
    {
        "embedding_index": [0, 1, 2, 3, 4],
        "frame_id": ["v1_b", "v2_a", "v1_a", "v3_a", "v1_c"],
        "video_id": ["v1", "v2", "v1", "v3", "v1"],
        "frame_idx": [10, 7, 5, 3, 20],
        "timestamp_ms": [400, 280, 200, 120, 800],
    }
)
_SCORES = np.array(
    [[0.1, 0.9, 0.2, 0.3, 0.4], [0.5, 0.1, 0.8, 0.2, 0.6]], dtype=np.float32
)


class _FakeIndex:
    """Exact index whose search returns every frame in descending score order."""

    mapping = _MAPPING

    class index:  # noqa: N801 - mirrors DenseIndex.index.ntotal
        ntotal = 5

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(-_SCORES, axis=1)[:, :top_k]
        return np.take_along_axis(_SCORES, order, axis=1), order


class _FakeBatch:
    def __init__(self, texts: list[str]) -> None:
        self.vectors = np.zeros((len(texts), 4), dtype=np.float32)


class _FakeRetrieval:
    """Shortlist only v1 and v3, so v2 must never reach the aligner."""

    visual_index = _FakeIndex()

    def search(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        frame_id = "v1_b" if query == "first event" else "v3_a"
        return [RetrievalCandidate(frame_id=frame_id)]

    def encode_text_batch(self, texts: list[str], source_family: str) -> _FakeBatch:
        assert source_family == "visual"
        return _FakeBatch(texts)


def test_shortlisted_videos_are_rescored_in_canonical_frame_order() -> None:
    results = event_video_scores(
        _FakeRetrieval(),
        ["first event", "second event"],
    )
    assert [item.video_id for item in results] == ["v1", "v3"]
    assert results[0].frame_ids == ("v1_a", "v1_b", "v1_c")
    assert results[0].frame_idx == (5, 10, 20)
    assert np.array_equal(results[0].timestamps_ms, [200.0, 400.0, 800.0])
    assert np.array_equal(results[0].scores, _SCORES[:, [2, 0, 4]])
    assert np.array_equal(results[1].scores, _SCORES[:, [3]])
