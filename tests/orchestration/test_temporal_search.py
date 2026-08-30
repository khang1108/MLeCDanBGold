"""Tests for timed temporal search and canonical path materialization."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hcmai.common.config import AlignmentConfig
from hcmai.common.schemas import FrameRecord
from hcmai.orchestration.temporal_search import TemporalSearchService
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class FakeRetrieval:
    """Return a controllable full-corpus event/frame score matrix."""

    def __init__(self, scores: list[VideoEventScores]) -> None:
        self.scores = scores
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def score_event_videos(self, events, *, chunk_size):
        """Capture normalized events and chunk size while returning test scores."""

        self.calls.append((tuple(events), chunk_size))
        return self.scores


class FakeData:
    """Expose canonical frame records for path materialization."""

    frames = {
        "v1-f0": FrameRecord(
            frame_id="v1-f0",
            video_id="v1",
            frame_idx=0,
            timestamp_ms=0,
            image_path="v1-f0.jpg",
            width=640,
            height=360,
        ),
        "v1-f1": FrameRecord(
            frame_id="v1-f1",
            video_id="v1",
            frame_idx=1,
            timestamp_ms=1_000,
            image_path="v1-f1.jpg",
            width=640,
            height=360,
        ),
    }

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Return the canonical frame identified by a ranked path entry."""

        return self.frames[frame_id]


def _scores() -> VideoEventScores:
    """Return a two-event score matrix with one unique chronological path."""

    return VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["v1-f0", "v1-f1"], dtype=object),
        frame_idx=np.array([0, 1]),
        timestamps_ms=np.array([0, 1_000]),
        scores=np.array([[0.9, 0.1], [0.1, 0.8]]),
    )


def _service(scores: list[VideoEventScores]) -> TemporalSearchService:
    """Build the temporal search service under deterministic defaults."""

    return TemporalSearchService(
        FakeData(),
        FakeRetrieval(scores),
        AlignmentConfig(lambda_gap=0.0, chunk_size=123),
    )


def test_temporal_search_returns_canonical_paths_and_timings() -> None:
    """Materialize aligned paths without recomputing canonical identity."""

    service = _service([_scores()])

    search = service.search(["e1", "e2"], top_k=2)

    assert search.paths[0].frame_ids == ("v1-f0", "v1-f1")
    assert search.paths[0].frame_idxs == (0, 1)
    assert search.paths[0].timestamps_ms == (0, 1_000)
    assert search.retrieval_ms >= 0
    assert search.alignment_ms >= 0


def test_temporal_search_rejects_noncanonical_frame_idx() -> None:
    """Reject retrieval metadata that drifts from canonical frame identity."""

    invalid = replace(_scores(), frame_idx=np.array([99, 1]))

    with pytest.raises(ValueError, match="frame_idx"):
        _service([invalid]).search(["e1", "e2"], top_k=2)
