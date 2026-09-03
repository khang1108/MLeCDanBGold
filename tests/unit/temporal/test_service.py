"""Tests for timed temporal search and canonical metadata validation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hcmai.common.config import AlignmentConfig
from hcmai.corpus import Frame
from hcmai.orchestration.workflows.temporal_search import TemporalSearchService
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class FakeRetrieval:
    """Return one controllable event/frame score matrix."""

    def __init__(self, scores: list[VideoEventScores]) -> None:
        self.scores = scores
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def score_event_videos(self, events, *, chunk_size):
        """Capture normalized events while behaving like the temporal API."""

        self.calls.append((tuple(events), chunk_size))
        return self.scores


class FakeData:
    """Expose two canonical frames for alignment materialization."""

    frames = {
        "f0": Frame(
            frame_id="f0",
            video_id="V01",
            frame_idx=0,
            timestamp_ms=0,
            image_path="f0.jpg",
        ),
        "f1": Frame(
            frame_id="f1",
            video_id="V01",
            frame_idx=1,
            timestamp_ms=1_000,
            image_path="f1.jpg",
        ),
    }

    def frame(self, frame_id: str) -> Frame:
        """Return the canonical frame identified by the score metadata."""

        return self.frames[frame_id]


def _events() -> tuple[str, ...]:
    """Build the two-event input shared by the alignment fixtures."""

    return ("first", "second")


def _scores() -> VideoEventScores:
    """Return one canonical score matrix with a unique monotonic path."""

    return VideoEventScores(
        video_id="V01",
        frame_ids=np.array(["f0", "f1"], dtype=object),
        frame_idx=np.array([0, 1]),
        timestamps_ms=np.array([0, 1_000]),
        scores=np.array([[0.9, 0.1], [0.1, 0.8]]),
    )


def _service(scores: list[VideoEventScores]) -> TemporalSearchService:
    """Build the service under deterministic temporal-search settings."""

    return TemporalSearchService(
        FakeData(),
        FakeRetrieval(scores),
        AlignmentConfig(lambda_gap=0.0, chunk_size=123),
    )


def test_temporal_search_returns_canonical_id_path() -> None:
    """Expose canonical frame identity and timings for one aligned path."""

    service = _service([_scores()])

    result = service.search(_events(), top_k=5)

    assert result.paths[0].video_id == "V01"
    assert result.paths[0].frame_ids == ("f0", "f1")
    assert result.paths[0].frame_idxs == (0, 1)
    assert result.paths[0].timestamps_ms == (0, 1_000)
    assert result.retrieval_ms >= 0
    assert result.alignment_ms >= 0


def test_temporal_search_rejects_mismatched_score_matrix_rows() -> None:
    """Reject a retriever that does not return one row for every plan event."""

    invalid = replace(_scores(), scores=np.array([[0.9, 0.1]]))

    with pytest.raises(ValueError, match="shape"):
        _service([invalid]).search(_events(), top_k=1)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (replace(_scores(), video_id="wrong-video"), "video identity"),
        (replace(_scores(), frame_idx=np.array([99, 1])), "frame_idx"),
        (replace(_scores(), timestamps_ms=np.array([99, 1_000])), "timestamp"),
    ],
)
def test_temporal_search_rejects_noncanonical_score_metadata(
    invalid: VideoEventScores,
    message: str,
) -> None:
    """Protect every canonical identity field from retrieval-artifact drift."""

    with pytest.raises(ValueError, match=message):
        _service([invalid]).search(_events(), top_k=1)


def test_temporal_search_requires_a_positive_path_budget() -> None:
    """Fail early rather than leaving DP behavior undefined for zero paths."""

    with pytest.raises(ValueError, match="greater than zero"):
        _service([_scores()]).search(_events(), top_k=0)
