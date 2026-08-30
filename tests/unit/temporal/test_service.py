"""Tests for stateless temporal alignment and canonical metadata validation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hcmai.common.config import AlignmentConfig
from hcmai.common.schemas import (
    AlignmentEvent,
    AlignmentPlan,
    FrameRecord,
    SearchFilters,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.service import TemporalAlignmentService


class FakeRetrieval:
    """Return one controllable event/frame score matrix."""

    def __init__(self, scores: list[VideoEventScores]) -> None:
        self.scores = scores
        self.calls: list[tuple[list[str], object]] = []

    def score_event_videos(self, events, filters=None, **kwargs):
        """Capture planner input while behaving like the generic service API."""

        del kwargs
        self.calls.append((list(events), filters))
        return self.scores


class FakeData:
    """Expose two canonical frames for alignment materialization."""

    frames = {
        "f0": FrameRecord(
            frame_id="f0",
            video_id="V01",
            frame_idx=0,
            timestamp_ms=0,
            image_path="f0.jpg",
            width=640,
            height=360,
        ),
        "f1": FrameRecord(
            frame_id="f1",
            video_id="V01",
            frame_idx=1,
            timestamp_ms=1_000,
            image_path="f1.jpg",
            width=640,
            height=360,
        ),
    }

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Return the canonical frame identified by the score metadata."""

        return self.frames[frame_id]


def _plan() -> AlignmentPlan:
    """Build the two-event plan shared by the alignment fixtures."""

    return AlignmentPlan(
        events=(
            AlignmentEvent(event_id="e0", text="first", order=0),
            AlignmentEvent(event_id="e1", text="second", order=1),
        )
    )


def _scores() -> VideoEventScores:
    """Return one canonical score matrix with a unique monotonic path."""

    return VideoEventScores(
        video_id="V01",
        frame_ids=np.array(["f0", "f1"], dtype=object),
        frame_idx=np.array([0, 1]),
        timestamps_ms=np.array([0, 1_000]),
        scores=np.array([[0.9, 0.1], [0.1, 0.8]]),
    )


def _service(scores: list[VideoEventScores]) -> TemporalAlignmentService:
    """Build the service under the default deterministic alignment settings."""

    return TemporalAlignmentService(FakeData(), FakeRetrieval(scores), AlignmentConfig(lambda_gap=0.0))


def test_alignment_service_materializes_canonical_path() -> None:
    """Expose only path frames that agree with canonical frame metadata."""

    service = _service([_scores()])

    result = service.align(_plan(), max_paths=5)

    assert len(result.paths) == 1
    assert [frame.frame_id for frame in result.paths[0].frames] == ["f0", "f1"]
    assert result.paths[0].event_ids == ("e0", "e1")
    assert result.candidate_video_count == 1


def test_alignment_service_rejects_mismatched_score_matrix_rows() -> None:
    """Reject a retriever that does not return one row for every plan event."""

    invalid = replace(_scores(), scores=np.array([[0.9, 0.1]]))

    with pytest.raises(ValueError, match="shape"):
        _service([invalid]).align(_plan(), max_paths=1)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (replace(_scores(), video_id="wrong-video"), "video identity"),
        (replace(_scores(), frame_idx=np.array([99, 1])), "frame_idx"),
        (replace(_scores(), timestamps_ms=np.array([99, 1_000])), "timestamp"),
    ],
)
def test_alignment_service_rejects_noncanonical_score_metadata(
    invalid: VideoEventScores,
    message: str,
) -> None:
    """Protect every canonical identity field from retrieval-artifact drift."""

    with pytest.raises(ValueError, match=message):
        _service([invalid]).align(_plan(), max_paths=1)


def test_alignment_service_requires_a_positive_path_budget() -> None:
    """Fail early rather than leaving DP behavior undefined for zero paths."""

    with pytest.raises(ValueError, match="greater than zero"):
        _service([_scores()]).align(_plan(), max_paths=0)


def test_alignment_service_rejects_ambiguous_minimum_score_filter() -> None:
    """Protect future non-KIS callers from silently ignoring ``min_score``."""

    plan = _plan().model_copy(update={"filters": SearchFilters(min_score=0.5)})

    with pytest.raises(ValueError, match="min_score"):
        _service([_scores()]).align(plan, max_paths=1)
