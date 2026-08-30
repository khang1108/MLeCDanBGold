"""Tests for KIS projection of canonical temporal alignment paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hcmai.common.schemas import FrameRecord, SearchFilters, SearchRequest
from hcmai.orchestration.workflows.base import TaskPipelineRequestError
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.temporal import AlignedPath


class FakeAlignment:
    """Return a fixed three-event path and retain KIS's event text."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []

    def search(self, events, *, top_k):
        """Return one canonical-ID path without reaching a model or index."""

        self.calls.append((events, top_k))
        return SimpleNamespace(
            paths=(
                AlignedPath(
                    video_id="V01",
                    score=2.4,
                    frame_ids=("f0", "f1", "f2"),
                    frame_idxs=(0, 1, 2),
                    timestamps_ms=(0, 1_000, 2_000),
                ),
            ),
        )


class FakeData:
    """Canonical materializer fixture for the frames returned by alignment."""

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Resolve the requested synthetic frame through canonical metadata."""

        index = int(frame_id[1:])
        return FrameRecord(
            frame_id=frame_id,
            video_id="V01",
            frame_idx=index,
            timestamp_ms=index * 1_000,
            image_path=f"{frame_id}.jpg",
            width=640,
            height=360,
        )

    def get_evidence(self, frame_id, source):
        """Keep specialist evidence optional in this visual-only baseline."""

        del frame_id, source
        return None


def test_kis_returns_middle_frame_and_preserves_alignment_path() -> None:
    """Use path score and retain all path IDs in one KIS result."""

    alignment = FakeAlignment()
    response = KISPipeline(FakeData(), alignment).execute(
        SearchRequest(
            query="first. second. third.",
            events=["first", "second", "third"],
            top_k=5,
        )
    )

    assert response.total_results == 1
    assert response.results[0].frame_id == "f1"
    assert response.results[0].frame_ids == ["f0", "f1", "f2"]
    assert response.results[0].scores.final == 2.4
    assert "rerank" not in response.trace.stages
    assert alignment.calls[0][0] == ("first", "second", "third")
    assert alignment.calls[0][1] == 5


def test_kis_rejects_ambiguous_minimum_score_before_alignment() -> None:
    """Never silently apply a single-frame threshold to a multi-event path."""

    alignment = FakeAlignment()
    pipeline = KISPipeline(FakeData(), alignment)

    with pytest.raises(TaskPipelineRequestError, match="min_score"):
        pipeline.execute(
            SearchRequest(
                query="first then second",
                filters=SearchFilters(min_score=0.5),
            )
        )

    assert alignment.calls == []
