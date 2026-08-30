"""Tests for TRAKE projection through the shared alignment service."""

from __future__ import annotations

from types import SimpleNamespace

from hcmai.common.schemas import FrameRecord, TRAKERequest
from hcmai.orchestration.workflows.trake import TRAKEPipeline
from hcmai.temporal import AlignedPath


class FakeAlignment:
    """Return one known canonical alignment path without dense retrieval."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []

    def search(self, events, *, top_k):
        """Record event text and return one canonical-ID path."""

        self.calls.append((tuple(events), top_k))
        return SimpleNamespace(
            paths=(
                AlignedPath(
                    video_id="V01",
                    score=1.5,
                    frame_ids=("f0", "f1"),
                    frame_idxs=(10, 20),
                    timestamps_ms=(1_000, 2_000),
                ),
            ),
        )


class FakeData:
    """Canonical frame lookup used by TRAKE response projection."""

    _frames = {
        "f0": FrameRecord(
            frame_id="f0",
            video_id="V01",
            frame_idx=10,
            timestamp_ms=1_000,
            image_path="f0.jpg",
            width=640,
            height=360,
        ),
        "f1": FrameRecord(
            frame_id="f1",
            video_id="V01",
            frame_idx=20,
            timestamp_ms=2_000,
            image_path="f1.jpg",
            width=640,
            height=360,
        ),
    }

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Resolve one canonical frame ID for submission materialization."""

        return self._frames[frame_id]


def test_trake_pipeline_uses_shared_alignment_service() -> None:
    """Keep TRAKE's public response while delegating all temporal logic."""

    alignment = FakeAlignment()
    response = TRAKEPipeline(FakeData(), alignment).execute(
        TRAKERequest(
            query="first then second",
            events=["first", "second"],
            top_k=5,
        )
    )

    assert response.total_results == 1
    assert response.submissions[0].frame_ids == ["f0", "f1"]
    assert alignment.calls[0][0] == ("first", "second")
    assert alignment.calls[0][1] == 5
