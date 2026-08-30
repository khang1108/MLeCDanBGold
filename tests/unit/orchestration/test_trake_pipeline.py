"""Tests for TRAKE projection through the shared alignment service."""

from __future__ import annotations

from types import SimpleNamespace

from hcmai.common.schemas import AlignmentPath, FrameRecord, TRAKERequest
from hcmai.orchestration.workflows.trake import TRAKEPipeline


class FakeAlignment:
    """Return one known canonical alignment path without dense retrieval."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []

    def align(self, plan, *, max_paths):
        """Record the constructed plan and return a path for its event IDs."""

        self.calls.append((plan, max_paths))
        frames = (
            FrameRecord(
                frame_id="f0",
                video_id="V01",
                frame_idx=10,
                timestamp_ms=1_000,
                image_path="f0.jpg",
                width=640,
                height=360,
            ),
            FrameRecord(
                frame_id="f1",
                video_id="V01",
                frame_idx=20,
                timestamp_ms=2_000,
                image_path="f1.jpg",
                width=640,
                height=360,
            ),
        )
        return SimpleNamespace(
            paths=(
                AlignmentPath(
                    path_id="path-1",
                    video_id="V01",
                    frames=frames,
                    event_ids=("e0", "e1"),
                    score=1.5,
                ),
            )
        )


def test_trake_pipeline_uses_shared_alignment_service() -> None:
    """Keep TRAKE's public response while delegating all temporal logic."""

    alignment = FakeAlignment()
    response = TRAKEPipeline(alignment).execute(
        TRAKERequest(
            query="first then second",
            events=["first", "second"],
            top_k=5,
        )
    )

    assert response.total_results == 1
    assert response.submissions[0].frame_ids == ["f0", "f1"]
    assert [event.text for event in alignment.calls[0][0].events] == [
        "first",
        "second",
    ]
    assert alignment.calls[0][1] == 5
