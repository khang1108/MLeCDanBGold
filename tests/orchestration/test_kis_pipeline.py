"""Regression coverage for the Phase A KIS aligned-path projection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hcmai.api.contracts import SearchRequest
from hcmai.common.schemas import FrameRecord, RetrievalSource
from hcmai.orchestration.temporal_search import TemporalSearchResult
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.temporal import AlignedPath


class FakeAlignment:
    """Return one five-event canonical path with fixed shared timings."""

    def __init__(self) -> None:
        """Record calls so the test verifies the shared-service boundary."""

        self.calls: list[tuple[tuple[str, ...], int]] = []

    def search(
        self,
        events: tuple[str, ...],
        *,
        top_k: int,
    ) -> TemporalSearchResult:
        """Return a path without invoking retrieval or dynamic programming."""

        self.calls.append((events, top_k))
        return TemporalSearchResult(
            paths=(
                AlignedPath(
                    video_id="V01",
                    score=2.73,
                    frame_ids=("f0", "f1", "f2", "f3", "f4"),
                    frame_idxs=(0, 1, 2, 3, 4),
                    timestamps_ms=(0, 1_000, 2_000, 3_000, 4_000),
                ),
            ),
            retrieval_ms=12.5,
            alignment_ms=7.25,
        )


class FakeData:
    """Expose only the canonical data reads KIS materialization needs."""

    video_metadata_store = SimpleNamespace(
        get=lambda video_id: SimpleNamespace(title="Cooking Episode")
    )

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Resolve a synthetic canonical frame record."""

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

    def get_evidence(self, frame_id: str, source: RetrievalSource) -> str | None:
        """Provide frame-native evidence only for the representative frame."""

        assert frame_id == "f2"
        return {
            RetrievalSource.CAPTION: "chef coats ingredient",
            RetrievalSource.OCR: "FLOUR",
        }.get(source)

    def get_object_counts(self, frame_id: str) -> dict[str, int] | None:
        """Return unordered labels to verify deterministic metadata ordering."""

        assert frame_id == "f2"
        return {"person": 1, "bowl": 1}

    def get_transcript_segments_at_time(
        self,
        video_id: str,
        timestamp_ms: int,
    ) -> list[SimpleNamespace]:
        """Return transcript evidence only at the representative timestamp."""

        assert (video_id, timestamp_ms) == ("V01", 2_000)
        return [SimpleNamespace(text=" coat it with flour ")]


def test_kis_projects_middle_frame_and_materializes_representative_metadata() -> None:
    """Keep every aligned path entry while selecting the deterministic midpoint."""

    alignment = FakeAlignment()
    response = KISPipeline(FakeData(), alignment).execute(
        SearchRequest(query="e1\ne2\ne3\ne4\ne5", top_k=1)
    )
    result = response.results[0]

    assert response.events == ["e1", "e2", "e3", "e4", "e5"]
    assert result.frame_id == "f2"
    assert result.frame_idx == 2
    assert result.frame_ids == ["f0", "f1", "f2", "f3", "f4"]
    assert result.timestamps_ms == [0, 1_000, 2_000, 3_000, 4_000]
    assert len(result.thumbnail_urls) == 5
    assert result.score == pytest.approx(2.73)
    assert result.metadata.title == "Cooking Episode"
    assert result.metadata.caption == "chef coats ingredient"
    assert result.metadata.ocr == "FLOUR"
    assert result.metadata.objects == ["bowl", "person"]
    assert result.metadata.asr == "coat it with flour"
    assert response.latency.retrieval_ms == pytest.approx(12.5)
    assert response.latency.alignment_ms == pytest.approx(7.25)
    assert alignment.calls == [(("e1", "e2", "e3", "e4", "e5"), 1)]
