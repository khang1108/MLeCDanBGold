"""Regression coverage for Phase A KIS aligned-path materialization."""

from __future__ import annotations

from hcmai.common.schemas import FrameRecord, RetrievalSource
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.temporal import AlignedPath


class _Data:
    """Expose the minimal canonical data surface used by the materializer."""

    video_metadata_store = None

    def __init__(self) -> None:
        """Record requested evidence modalities for the baseline assertion."""

        self.sources: list[RetrievalSource] = []

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Return one canonical representative frame."""

        return FrameRecord(
            frame_id=frame_id,
            video_id="V01",
            frame_idx=10,
            timestamp_ms=1_000,
            image_path="frame-1.jpg",
            width=640,
            height=360,
        )

    def get_evidence(self, frame_id: str, source: RetrievalSource) -> str | None:
        """Record specialist evidence reads and return no optional text."""

        assert frame_id == "frame-1"
        self.sources.append(source)
        return None

    def get_object_counts(self, frame_id: str) -> None:
        """Return no optional object evidence."""

        assert frame_id == "frame-1"
        return None

    def get_transcript_segments_at_time(
        self,
        video_id: str,
        timestamp_ms: int,
    ) -> list[object]:
        """Return no ASR segments for the representative timestamp."""

        assert (video_id, timestamp_ms) == ("V01", 1_000)
        return []


def test_materializer_exposes_raw_path_score_without_context_retrieval() -> None:
    """Keep detached context scoring out of the Phase A KIS projection."""

    data = _Data()
    result = SearchMaterializer(data).build_kis_result(
        AlignedPath(
            video_id="V01",
            score=0.73,
            frame_ids=("frame-1",),
            frame_idxs=(10,),
            timestamps_ms=(1_000,),
        )
    )

    assert result.score == 0.73
    assert data.sources == [RetrievalSource.CAPTION, RetrievalSource.OCR]
    assert "context" not in result.metadata.model_dump()
