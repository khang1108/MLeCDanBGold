"""Regression coverage for Phase A KIS aligned-path materialization."""

from __future__ import annotations

from hcmai.common.schemas import FrameRecord
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.temporal import AlignedPath


class _Data:
    """Expose the minimal canonical data surface used by the materializer."""

    def frame(self, frame_id: str) -> FrameRecord:
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

    def caption(self, frame_id: str) -> str | None:
        """Return no optional caption evidence."""

        assert frame_id == "frame-1"
        return None

    def ocr(self, frame_id: str) -> str | None:
        """Return no optional OCR evidence."""

        assert frame_id == "frame-1"
        return None

    def objects(self, frame_id: str) -> tuple[str, ...]:
        """Return no optional object evidence."""

        assert frame_id == "frame-1"
        return ()

    def transcript(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
    ) -> str | None:
        """Return no ASR segments for the representative timestamp."""

        assert (video_id, start_ms, end_ms) == ("V01", 1_000, 1_001)
        return None

    def title(self, video_id: str) -> str | None:
        """Return no optional organizer title."""

        assert video_id == "V01"
        return None


def test_materializer_exposes_raw_path_score_without_context_retrieval() -> None:
    """Keep detached context scoring out of the Phase A KIS projection."""

    corpus = _Data()
    result = SearchMaterializer(corpus).build_kis_result(
        AlignedPath(
            video_id="V01",
            score=0.73,
            frame_ids=("frame-1",),
            frame_idxs=(10,),
            timestamps_ms=(1_000,),
        )
    )

    assert result.score == 0.73
    assert "context" not in result.metadata.model_dump()
