"""Regression coverage for Phase A KIS aligned-path materialization."""

from __future__ import annotations

import pytest

from hcmai.corpus import Frame
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.temporal import AlignedPath


class _Data:
    """Expose the minimal canonical data surface used by the materializer."""

    def frame(self, frame_id: str) -> Frame:
        """Return one canonical representative frame."""

        return Frame(
            frame_id=frame_id,
            video_id="V01",
            frame_idx=10,
            timestamp_ms=1_000,
            image_path="frame-1.jpg",
            fps=29.97,
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
    assert result.fps == 29.97
    assert "context" not in result.metadata.model_dump()


class _MultiFrameData:
    """Expose canonical frames with independently verifiable metadata."""

    frames = {
        "frame-1": Frame(
            frame_id="frame-1",
            video_id="V01",
            frame_idx=10,
            timestamp_ms=1_000,
            image_path="frame-1.jpg",
        ),
        "frame-2": Frame(
            frame_id="frame-2",
            video_id="V01",
            frame_idx=20,
            timestamp_ms=2_000,
            image_path="frame-2.jpg",
        ),
        "alias-frame": Frame(
            frame_id="frame-2",
            video_id="V01",
            frame_idx=20,
            timestamp_ms=2_000,
            image_path="frame-2.jpg",
        ),
        "foreign-frame": Frame(
            frame_id="foreign-frame",
            video_id="V02",
            frame_idx=20,
            timestamp_ms=2_000,
            image_path="foreign-frame.jpg",
        ),
    }

    def frame(self, frame_id: str) -> Frame:
        """Return the frame associated with one canonical identifier."""

        return self.frames[frame_id]


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (
            AlignedPath(
                video_id="V01",
                score=1.0,
                frame_ids=("frame-1", "alias-frame"),
                frame_idxs=(10, 20),
                timestamps_ms=(1_000, 2_000),
            ),
            "frame_id",
        ),
        (
            AlignedPath(
                video_id="V01",
                score=1.0,
                frame_ids=("frame-1", "foreign-frame"),
                frame_idxs=(10, 20),
                timestamps_ms=(1_000, 2_000),
            ),
            "video_id",
        ),
        (
            AlignedPath(
                video_id="V01",
                score=1.0,
                frame_ids=("frame-1", "frame-2"),
                frame_idxs=(10, 99),
                timestamps_ms=(1_000, 2_000),
            ),
            "frame_idx",
        ),
        (
            AlignedPath(
                video_id="V01",
                score=1.0,
                frame_ids=("frame-1", "frame-2"),
                frame_idxs=(10, 20),
                timestamps_ms=(1_000, 9_999),
            ),
            "timestamp",
        ),
    ],
)
def test_validate_aligned_path_rejects_later_canonical_mismatch(
    path: AlignedPath,
    message: str,
) -> None:
    """Validate every frame, including entries after a valid first frame."""

    with pytest.raises(ValueError, match=message):
        SearchMaterializer(_MultiFrameData()).validate_aligned_path(path)


def test_build_kis_result_selects_the_upper_middle_canonical_frame() -> None:
    """Use the existing representative-frame rule after validating every entry."""

    class _KisMultiFrameData(_MultiFrameData):
        @staticmethod
        def caption(frame_id: str) -> None:
            return None

        @staticmethod
        def ocr(frame_id: str) -> None:
            return None

        @staticmethod
        def objects(frame_id: str) -> tuple[str, ...]:
            return ()

        @staticmethod
        def transcript(video_id: str, start_ms: int, end_ms: int) -> None:
            return None

        @staticmethod
        def title(video_id: str) -> None:
            return None

    result = SearchMaterializer(_KisMultiFrameData()).build_kis_result(
        AlignedPath(
            video_id="V01",
            score=0.73,
            frame_ids=("frame-1", "frame-2"),
            frame_idxs=(10, 20),
            timestamps_ms=(1_000, 2_000),
        )
    )

    assert result.frame_id == "frame-2"
    assert result.frame_idx == 20
    assert result.timestamp_ms == 2_000


def test_build_trake_path_preserves_all_path_arrays_and_score() -> None:
    """Project TRAKE output without altering canonical coordinates or score."""

    path = AlignedPath(
        video_id="V01",
        score=0.73,
        frame_ids=("frame-1", "frame-2"),
        frame_idxs=(10, 20),
        timestamps_ms=(1_000, 2_000),
    )

    result = SearchMaterializer.build_trake_path(path)

    assert result.video_id == "V01"
    assert result.score == 0.73
    assert result.frame_ids == ["frame-1", "frame-2"]
    assert result.frame_idxs == [10, 20]
    assert result.timestamps_ms == [1_000, 2_000]
