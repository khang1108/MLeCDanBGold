"""Characterize fields exposed by the pre-migration runtime loaders.

Every artifact is a tiny Parquet fixture created in ``tmp_path``.  The tests
intentionally include nullable evidence values to freeze their meaning as
``None`` instead of fabricated empty strings or scores.
"""

import json
from pathlib import Path

import pandas as pd

from hcmai.common.schemas import (
    CaptionEvidence,
    OCREvidence,
    ProcessingStatus,
    RetrievalSource,
    TranscriptSegment,
)
from hcmai.corpus.models import TranscriptSegment as RuntimeTranscriptSegment
from hcmai.data.pipeline import DataService
from hcmai.corpus.stores.frame import FrameStore


def _write_frame(path: Path) -> None:
    """Write the smallest current-schema frame table accepted by FrameStore."""

    pd.DataFrame(
        [
            {
                "frame_id": "frame-001",
                "video_id": "video-001",
                "frame_idx": 42,
                "timestamp_ms": 1_250,
                "image_path": "data/keyframes/video-001/42.jpg",
                "thumbnail_path": None,
                "width": 1_920,
                "height": 1_080,
            }
        ]
    ).to_parquet(path, index=False)


def test_frame_store_exposes_current_canonical_fields(tmp_path: Path) -> None:
    """Preserve frame identity and image paths through the current loader."""

    path = tmp_path / "artifacts/frame_store/frames.parquet"
    path.parent.mkdir(parents=True)
    _write_frame(path)

    frame = FrameStore(path).get("frame-001")

    assert (
        frame.frame_id,
        frame.video_id,
        frame.frame_idx,
        frame.timestamp_ms,
        frame.image_path,
        frame.thumbnail_path,
    ) == (
        "frame-001",
        "video-001",
        42,
        1_250,
        "data/keyframes/video-001/42.jpg",
        None,
    )


def test_evidence_and_transcript_loaders_preserve_nullable_fields(
    tmp_path: Path,
) -> None:
    """Expose current evidence fields while keeping missing values nullable."""

    frames_path = tmp_path / "frames.parquet"
    _write_frame(frames_path)

    caption_path = tmp_path / "captions.parquet"
    pd.DataFrame(
        [
            CaptionEvidence(
                frame_id="frame-001",
                video_id="video-001",
                frame_idx=42,
                timestamp_ms=1_250,
                text=None,
                artifact_version="caption-v1",
                model_name="caption-model",
                model_revision=None,
                frame_store_id=None,
                status=ProcessingStatus.COMPLETED,
            ).model_dump(mode="json")
        ]
    ).to_parquet(caption_path, index=False)

    ocr_path = tmp_path / "ocr.parquet"
    pd.DataFrame(
        [
            OCREvidence(
                frame_id="frame-001",
                video_id="video-001",
                frame_idx=42,
                timestamp_ms=1_250,
                raw_text=None,
                normalized_text=None,
                artifact_version="ocr-v1",
                model_name="ocr-model",
                model_revision=None,
                frame_store_id=None,
                status=ProcessingStatus.COMPLETED,
            ).model_dump(mode="json")
        ]
    ).to_parquet(ocr_path, index=False)

    object_path = tmp_path / "objects" / "frames.parquet"
    object_path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "frame_id": "frame-001",
                "video_id": "video-001",
                "frame_idx": 42,
                "timestamp_ms": 1_250,
                "counts_json": json.dumps({}),
                "summary": None,
                "detection_count": 0,
                "frame_store_id": None,
                "artifact_version": "object-v1",
                "status": ProcessingStatus.COMPLETED.value,
                "error_code": None,
                "error_message": None,
            }
        ]
    ).to_parquet(object_path, index=False)

    transcript_path = tmp_path / "transcripts.parquet"
    pd.DataFrame(
        [
            TranscriptSegment(
                segment_id="video-001-segment-000",
                video_id="video-001",
                segment_index=0,
                start_ms=1_000,
                end_ms=1_500,
                text="A brief utterance.",
                language="en",
                speaker_id=None,
                confidence=None,
                model_name=None,
                model_revision=None,
            ).model_dump(mode="json")
        ]
    ).to_parquet(transcript_path, index=False)

    data = DataService.load(
        frames_path,
        {
            RetrievalSource.CAPTION: caption_path,
            RetrievalSource.OCR: ocr_path,
        },
        object_path=object_path,
        transcript_path=transcript_path,
    )

    caption = next(data.iter_evidence(RetrievalSource.CAPTION))
    ocr = next(data.iter_evidence(RetrievalSource.OCR))
    transcript = data.get_transcript_segments_at_time("video-001", 1_250)[0]

    assert caption.text is None
    assert data.get_evidence("frame-001", RetrievalSource.CAPTION) is None
    assert ocr.raw_text is None
    assert ocr.normalized_text is None
    assert data.get_evidence("frame-001", RetrievalSource.OCR) is None
    objects = data.get_object_evidence("frame-001")
    assert objects is not None
    assert objects.counts == {}
    assert objects.summary is None
    assert objects.detection_count == 0
    assert transcript == RuntimeTranscriptSegment(
        segment_id="video-001-segment-000",
        video_id="video-001",
        segment_index=0,
        start_ms=1_000,
        end_ms=1_500,
        text="A brief utterance.",
    )
