"""Typed specialist-evidence access through the public data facade."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hcmai.common.schemas import (
    CaptionEvidence,
    FrameContext,
    ObjectEvidence,
    OCREvidence,
    RetrievalSource,
    TranscriptSegment,
)
from hcmai.data.pipeline import DataService
from hcmai.data.stores import CaptionStore, FrameContextStore, ObjectStore


def _write_frames(root: Path) -> Path:
    path = root / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "timestamp_ms": 1_000,
                "image_path": "f1.jpg",
                "width": 10,
                "height": 10,
            }
        ]
    ).to_parquet(path, index=False)
    return path


def _write_specialist_artifacts(root: Path) -> tuple[Path, Path, Path, Path]:
    caption_path = root / "captions.parquet"
    pd.DataFrame(
        [
            CaptionEvidence(
                frame_id="f1",
                video_id="v1",
                frame_idx=10,
                text="A person runs.",
                artifact_version="caption-v1",
                model_name="caption-model",
            ).model_dump(mode="json")
        ]
    ).to_parquet(caption_path, index=False)

    ocr_path = root / "ocr.parquet"
    pd.DataFrame(
        [
            OCREvidence(
                frame_id="f1",
                video_id="v1",
                frame_idx=10,
                raw_text="Café",
                normalized_text="cafe",
                quality_score=0.9,
                region_count=1,
                artifact_version="ocr-v1",
                model_name="ocr-model",
            ).model_dump(mode="json")
        ]
    ).to_parquet(ocr_path, index=False)

    object_dir = root / "objects"
    object_dir.mkdir()
    object_path = object_dir / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "counts_json": json.dumps({"person": 2}),
                "summary": "person x2",
                "detection_count": 2,
                "frame_store_id": None,
                "artifact_version": "object-v1",
                "status": "completed",
                "error_code": None,
                "error_message": None,
            }
        ]
    ).to_parquet(object_path, index=False)
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "detection_index": index,
                "label": "person",
                "confidence": confidence,
                "x_min": 0.1,
                "y_min": 0.2,
                "x_max": 0.3,
                "y_max": 0.4,
            }
            for index, confidence in enumerate((0.9, 0.8))
        ]
    ).to_parquet(object_dir / "detections.parquet", index=False)

    context_path = root / "context.parquet"
    pd.DataFrame(
        [
            FrameContext(
                frame_id="f1",
                video_id="v1",
                frame_idx=10,
                caption_text="A person runs.",
                ocr_text="cafe",
                object_summary="person x2",
                context_text="[CAPTION]\nA person runs.",
                caption_available=True,
                ocr_quality=0.9,
                object_count=2,
                context_version="frame-context-v1",
                caption_version="caption-v1",
                ocr_version="ocr-v1",
                object_version="object-v1",
            ).model_dump(mode="json")
        ]
    ).to_parquet(context_path, index=False)
    return caption_path, ocr_path, object_path, context_path


def test_data_service_exposes_typed_specialist_evidence(tmp_path: Path) -> None:
    frames_path = _write_frames(tmp_path)
    caption_path, ocr_path, object_path, context_path = (
        _write_specialist_artifacts(tmp_path)
    )

    data = DataService.load(
        frames_path,
        {
            RetrievalSource.CAPTION: caption_path,
            RetrievalSource.OCR: ocr_path,
        },
        object_path=object_path,
        context_path=context_path,
    )

    assert isinstance(
        next(data.iter_evidence(RetrievalSource.CAPTION)), CaptionEvidence
    )
    assert isinstance(next(data.iter_evidence(RetrievalSource.OCR)), OCREvidence)
    assert data.get_evidence("f1", RetrievalSource.CAPTION) == "A person runs."
    assert data.get_evidence("f1", RetrievalSource.OCR) == "cafe"

    objects = data.get_object_evidence("f1")
    assert isinstance(objects, ObjectEvidence)
    assert objects.counts == {"person": 2}
    assert [item.confidence for item in objects.detections] == [0.9, 0.8]

    context = data.get_frame_context("f1")
    assert isinstance(context, FrameContext)
    assert context.context_version == "frame-context-v1"


def test_data_service_returns_half_open_transcript_overlap_chronologically(
    tmp_path: Path,
) -> None:
    frames_path = _write_frames(tmp_path)
    transcript_path = tmp_path / "transcripts.parquet"
    segments = [
        TranscriptSegment(
            segment_id="late-boundary",
            video_id="v1",
            segment_index=0,
            start_ms=2_000,
            end_ms=2_500,
            text="excluded at end boundary",
            language="vi",
        ),
        TranscriptSegment(
            segment_id="second",
            video_id="v1",
            segment_index=1,
            start_ms=1_900,
            end_ms=2_100,
            text="second",
            language="vi",
        ),
        TranscriptSegment(
            segment_id="first",
            video_id="v1",
            segment_index=2,
            start_ms=1_000,
            end_ms=1_500,
            text="first",
            language="vi",
        ),
        TranscriptSegment(
            segment_id="early-boundary",
            video_id="v1",
            segment_index=3,
            start_ms=500,
            end_ms=1_000,
            text="excluded at start boundary",
            language="vi",
        ),
    ]
    pd.DataFrame(
        [segment.model_dump(mode="json") for segment in segments]
    ).to_parquet(transcript_path, index=False)

    data = DataService.load(frames_path, transcript_path=transcript_path)

    assert [
        segment.segment_id
        for segment in data.get_transcript_segments("v1", 1_000, 2_000)
    ] == ["first", "second"]
    assert data.get_transcript_segments("missing", 1_000, 2_000) == []


def test_load_evidence_rejects_noncanonical_typed_identity(tmp_path: Path) -> None:
    frames_path = _write_frames(tmp_path)
    caption_path = tmp_path / "captions.parquet"
    pd.DataFrame(
        [
            CaptionEvidence(
                frame_id="f1",
                video_id="wrong-video",
                frame_idx=10,
                text="caption",
                artifact_version="caption-v1",
                model_name="caption-model",
            ).model_dump(mode="json")
        ]
    ).to_parquet(caption_path, index=False)
    data = DataService.load(frames_path)

    with pytest.raises(ValueError, match="canonical identity"):
        data.load_evidence(RetrievalSource.CAPTION, caption_path)


def test_missing_or_null_specialist_evidence_returns_none(tmp_path: Path) -> None:
    frames_path = _write_frames(tmp_path)
    caption_path = tmp_path / "captions.parquet"
    pd.DataFrame(
        [
            CaptionEvidence(
                frame_id="f1",
                video_id="v1",
                frame_idx=10,
                text=None,
                model_revision=None,
                artifact_version="caption-v1",
                model_name="caption-model",
            ).model_dump(mode="json")
        ]
    ).to_parquet(caption_path, index=False)
    data = DataService.load(
        frames_path, {RetrievalSource.CAPTION: caption_path}
    )

    assert data.get_evidence("f1", RetrievalSource.CAPTION) is None
    assert data.get_evidence("missing", RetrievalSource.CAPTION) is None
    assert data.get_evidence("f1", RetrievalSource.OCR) is None
    assert data.get_object_evidence("f1") is None
    assert data.get_frame_context("f1") is None
    assert data.get_transcript_segments("v1", 0, 1) == []


def test_public_stores_validate_malformed_and_incomplete_artifacts(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed-caption.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": None,
                "frame_idx": 10,
                "text": "caption",
                "artifact_version": "caption-v1",
                "model_name": "caption-model",
            }
        ]
    ).to_parquet(malformed, index=False)
    with pytest.raises(ValueError, match="Malformed CaptionEvidence row 0"):
        CaptionStore(malformed)

    object_dir = tmp_path / "incomplete-objects"
    object_dir.mkdir()
    object_path = object_dir / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "counts_json": json.dumps({"person": 1}),
                "summary": "person x1",
                "detection_count": 1,
                "artifact_version": "object-v1",
                "status": "completed",
            }
        ]
    ).to_parquet(object_path, index=False)
    with pytest.raises(FileNotFoundError, match="detections.parquet"):
        ObjectStore(object_path)

    with pytest.raises(FileNotFoundError, match="not a file"):
        FrameContextStore(tmp_path / "missing-context.parquet")

    with pytest.raises(FileNotFoundError, match="Transcript artifact does not exist"):
        DataService.load(
            _write_frames(tmp_path),
            transcript_path=tmp_path / "missing-transcripts",
        )


def test_frame_context_store_exposes_context_text(tmp_path: Path) -> None:
    _, _, _, context_path = _write_specialist_artifacts(tmp_path)
    store = FrameContextStore(context_path)

    assert store.get_text("f1") == "[CAPTION]\nA person runs."
