"""Typed artifact validation and public Corpus evidence projections."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hcmai.corpus import Corpus
from hcmai.corpus.stores import CaptionStore, FrameContextStore, ObjectStore
from offline.enrichment.object_artifacts import write_object_artifacts
from hcmai.retrieval.models import RetrievalSource
from offline.enrichment.caption.models import CaptionEvidence
from offline.enrichment.context.models import FrameContext
from offline.enrichment.models import ProcessingStatus
from offline.enrichment.objects.models import ObjectDetection, ObjectEvidence
from offline.enrichment.ocr.models import OCREvidence
from offline.enrichment.transcripts.models import TranscriptSegment


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
                timestamp_ms=1_000,
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
                timestamp_ms=1_000,
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
                "timestamp_ms": 1_000,
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
                "frame_idx": 10,
                "timestamp_ms": 1_000,
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
                timestamp_ms=1_000,
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


def test_corpus_projects_runtime_evidence_without_hiding_specialist_stores(
    tmp_path: Path,
) -> None:
    frames_path = _write_frames(tmp_path)
    caption_path, ocr_path, object_path, context_path = (
        _write_specialist_artifacts(tmp_path)
    )

    corpus = Corpus.open(
        frames_path,
        {
            RetrievalSource.CAPTION: caption_path,
            RetrievalSource.OCR: ocr_path,
        },
        object_counts_path=object_path,
    )

    assert corpus.caption("f1") == "A person runs."
    assert corpus.ocr("f1") == "cafe"
    assert corpus.objects("f1") == ("person",)

    context = FrameContextStore(context_path).get("f1")
    assert not isinstance(context, FrameContext)
    assert context.context_version == "frame-context-v1"


def test_corpus_projects_runtime_metadata_from_specialist_stores(
    tmp_path: Path,
) -> None:
    """Join frame-native evidence, timeline ASR, and video metadata per frame."""

    frames_path = _write_frames(tmp_path)
    caption_path, ocr_path, object_path, _ = _write_specialist_artifacts(tmp_path)
    transcript_path = tmp_path / "transcripts.parquet"
    segment = TranscriptSegment(
        segment_id="v1_segment_000000",
        video_id="v1",
        segment_index=0,
        start_ms=900,
        end_ms=1_100,
        text="A runner passes the cafe.",
        language="en",
    )
    pd.DataFrame([segment.model_dump(mode="json")]).to_parquet(
        transcript_path, index=False
    )
    metadata_root = tmp_path / "media-info"
    metadata_root.mkdir()
    (metadata_root / "v1.json").write_text(
        json.dumps(
            {
                "title": "Morning run",
                "watch_url": "https://example.test/watch?v=v1",
            }
        ),
        encoding="utf-8",
    )

    corpus = Corpus.open(
        frames_path,
        {
            RetrievalSource.CAPTION: caption_path,
            RetrievalSource.OCR: ocr_path,
        },
        object_counts_path=object_path,
        transcript_path=transcript_path,
        video_metadata_path=metadata_root,
    )

    assert corpus.caption("f1") == "A person runs."
    assert corpus.ocr("f1") == "cafe"
    assert corpus.objects("f1") == ("person",)
    assert corpus.title("v1") == "Morning run"
    assert corpus.transcript_segments("v1", 1_000, 1_001) == (
        corpus.transcript_segments("v1", 1_000, 1_001)[0],
    )


def test_corpus_returns_half_open_transcript_overlap_chronologically(
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

    corpus = Corpus.open(frames_path, transcript_path=transcript_path)

    assert [
        segment.segment_id
        for segment in corpus.transcript_segments("v1", 1_000, 2_000)
    ] == ["first", "second"]
    assert corpus.transcript_segments("missing", 1_000, 2_000) == ()
    assert corpus.transcript_segments("v1", 1_200, 1_200) == ()

    with pytest.raises(ValueError, match="start_ms.*non-negative"):
        corpus.transcript_segments("v1", -1, 1_000)
    with pytest.raises(ValueError, match="end_ms.*start_ms"):
        corpus.transcript_segments("v1", 2_000, 1_000)


def test_load_evidence_rejects_noncanonical_typed_identity(tmp_path: Path) -> None:
    frames_path = _write_frames(tmp_path)
    caption_path = tmp_path / "captions.parquet"
    pd.DataFrame(
        [
            CaptionEvidence(
                frame_id="f1",
                video_id="wrong-video",
                frame_idx=10,
                timestamp_ms=1_000,
                text="caption",
                artifact_version="caption-v1",
                model_name="caption-model",
            ).model_dump(mode="json")
        ]
    ).to_parquet(caption_path, index=False)
    with pytest.raises(ValueError, match="canonical identity"):
        Corpus.open(frames_path, {RetrievalSource.CAPTION: caption_path})


def test_missing_or_null_specialist_evidence_returns_none(tmp_path: Path) -> None:
    frames_path = _write_frames(tmp_path)
    caption_path = tmp_path / "captions.parquet"
    pd.DataFrame(
        [
            CaptionEvidence(
                frame_id="f1",
                video_id="v1",
                frame_idx=10,
                timestamp_ms=1_000,
                text=None,
                model_revision=None,
                artifact_version="caption-v1",
                model_name="caption-model",
            ).model_dump(mode="json")
        ]
    ).to_parquet(caption_path, index=False)
    corpus = Corpus.open(
        frames_path, {RetrievalSource.CAPTION: caption_path}
    )

    assert corpus.caption("f1") is None
    assert corpus.caption("missing") is None
    assert corpus.ocr("f1") is None
    assert corpus.objects("f1") == ()
    assert corpus.transcript_segments("v1", 0, 1) == ()


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
                "timestamp_ms": 1_000,
                "text": "caption",
                "artifact_version": "caption-v1",
                "model_name": "caption-model",
            }
        ]
    ).to_parquet(malformed, index=False)
    with pytest.raises(ValueError, match="video_id.*canonical representation"):
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
                "timestamp_ms": 1_000,
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
        Corpus.open(
            _write_frames(tmp_path),
            transcript_path=tmp_path / "missing-transcripts",
        )


def test_frame_context_store_exposes_context_text(tmp_path: Path) -> None:
    _, _, _, context_path = _write_specialist_artifacts(tmp_path)
    store = FrameContextStore(context_path)

    assert store.get_text("f1") == "[CAPTION]\nA person runs."


def _caption_row(**updates: object) -> dict[str, object]:
    """Return one raw typed-caption artifact row for validation tests."""

    row: dict[str, object] = {
        "frame_id": "f1",
        "video_id": "v1",
        "frame_idx": 10,
        "timestamp_ms": 1_000,
        "text": "caption",
        "artifact_version": "caption-v1",
        "model_name": "caption-model",
    }
    row.update(updates)
    return row


def test_typed_store_rejects_whitespace_normalized_duplicate_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate-normalized-caption.parquet"
    pd.DataFrame(
        [_caption_row(), _caption_row(frame_id=" f1")]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="frame_id.*canonical representation"):
        CaptionStore(path)


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        ({"frame_id": " f1 "}, "frame_id"),
        ({"video_id": " v1 "}, "video_id"),
        ({"frame_idx": 10.0}, "frame_idx"),
        ({"frame_idx": True}, "frame_idx"),
        ({"frame_idx": "10"}, "frame_idx"),
        ({"timestamp_ms": 1_000.0}, "timestamp_ms"),
        ({"timestamp_ms": "1000"}, "timestamp_ms"),
    ],
)
def test_typed_store_rejects_coercible_raw_identity(
    tmp_path: Path,
    updates: dict[str, object],
    field: str,
) -> None:
    path = tmp_path / f"coercible-{field}.parquet"
    pd.DataFrame([_caption_row(**updates)]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match=field):
        CaptionStore(path)


def test_object_store_rejects_coercible_frame_identity(tmp_path: Path) -> None:
    object_dir = tmp_path / "coercible-objects"
    object_dir.mkdir()
    path = object_dir / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10.0,
                "timestamp_ms": 1_000,
                "counts_json": "{}",
                "summary": None,
                "detection_count": 0,
                "artifact_version": "object-v1",
                "status": "completed",
            }
        ]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="frame_idx"):
        ObjectStore(path)


def test_corpus_rejects_timestamp_mismatch_with_canonical_frame(
    tmp_path: Path,
) -> None:
    """Reject evidence whose timestamp disagrees with its canonical frame."""

    frames_path = _write_frames(tmp_path)
    caption_path = tmp_path / "timestamp-mismatch.parquet"
    pd.DataFrame(
        [
            _caption_row(timestamp_ms=999),
        ]
    ).to_parquet(caption_path, index=False)

    with pytest.raises(ValueError, match="canonical identity"):
        Corpus.open(
            frames_path,
            {RetrievalSource.CAPTION: caption_path},
        )


@pytest.mark.parametrize(
    ("field", "second_value"),
    [
        ("artifact_version", "caption-v2"),
        ("frame_store_id", "btc-v2"),
    ],
)
def test_typed_store_rejects_mixed_specialist_lineage(
    tmp_path: Path, field: str, second_value: str
) -> None:
    """Require one specialist artifact to have uniform version and lineage."""

    first = _caption_row(frame_store_id="btc-v1")
    second = _caption_row(
        frame_id="f2",
        frame_idx=20,
        timestamp_ms=2_000,
        frame_store_id="btc-v1",
    )
    second[field] = second_value
    path = tmp_path / "mixed-caption.parquet"
    pd.DataFrame([first, second]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="uniform.*(version|lineage)"):
        CaptionStore(path)


def test_typed_store_rejects_adjacent_manifest_identity_mismatch(
    tmp_path: Path,
) -> None:
    """Reject a valid row table coupled to a manifest for another artifact."""

    root = tmp_path / "caption"
    root.mkdir()
    path = root / "captions.parquet"
    pd.DataFrame(
        [_caption_row(frame_store_id="btc-v1")]
    ).to_parquet(path, index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": "caption-v2",
                "frame_store_id": "btc-v1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest.*artifact_version"):
        CaptionStore(path)


def test_corpus_compares_specialist_lineage_to_canonical_manifest(
    tmp_path: Path,
) -> None:
    """Tie typed specialist evidence to the loaded canonical frame store."""

    frame_root = tmp_path / "frame-store"
    frame_root.mkdir()
    frames_path = _write_frames(frame_root)
    (frame_root / "manifest.json").write_text(
        json.dumps({"frame_store_id": "btc-v1"}), encoding="utf-8"
    )

    caption_root = tmp_path / "caption"
    caption_root.mkdir()
    caption_path = caption_root / "captions.parquet"
    pd.DataFrame(
        [_caption_row(frame_store_id="btc-v2")]
    ).to_parquet(caption_path, index=False)
    (caption_root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_version": "caption-v1",
                "frame_store_id": "btc-v2",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frame_store_id.*canonical"):
        Corpus.open(
            frames_path,
            {RetrievalSource.CAPTION: caption_path},
        )


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        ({"frame_id": " f1"}, "frame_id"),
        ({"video_id": " v1"}, "video_id"),
        ({"frame_idx": 10.0}, "frame_idx"),
        ({"timestamp_ms": 1_000.0}, "timestamp_ms"),
        ({"detection_index": 0.0}, "detection_index"),
        ({"detection_index": True}, "detection_index"),
        ({"detection_index": "0"}, "detection_index"),
    ],
)
def test_object_store_rejects_coercible_detection_identity(
    tmp_path: Path,
    updates: dict[str, object],
    field: str,
) -> None:
    object_dir = tmp_path / f"coercible-detection-{field}"
    object_dir.mkdir()
    frame_path = object_dir / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "timestamp_ms": 1_000,
                "counts_json": json.dumps({"person": 1}),
                "summary": "person x1",
                "detection_count": 1,
                "artifact_version": "object-v1",
                "status": "completed",
            }
        ]
    ).to_parquet(frame_path, index=False)
    detection: dict[str, object] = {
        "frame_id": "f1",
        "video_id": "v1",
        "frame_idx": 10,
        "timestamp_ms": 1_000,
        "detection_index": 0,
        "label": "person",
        "confidence": 0.9,
        "x_min": 0.1,
        "y_min": 0.2,
        "x_max": 0.3,
        "y_max": 0.4,
    }
    detection.update(updates)
    pd.DataFrame([detection]).to_parquet(
        object_dir / "detections.parquet", index=False
    )

    with pytest.raises(ValueError, match=field):
        ObjectStore(frame_path)


@pytest.mark.parametrize("indices", [(-1, 1), (0, 2), (1, 0)])
def test_object_store_requires_contiguous_detection_indices(
    tmp_path: Path,
    indices: tuple[int, int],
) -> None:
    _, _, object_path, _ = _write_specialist_artifacts(tmp_path)
    detections_path = object_path.with_name("detections.parquet")
    table = pd.read_parquet(detections_path)
    table["detection_index"] = list(indices)
    table.to_parquet(detections_path, index=False)

    with pytest.raises(ValueError, match="contiguous.*detection_index"):
        ObjectStore(object_path)


def test_object_store_rejects_detection_frame_identity_mismatch(
    tmp_path: Path,
) -> None:
    """Reject flat detections aligned to a different canonical frame timestamp."""

    _, _, object_path, _ = _write_specialist_artifacts(tmp_path)
    detections_path = object_path.with_name("detections.parquet")
    table = pd.read_parquet(detections_path)
    table.loc[0, "timestamp_ms"] = 999
    table.to_parquet(detections_path, index=False)

    with pytest.raises(ValueError, match="canonical identity mismatch"):
        ObjectStore(object_path)


def test_object_writer_rejects_detection_frame_identity_mismatch(
    tmp_path: Path,
) -> None:
    """Reject inconsistent flat identities before publishing object artifacts."""

    row = ObjectEvidence(
        frame_id="f1",
        video_id="v1",
        frame_idx=10,
        timestamp_ms=1_000,
        detections=[
            ObjectDetection(
                label="person",
                confidence=0.9,
                x_min=0.1,
                y_min=0.2,
                x_max=0.3,
                y_max=0.4,
            )
        ],
        counts={"person": 1},
        summary="person x1",
        detection_count=1,
        artifact_version="object-v1",
    )

    with pytest.raises(ValueError, match="canonical identity"):
        write_object_artifacts(
            tmp_path / "objects",
            ["f1"],
            [row],
            [
                {
                    "frame_id": "f1",
                    "video_id": "v1",
                    "frame_idx": 10,
                    "timestamp_ms": 999,
                    "detection_index": 0,
                    **row.detections[0].model_dump(mode="json"),
                }
            ],
            {"artifact_version": "object-v1"},
        )


def test_object_store_requires_serialized_counts_json(tmp_path: Path) -> None:
    object_dir = tmp_path / "object-count-shape"
    object_dir.mkdir()
    path = object_dir / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "timestamp_ms": 1_000,
                "counts_json": {"person": 0},
                "summary": None,
                "detection_count": 0,
                "artifact_version": "object-v1",
                "status": "completed",
            }
        ]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="counts_json.*JSON string"):
        ObjectStore(path)


def test_object_store_loads_real_producer_bundle(tmp_path: Path) -> None:
    detection = ObjectDetection(
        label="person",
        confidence=0.9,
        x_min=0.1,
        y_min=0.2,
        x_max=0.3,
        y_max=0.4,
    )
    rows = [
        ObjectEvidence(
            frame_id="f1",
            video_id="v1",
            frame_idx=10,
            timestamp_ms=1_000,
            detections=[detection],
            counts={"person": 1},
            summary="person x1",
            detection_count=1,
            artifact_version="object-v1",
        ),
        ObjectEvidence(
            frame_id="f2",
            video_id="v1",
            frame_idx=20,
            timestamp_ms=2_000,
            artifact_version="object-v1",
        ),
        ObjectEvidence(
            frame_id="f3",
            video_id="v1",
            frame_idx=30,
            timestamp_ms=3_000,
            artifact_version="object-v1",
            status=ProcessingStatus.FAILED,
            error_code="MissingSource",
            error_message="source object file is missing",
        ),
    ]
    output = tmp_path / "producer-objects"
    write_object_artifacts(
        output,
        ["f1", "f2", "f3"],
        rows,
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "timestamp_ms": 1_000,
                "detection_index": 0,
                **detection.model_dump(mode="json"),
            }
        ],
        {"artifact_version": "object-v1"},
    )

    store = ObjectStore(output / "frames.parquet")

    assert [
        item.model_dump(mode="json") for item in store.get("f1").detections
    ] == [detection.model_dump(mode="json")]
    assert store.get("f2").detection_count == 0
    assert store.get("f3").status == ProcessingStatus.FAILED
