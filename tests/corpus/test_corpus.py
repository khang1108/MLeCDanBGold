"""Focused behavior tests for the public read-only :class:`Corpus` facade."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from hcmai.corpus import Corpus
from hcmai.retrieval.models import RetrievalSource
from offline.enrichment.caption.models import CaptionEvidence
from offline.enrichment.models import ProcessingStatus
from offline.enrichment.ocr.models import OCREvidence
from offline.enrichment.transcripts.models import TranscriptSegment


def _write_corpus_artifacts(root: Path) -> dict[str, Path]:
    """Create hand-checkable runtime artifact fixtures without production data."""

    frames_path = root / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "timestamp_ms": 1_000,
                "image_path": "keyframes/v1/10.jpg",
                "thumbnail_path": "keyframes/v1/thumb-10.jpg",
                "width": 10,
                "height": 10,
            }
        ]
    ).to_parquet(frames_path, index=False)

    caption_path = root / "captions.parquet"
    pd.DataFrame(
        [
            CaptionEvidence(
                frame_id="f1",
                video_id="v1",
                frame_idx=10,
                timestamp_ms=1_000,
                text="caption",
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
                raw_text="OCR",
                normalized_text="OCR",
                artifact_version="ocr-v1",
                model_name="ocr-model",
            ).model_dump(mode="json")
        ]
    ).to_parquet(ocr_path, index=False)

    object_counts_path = root / "object-counts.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "timestamp_ms": 1_000,
                "counts_json": json.dumps({"person": 2, "bowl": 1}),
                "status": ProcessingStatus.COMPLETED.value,
            }
        ]
    ).to_parquet(object_counts_path, index=False)

    transcript_path = root / "transcripts.parquet"
    pd.DataFrame(
        [
            TranscriptSegment(
                segment_id="later",
                video_id="v1",
                segment_index=0,
                start_ms=1_900,
                end_ms=2_100,
                text=" second ",
                language="en",
            ).model_dump(mode="json"),
            TranscriptSegment(
                segment_id="first",
                video_id="v1",
                segment_index=1,
                start_ms=1_000,
                end_ms=1_500,
                text="first",
                language="en",
            ).model_dump(mode="json"),
        ]
    ).to_parquet(transcript_path, index=False)

    metadata_path = root / "metadata"
    metadata_path.mkdir()
    (metadata_path / "v1.json").write_text(
        json.dumps({"title": "Episode 1"}), encoding="utf-8"
    )

    image_path = root / "keyframes" / "v1" / "10.jpg"
    thumbnail_path = root / "keyframes" / "v1" / "thumb-10.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    thumbnail_path.write_bytes(b"thumbnail")

    return {
        "frames": frames_path,
        "caption": caption_path,
        "ocr": ocr_path,
        "object_counts": object_counts_path,
        "transcript": transcript_path,
        "metadata": metadata_path,
        "image": image_path,
        "thumbnail": thumbnail_path,
    }


def _open_fixture_corpus(tmp_path: Path) -> tuple[Corpus, dict[str, Path]]:
    """Open the complete small fixture corpus used by projection tests."""

    paths = _write_corpus_artifacts(tmp_path)
    return (
        Corpus.open(
            paths["frames"],
            {
                RetrievalSource.CAPTION: paths["caption"],
                RetrievalSource.OCR: paths["ocr"],
            },
            dataset_root=tmp_path,
            object_counts_path=paths["object_counts"],
            transcript_path=paths["transcript"],
            video_metadata_path=paths["metadata"],
        ),
        paths,
    )


def test_open_requires_existing_frame_artifact(tmp_path: Path) -> None:
    """Corpus opens no generators and fails immediately for missing frames."""

    with pytest.raises(FileNotFoundError):
        Corpus.open(frames_path=tmp_path / "missing.parquet", dataset_root=tmp_path)


def test_public_open_signature_excludes_raw_object_and_context_paths() -> None:
    """Freeze the deliberately small read-only public construction contract."""

    assert list(inspect.signature(Corpus.open).parameters) == [
        "frames_path",
        "evidence_paths",
        "dataset_root",
        "object_counts_path",
        "transcript_path",
        "video_metadata_path",
    ]


def test_corpus_projects_read_only_evidence_and_paths(tmp_path: Path) -> None:
    """Expose stable public views without leaking specialist store records."""

    corpus, paths = _open_fixture_corpus(tmp_path)

    assert corpus.frame("f1").frame_idx == 10
    assert corpus.frames(["f1", "f1"]) == [corpus.frame("f1")] * 2
    assert corpus.caption("f1") == "caption"
    assert corpus.ocr("f1") == "OCR"
    assert corpus.objects("f1") == ("bowl", "person")
    assert corpus.title("v1") == "Episode 1"
    assert corpus.image_path("f1") == paths["image"]
    assert corpus.thumbnail_path("f1") == paths["thumbnail"]


def test_transcript_uses_ordered_half_open_overlap_and_none_for_no_match(
    tmp_path: Path,
) -> None:
    """Keep timeline range aggregation separate from frame-native evidence."""

    corpus, _ = _open_fixture_corpus(tmp_path)

    assert [
        segment.segment_id
        for segment in corpus.transcript_segments("v1", 1_000, 2_000)
    ] == ["first", "later"]
    assert corpus.transcript("v1", 1_000, 2_000) == "first second"
    assert corpus.transcript("v1", 2_100, 2_200) is None
    assert corpus.transcript_segments("missing", 1_000, 2_000) == ()
    assert corpus.transcript("v1", 1_200, 1_200) is None


def test_asset_paths_require_a_configured_dataset_root(tmp_path: Path) -> None:
    """Prevent relative artifact paths from being resolved against process CWD."""

    paths = _write_corpus_artifacts(tmp_path)
    corpus = Corpus.open(paths["frames"])

    with pytest.raises(RuntimeError, match="dataset_root"):
        corpus.image_path("f1")
