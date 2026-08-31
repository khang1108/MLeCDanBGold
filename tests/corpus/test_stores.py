"""Hand-checkable coverage for corpus-owned runtime artifact readers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hcmai.corpus.assets import (
    FrameAssetMissingError,
    FrameAssetOutsideRootError,
    FrameAssetResolver,
)
from hcmai.corpus.models import Frame, TranscriptSegment, VideoMetadata
from hcmai.corpus.stores import FrameStore, TranscriptStore, VideoMetadataStore
from offline.enrichment.transcripts.models import (
    TranscriptSegment as TranscriptSegmentArtifact,
)


def _write_frame_artifact(path: Path) -> None:
    """Write one artifact row with fields beyond the runtime frame view."""

    pd.DataFrame(
        [
            {
                "frame_id": "frame-001",
                "video_id": "video-001",
                "frame_idx": 42,
                "timestamp_ms": 1_250,
                "image_path": "keyframes/video-001/42.jpg",
                "thumbnail_path": None,
                "width": 1_920,
                "height": 1_080,
            }
        ]
    ).to_parquet(path, index=False)


def test_corpus_stores_materialize_runtime_dataclasses(tmp_path: Path) -> None:
    """Validate full artifact rows while exposing only runtime model fields."""

    frames_path = tmp_path / "frames.parquet"
    _write_frame_artifact(frames_path)

    transcript_path = tmp_path / "transcripts.parquet"
    pd.DataFrame(
        [
            TranscriptSegmentArtifact(
                segment_id="video-001-segment-000",
                video_id="video-001",
                segment_index=0,
                start_ms=1_000,
                end_ms=1_500,
                text="A brief utterance.",
                language="en",
            ).model_dump(mode="json")
        ]
    ).to_parquet(transcript_path, index=False)

    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    (metadata_root / "video-001.json").write_text(
        json.dumps({"title": "Example", "watch_url": "https://example.test/v"}),
        encoding="utf-8",
    )

    frame = FrameStore(frames_path).get("frame-001")
    segment = TranscriptStore(transcript_path).get("video-001-segment-000")
    metadata = VideoMetadataStore(metadata_root).get("video-001")

    assert frame == Frame(
        frame_id="frame-001",
        video_id="video-001",
        frame_idx=42,
        timestamp_ms=1_250,
        image_path="keyframes/video-001/42.jpg",
    )
    assert segment == TranscriptSegment(
        segment_id="video-001-segment-000",
        video_id="video-001",
        segment_index=0,
        start_ms=1_000,
        end_ms=1_500,
        text="A brief utterance.",
    )
    assert metadata == VideoMetadata(
        video_id="video-001",
        title="Example",
        video_url="https://example.test/v",
    )


def test_asset_resolver_preserves_rebasing_fallback_and_health(tmp_path: Path) -> None:
    """Keep root containment and portable keyframe resolution deterministic."""

    dataset_root = tmp_path / "dataset"
    image = dataset_root / "keyframes" / "video-001" / "42.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    resolver = FrameAssetResolver(dataset_root)
    frame = Frame(
        frame_id="frame-001",
        video_id="video-001",
        frame_idx=42,
        timestamp_ms=1_250,
        image_path="keyframes/video-001/42.jpg",
    )
    missing = Frame(
        frame_id="frame-002",
        video_id="video-001",
        frame_idx=43,
        timestamp_ms=1_500,
        image_path="keyframes/video-001/missing.jpg",
    )

    assert resolver.resolve_frame(frame) == image
    assert resolver.resolve_frame(frame, thumbnail=True) == image
    assert resolver.resolve_value("/legacy/data/keyframes/video-001/42.jpg") == image
    assert resolver.sample_status((frame, missing)).as_dict() == {
        "ready": False,
        "checked": 2,
        "available": 1,
        "missing": 1,
    }
    with pytest.raises(FrameAssetMissingError):
        resolver.resolve_frame(missing)
    with pytest.raises(FrameAssetOutsideRootError):
        resolver.resolve_value("../outside.jpg", require_file=False)
