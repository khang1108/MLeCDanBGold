from pathlib import Path
from dataclasses import asdict
from typing import cast

import pandas as pd
import pytest

from hcmai.corpus import Frame
from hcmai.corpus.assets import (
    FrameAssetMissingError,
    FrameAssetOutsideRootError,
    FrameAssetResolver,
)
from hcmai.corpus import Corpus
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService


def frame(frame_id: str, image_path: str) -> Frame:
    return Frame(
        frame_id=frame_id,
        video_id="video-1",
        frame_idx=1,
        timestamp_ms=0,
        image_path=image_path,
    )


def test_resolver_handles_relative_thumbnail_missing_and_escape(tmp_path: Path) -> None:
    image = tmp_path / "keyframes" / "video-1" / "1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    resolver = FrameAssetResolver(tmp_path)
    record = frame("f1", "keyframes/video-1/1.jpg")

    assert resolver.resolve_frame(record) == image
    assert resolver.resolve_frame(record, thumbnail=True) == image
    with pytest.raises(FrameAssetMissingError):
        resolver.resolve_value("keyframes/video-1/missing.jpg")
    with pytest.raises(FrameAssetOutsideRootError):
        resolver.resolve_value("../outside.jpg", require_file=False)


def test_resolver_rebases_legacy_absolute_keyframe_path(
    tmp_path: Path,
) -> None:
    """Serve portable artifacts created under a different machine root."""

    dataset_root = tmp_path / "data"
    image = dataset_root / "keyframes" / "video-1" / "1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    legacy = Path("/old/worker/data/keyframes/video-1/1.jpg")

    resolver = FrameAssetResolver(dataset_root)

    assert resolver.resolve_value(legacy) == image
    with pytest.raises(FrameAssetOutsideRootError):
        resolver.resolve_value("/old/worker/other/1.jpg", require_file=False)


def test_corpus_resolves_canonical_assets(tmp_path: Path) -> None:
    image = tmp_path / "keyframes" / "video-1" / "1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    rows = [
        asdict(frame("f1", "keyframes/video-1/1.jpg")),
        asdict(frame("f2", "keyframes/video-1/missing.jpg")),
    ]
    metadata = tmp_path / "frames.parquet"
    pd.DataFrame(rows).to_parquet(metadata, index=False)

    corpus = Corpus.open(metadata, dataset_root=tmp_path)

    assert corpus.image_path("f1") == image
    with pytest.raises(FrameAssetMissingError):
        corpus.image_path("f2")


def test_health_reports_asset_readiness_separately_from_metadata(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "frames.parquet"
    pd.DataFrame([
        asdict(frame("f1", "keyframes/video-1/missing.jpg"))
    ]).to_parquet(metadata, index=False)
    corpus = Corpus.open(metadata, dataset_root=tmp_path)
    service = SearchService(corpus, cast(RetrievalService, object()))

    capabilities = service.health()["capabilities"]

    assert capabilities["search"] is True
    assert capabilities["frame_assets"] is False
    assert capabilities["frame_asset_status"] == {
        "ready": False,
        "checked": 1,
        "available": 0,
        "missing": 1,
    }
