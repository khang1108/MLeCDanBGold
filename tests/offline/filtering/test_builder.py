"""Tests for deterministic offline Filter catalog publication."""

from __future__ import annotations

import json
import sqlite3

from pathlib import Path

import pandas as pd
import pytest

from hcmai.filtering.catalog import FilterCatalog
from offline.enrichment.caption.models import CaptionEvidence
from offline.enrichment.ocr.models import OCREvidence
from offline.enrichment.transcripts.models import TranscriptSegment
from offline.filtering.builder import (
    FilterCatalogBuildConfig,
    FilterCatalogBuildError,
    build_filter_catalog,
)


def _write_inputs(root: Path) -> dict[str, Path]:
    """Write six canonical frames with boundary-focused specialist evidence."""

    frames = [
        {
            "frame_id": f"{video_id}_{index:06d}",
            "video_id": video_id,
            "frame_idx": index * 25,
            "timestamp_ms": timestamp_ms,
            "image_path": f"keyframes/{video_id}/{index}.jpg",
        }
        for video_id, index, timestamp_ms in (
            ("L21_V001", 1, 0),
            ("L21_V001", 2, 999),
            ("L21_V001", 3, 1000),
            ("L21_V001", 4, 1500),
            ("L22_V002", 1, 500),
            ("L22_V002", 2, 2500),
        )
    ]
    frames_path = root / "frames.parquet"
    pd.DataFrame(frames).to_parquet(frames_path, index=False)

    captions_path = root / "captions.parquet"
    pd.DataFrame(
        [
            CaptionEvidence(
                frame_id=row["frame_id"],
                video_id=row["video_id"],
                frame_idx=row["frame_idx"],
                timestamp_ms=row["timestamp_ms"],
                text="Cảnh có ÁO đỏ" if index == 0 else f"caption {index}",
                artifact_version="caption-v1",
                model_name="fixture-caption",
            ).model_dump(mode="json")
            for index, row in enumerate(frames[:-1])
        ]
    ).to_parquet(captions_path, index=False)

    ocr_path = root / "ocr.parquet"
    pd.DataFrame(
        [
            OCREvidence(
                frame_id=row["frame_id"],
                video_id=row["video_id"],
                frame_idx=row["frame_idx"],
                timestamp_ms=row["timestamp_ms"],
                raw_text="ĐƯỜNG phố",
                normalized_text="ĐƯỜNG phố",
                artifact_version="ocr-v1",
                model_name="fixture-ocr",
            ).model_dump(mode="json")
            for row in frames[:2]
        ]
    ).to_parquet(ocr_path, index=False)

    objects_path = root / "objects.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": frames[2]["frame_id"],
                "video_id": frames[2]["video_id"],
                "frame_idx": frames[2]["frame_idx"],
                "timestamp_ms": frames[2]["timestamp_ms"],
                "counts_json": json.dumps({"person": 3, "car": 1}),
                "status": "completed",
            },
            {
                "frame_id": frames[3]["frame_id"],
                "video_id": frames[3]["video_id"],
                "frame_idx": frames[3]["frame_idx"],
                "timestamp_ms": frames[3]["timestamp_ms"],
                "counts_json": json.dumps({"person": 1}),
                "status": "completed",
            },
        ]
    ).to_parquet(objects_path, index=False)

    transcripts_path = root / "transcripts.parquet"
    pd.DataFrame(
        [
            TranscriptSegment(
                segment_id="first",
                video_id="L21_V001",
                segment_index=0,
                start_ms=0,
                end_ms=1000,
                text="Xin chào",
                language="vi",
            ).model_dump(mode="json"),
            TranscriptSegment(
                segment_id="second",
                video_id="L21_V001",
                segment_index=1,
                start_ms=1000,
                end_ms=2000,
                text="Thế giới",
                language="vi",
            ).model_dump(mode="json"),
        ]
    ).to_parquet(transcripts_path, index=False)

    metadata_path = root / "media-info"
    metadata_path.mkdir()
    (metadata_path / "L21_V001.json").write_text(
        json.dumps({"title": "Tập ĐẦU"}), encoding="utf-8"
    )
    (metadata_path / "L22_V002.json").write_text(
        json.dumps({"title": "Episode two"}), encoding="utf-8"
    )

    return {
        "frames": frames_path,
        "caption": captions_path,
        "ocr": ocr_path,
        "objects": objects_path,
        "transcripts": transcripts_path,
        "metadata": metadata_path,
    }


def _config(root: Path, paths: dict[str, Path]) -> FilterCatalogBuildConfig:
    """Create a complete build config using the shared fixture artifacts."""

    return FilterCatalogBuildConfig(
        frames_path=paths["frames"],
        video_metadata_path=paths["metadata"],
        caption_path=paths["caption"],
        ocr_path=paths["ocr"],
        object_counts_path=paths["objects"],
        transcripts_path=paths["transcripts"],
        output_path=root / "filter_catalog.sqlite",
        catalog_version="fixture-catalog-v1",
        source_lineage={"frame_store_id": "fixture-frames-v1"},
        batch_size=2,
    )


def test_builder_preserves_identity_counts_normalization_and_asr_boundaries(
    tmp_path: Path,
) -> None:
    """Materialize exact frame-local evidence without rewriting coordinates."""

    paths = _write_inputs(tmp_path)
    config = _config(tmp_path, paths)

    report = build_filter_catalog(config)

    assert report.frame_count == 6
    assert report.output_path == config.output_path
    catalog = FilterCatalog.open(config.output_path, pool_size=1)
    assert catalog.info.availability.title is True
    assert catalog.info.availability.asr is True
    with catalog.connection() as connection:
        boundary_rows = connection.execute(
            """
            SELECT frame_id, video_id, frame_idx, timestamp_ms, folder_id,
                   title_norm, caption_norm, ocr_norm, asr, asr_norm
            FROM frames WHERE video_id = 'L21_V001'
            ORDER BY timestamp_ms
            """
        ).fetchall()
        assert tuple(boundary_rows[1][0:5]) == (
            "L21_V001_000002", "L21_V001", 50, 999, "L21"
        )
        assert boundary_rows[0]["title_norm"] == "tap dau"
        assert boundary_rows[0]["caption_norm"] == "canh co ao do"
        assert boundary_rows[0]["ocr_norm"] == "duong pho"
        assert boundary_rows[1]["asr_norm"] == "xin chao"
        assert boundary_rows[2]["asr_norm"] == "the gioi"
        counts = connection.execute(
            """
            SELECT label_norm, object_count FROM frame_objects
            WHERE frame_id = 'L21_V001_000003' ORDER BY label_norm
            """
        ).fetchall()
        assert [tuple(row) for row in counts] == [("car", 1), ("person", 3)]
    catalog.close()


def test_builder_records_globally_unavailable_modalities(tmp_path: Path) -> None:
    """Distinguish a missing artifact from per-frame missing evidence."""

    paths = _write_inputs(tmp_path)
    config = _config(tmp_path, paths)
    config = FilterCatalogBuildConfig(
        frames_path=config.frames_path,
        output_path=config.output_path,
        catalog_version=config.catalog_version,
    )

    build_filter_catalog(config)

    catalog = FilterCatalog.open(config.output_path, pool_size=1)
    assert catalog.info.availability.title is False
    assert catalog.info.availability.caption is False
    assert catalog.info.availability.ocr is False
    assert catalog.info.availability.objects is False
    assert catalog.info.availability.asr is False
    with catalog.connection() as connection:
        row = connection.execute(
            "SELECT title, caption, ocr, asr FROM frames LIMIT 1"
        ).fetchone()
        assert tuple(row) == (None, None, None, None)
    catalog.close()


def test_failed_identity_validation_keeps_previous_catalog(tmp_path: Path) -> None:
    """Never replace the published catalog with a partial invalid build."""

    paths = _write_inputs(tmp_path)
    caption_table = pd.read_parquet(paths["caption"])
    bad_row = dict(caption_table.iloc[0])
    bad_row["frame_id"] = "L21_V001_unknown"
    pd.concat([caption_table, pd.DataFrame([bad_row])], ignore_index=True).to_parquet(
        paths["caption"], index=False
    )
    config = _config(tmp_path, paths)
    config.output_path.write_bytes(b"previous-catalog")

    with pytest.raises(FilterCatalogBuildError, match="unknown frame_id"):
        build_filter_catalog(config)

    assert config.output_path.read_bytes() == b"previous-catalog"
    assert list(tmp_path.glob(".filter_catalog.sqlite.*.tmp")) == []


def test_builder_rejects_coordinate_mismatch(tmp_path: Path) -> None:
    """Reject evidence that reuses an ID with different submission identity."""

    paths = _write_inputs(tmp_path)
    table = pd.read_parquet(paths["objects"])
    table.loc[0, "frame_idx"] = 999
    table.to_parquet(paths["objects"], index=False)

    with pytest.raises(FilterCatalogBuildError, match="canonical identity"):
        build_filter_catalog(_config(tmp_path, paths))

