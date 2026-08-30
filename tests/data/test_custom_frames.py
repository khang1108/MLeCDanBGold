"""Tests for validation and Parquet materialization of native custom frames."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from offline.ingestion.custom_frames import CustomFrameStoreConfig
from offline.ingestion.custom_frames import (
    iter_native_frame_records,
    materialize_custom_frame_store,
    validate_native_video_bundle,
)
from hcmai.corpus.stores.frame import FrameStore


def _write_json(path: Path, value: object) -> None:
    """Write one compact JSON fixture document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _frame_row(
    video_id: str,
    sample_index: int,
    *,
    fps: float,
    timestamp_ms: int,
    fps_numerator: int = 30_000,
    fps_denominator: int = 1_001,
) -> dict[str, object]:
    """Build one native JSONL row with files populated by the bundle writer."""

    filename = f"{sample_index:09d}.jpg"
    return {
        "frame_id": f"{video_id}_raw1fps_{sample_index:09d}",
        "video_id": video_id,
        "sample_index": sample_index,
        "target_timestamp_ms": sample_index * 1_000,
        "timestamp_ms": timestamp_ms,
        "frame_idx": math.floor(math.ceil(fps) * timestamp_ms / 1_000),
        "avg_fps": fps,
        "avg_fps_num": fps_numerator,
        "avg_fps_den": fps_denominator,
        "pts": timestamp_ms,
        "time_base_num": 1,
        "time_base_den": 1_000,
        "width": 80,
        "height": 40,
        "image_path": f"images/{filename}",
        "enrichment_image_path": f"enrichment_images/{filename}",
        "image_size_bytes": 0,
        "enrichment_image_size_bytes": 0,
    }


def write_valid_native_bundle(
    run_root: Path,
    video_id: str = "L01_V001",
    *,
    fps: float = 29.97,
    count: int = 2,
    status: str = "published",
    timestamps_ms: list[int] | None = None,
) -> Path:
    """Write a complete small native bundle with consistent image byte sizes."""

    directory_name = "published" if status == "published" else "staging"
    bundle = run_root / directory_name / video_id
    timestamps = timestamps_ms or [index * 1_000 for index in range(count)]
    if len(timestamps) != count:
        raise ValueError("fixture timestamp count must match native row count")
    rows = [
        _frame_row(video_id, index, fps=fps, timestamp_ms=timestamps[index])
        for index in range(count)
    ]
    for row in rows:
        durable = bundle / str(row["image_path"])
        enrichment = bundle / str(row["enrichment_image_path"])
        durable.parent.mkdir(parents=True, exist_ok=True)
        enrichment.parent.mkdir(parents=True, exist_ok=True)
        durable.write_bytes(b"durable-image")
        enrichment.write_bytes(b"enrichment-image")
        row["image_size_bytes"] = durable.stat().st_size
        row["enrichment_image_size_bytes"] = enrichment.stat().st_size
    (bundle / "frames.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(
        bundle / "manifest.json",
        {
            "video_id": video_id,
            "status": status,
            "duration_ms": count * 1_000,
            "expected_frame_count": count,
            "emitted_frame_count": count,
            "avg_fps": fps,
            "avg_fps_num": 30_000,
            "avg_fps_den": 1_001,
            "extractor_version": "hcmai-keyframes-extractor/0.1.0",
            "config_hash": "fixture-config-hash",
            "frames_jsonl": "frames.jsonl",
        },
    )
    return bundle


def test_native_rows_map_to_frame_records_without_keyframe_order(
    tmp_path: Path,
) -> None:
    """Preserve custom identity and actual timestamps without BTC keyframe order."""

    bundle = write_valid_native_bundle(tmp_path, fps=29.97, count=2)

    report = validate_native_video_bundle(
        bundle,
        run_root=tmp_path,
        expected_status="published",
    )
    records = list(iter_native_frame_records(bundle, run_root=tmp_path))

    assert report.frame_count == 2
    assert report.duplicate_submission_coordinate_groups == 0
    assert records[0].frame_id == "L01_V001_raw1fps_000000000"
    assert records[0].keyframe_order is None
    assert records[1].frame_idx == math.floor(
        math.ceil(29.97) * records[1].timestamp_ms / 1_000
    )
    assert records[1].image_path == "published/L01_V001/images/000000001.jpg"


def test_native_validation_rejects_formula_mismatch_but_allows_coordinate_collision(
    tmp_path: Path,
) -> None:
    """Reject changed competition coordinates while retaining valid internal frames."""

    bundle = write_valid_native_bundle(
        tmp_path,
        fps=1.1,
        count=2,
        timestamps_ms=[0, 1],
    )
    collision_report = validate_native_video_bundle(bundle, run_root=tmp_path)
    assert collision_report.duplicate_submission_coordinate_groups == 1

    rows = [
        json.loads(line)
        for line in (bundle / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["frame_idx"] = 99
    (bundle / "frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frame_idx formula"):
        validate_native_video_bundle(bundle, run_root=tmp_path)


def test_native_validation_rejects_missing_or_escaping_images(tmp_path: Path) -> None:
    """Treat image paths and byte-size coverage as a correctness boundary."""

    bundle = write_valid_native_bundle(tmp_path, count=1)
    image = bundle / "images" / "000000000.jpg"
    image.unlink()
    with pytest.raises(ValueError, match="regular file"):
        validate_native_video_bundle(bundle, run_root=tmp_path)

    bundle = write_valid_native_bundle(tmp_path / "escaped", count=1)
    rows = [
        json.loads(line)
        for line in (bundle / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["image_path"] = "../../outside.jpg"
    (bundle / "frames.jsonl").write_text(
        json.dumps(rows[0]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be relative"):
        validate_native_video_bundle(bundle, run_root=tmp_path / "escaped")


def test_materialize_custom_frame_store_publishes_validated_bundle(
    tmp_path: Path,
) -> None:
    """Create one atomic FrameStore without replacing custom identity or images."""

    write_valid_native_bundle(tmp_path, "L01_V001", count=2)
    write_valid_native_bundle(tmp_path, "L01_V002", count=1)

    output = materialize_custom_frame_store(
        CustomFrameStoreConfig(
            run_root=tmp_path,
            output_root=tmp_path / "corpus",
            frame_store_id="custom-raw1fps-v1",
            selected_video_ids=("L01_V002", "L01_V001"),
        )
    )

    table = pd.read_parquet(output)
    assert table["frame_id"].tolist() == [
        "L01_V001_raw1fps_000000000",
        "L01_V001_raw1fps_000000001",
        "L01_V002_raw1fps_000000000",
    ]
    assert table["keyframe_order"].isna().all()
    assert len(FrameStore(output)) == 3
    manifest = json.loads((tmp_path / "corpus" / "manifest.json").read_text())
    assert manifest["source"] == "custom_raw_video_1fps"
    assert manifest["frame_count"] == 3
    assert manifest["video_count"] == 2


def test_materialization_refuses_one_missing_published_video(tmp_path: Path) -> None:
    """Fail the whole corpus publication rather than silently dropping a video."""

    write_valid_native_bundle(tmp_path, "L01_V001", count=1)

    with pytest.raises(ValueError, match="missing validated published bundle"):
        materialize_custom_frame_store(
            CustomFrameStoreConfig(
                run_root=tmp_path,
                output_root=tmp_path / "corpus",
                frame_store_id="custom-raw1fps-v1",
                selected_video_ids=("L01_V001", "L01_V002"),
            )
        )
