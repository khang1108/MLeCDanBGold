"""Tests for AIC corpus inventory, ingestion, and validation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from hcmai.data import (
    ingest_dataset,
    inventory_corpus,
    prepare_dataset,
    validate_dataset,
)


@pytest.fixture
def aic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Create a two-video fixture with the official Kaggle folder layout."""

    dataset_root = tmp_path / "aic-dataset"
    output_root = tmp_path / "prepared"
    mapping_root = (
        dataset_root / "map-keyframes-aic25-b1" / "map-keyframes"
    )
    keyframe_root = dataset_root / "Keyframes_L21" / "keyframes"
    mapping_root.mkdir(parents=True)

    videos = {
        "L21_V001": {
            "n": [1, 3, 10],
            "pts_time": [0.04, 0.44, 1.20],
            "fps": [25.0, 25.0, 25.0],
            "frame_idx": [7, 81, 211],
            "filenames": ["001.jpg", "0003.jpg", "10.jpg"],
            "size": (96, 64),
            "color": (180, 20, 20),
        },
        "L21_V002": {
            "n": [1, 2, 9],
            "pts_time": [0.05, 0.55, 1.25],
            "fps": [30.0, 30.0, 30.0],
            "frame_idx": [4, 99, 305],
            "filenames": ["000001.jpg", "2.jpg", "009.jpg"],
            "size": (80, 120),
            "color": (20, 20, 180),
        },
    }
    for video_id, values in videos.items():
        pd.DataFrame(
            {
                column: values[column]
                for column in ("n", "pts_time", "fps", "frame_idx")
            }
        ).to_csv(mapping_root / f"{video_id}.csv", index=False)
        video_root = keyframe_root / video_id
        video_root.mkdir(parents=True)
        for filename in values["filenames"]:
            Image.new(
                "RGB",
                values["size"],
                values["color"],
            ).save(video_root / filename)

    return dataset_root, output_root


def test_inventory_reports_corpus_and_round_robin_fixture(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Inventory the full corpus while sampling both videos deterministically."""

    dataset_root, output_root = aic_dataset

    report = inventory_corpus(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
        limit=4,
    )

    assert report["counts"] == {
        "mapping_files": 2,
        "mapping_rows": 6,
        "canonical_mapping_rows": 6,
        "mapping_collisions": 0,
        "discarded_aliases": 0,
        "keyframe_images": 6,
        "media_info_files": 0,
        "video_files": 0,
        "corrupt_images": 0,
        "duplicate_images": 0,
        "duplicate_mappings": 0,
    }
    assert report["fps"] == {"25": 3, "30": 3}
    assert report["mapping_coverage"] == {
        "matched": 6,
        "missing_images": 0,
        "missing_mappings": 0,
        "ratio": 1.0,
    }
    assert [
        (sample["video_id"], sample["n"])
        for sample in report["audit_samples"]
    ] == [
        ("L21_V001", 1),
        ("L21_V002", 1),
        ("L21_V002", 2),
        ("L21_V001", 3),
    ]
    assert report["unavailable"] == {
        "duration": True,
        "vfr": True,
        "audio": True,
    }
    assert (output_root / "reports" / "corpus_inventory.md").is_file()
    saved = json.loads(
        (output_root / "reports" / "corpus_inventory.json").read_text()
    )
    assert saved["audit_samples"] == report["audit_samples"]


def test_ingest_uses_authoritative_mapping_and_creates_thumbnails(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Build stable records from CSV frame indexes and numeric image stems."""

    dataset_root, output_root = aic_dataset

    frames_path = ingest_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
        thumbnail_max_edge=32,
    )
    frames = pd.read_parquet(frames_path)

    assert frames["frame_id"].tolist() == [
        "L21_V001_00000007",
        "L21_V001_00000081",
        "L21_V001_00000211",
        "L21_V002_00000004",
        "L21_V002_00000099",
        "L21_V002_00000305",
    ]
    assert frames["frame_idx"].tolist() == [7, 81, 211, 4, 99, 305]
    assert frames["timestamp_ms"].tolist() == [40, 440, 1200, 50, 550, 1250]
    assert frames[["width", "height"]].values.tolist() == [
        [96, 64],
        [96, 64],
        [96, 64],
        [80, 120],
        [80, 120],
        [80, 120],
    ]
    assert Path(frames.iloc[1]["image_path"]).name == "0003.jpg"
    for thumbnail in frames["thumbnail_path"]:
        thumbnail_path = Path(thumbnail)
        assert thumbnail_path.is_absolute()
        with Image.open(thumbnail_path) as image:
            image.verify()
            assert max(image.size) == 32


def test_collision_keeps_smallest_n_and_preserves_source_images(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Canonicalize duplicate frame indexes and audit every discarded alias."""

    dataset_root, output_root = aic_dataset
    mapping_path = (
        dataset_root
        / "map-keyframes-aic25-b1"
        / "map-keyframes"
        / "L21_V001.csv"
    )
    mapping = pd.read_csv(mapping_path)
    mapping["frame_idx"] = [7, 7, 7]
    mapping.to_csv(mapping_path, index=False)

    inventory = inventory_corpus(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )
    frames_path = ingest_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )
    frames = pd.read_parquet(frames_path)
    collisions = pd.read_csv(
        output_root / "reports" / "mapping_collisions.csv"
    )

    assert inventory["counts"]["mapping_rows"] == 6
    assert inventory["counts"]["canonical_mapping_rows"] == 4
    assert inventory["counts"]["mapping_collisions"] == 1
    assert inventory["counts"]["discarded_aliases"] == 2
    assert frames.loc[frames["video_id"] == "L21_V001", "frame_idx"].tolist() == [7]
    kept_image = frames.loc[
        frames["video_id"] == "L21_V001",
        "image_path",
    ].item()
    assert Path(kept_image).stem == "001"
    assert collisions["canonical_n"].tolist() == [1, 1]
    assert collisions["discarded_n"].tolist() == [3, 10]
    assert set(collisions["policy"]) == {"keep_smallest_n"}
    for row in collisions.itertuples(index=False):
        discarded = Path(row.discarded_image_path)
        assert discarded.is_file()
        digest = hashlib.sha256(discarded.read_bytes()).hexdigest()
        assert digest == row.discarded_sha256

    validation = validate_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
        deep=True,
        metadata_path=frames_path,
    )
    assert validation["valid"] is True
    assert validation["row_count"] == 4
    assert validation["mapping_rows"] == 6
    assert validation["canonical_mapping_rows"] == 4
    assert validation["mapping_collisions"] == 1
    assert validation["discarded_aliases"] == 2
    checksums = (output_root / "checksums.sha256").read_text()
    assert "Keyframes_L21/keyframes/L21_V001/0003.jpg" in checksums
    assert "Keyframes_L21/keyframes/L21_V001/10.jpg" in checksums


def test_ingest_resume_is_idempotent(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Reuse complete video shards without duplicating canonical records."""

    dataset_root, output_root = aic_dataset
    first_path = ingest_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )
    first = pd.read_parquet(first_path)

    second_path = ingest_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )
    second = pd.read_parquet(second_path)
    report = json.loads(
        (output_root / "reports" / "extraction_report.json").read_text()
    )

    pd.testing.assert_frame_equal(first, second)
    assert second["frame_id"].is_unique
    assert report["created_frames"] == 0
    assert report["resumed_frames"] == 6
    assert report["successful_videos"] == []
    assert report["skipped_videos"] == ["L21_V001", "L21_V002"]


def test_resume_rebuilds_a_changed_mapping_shard(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Rebuild only the video whose authoritative mapping changed."""

    dataset_root, output_root = aic_dataset
    ingest_dataset(dataset_root, output_root, "aic2025_s1_v2")
    mapping_path = (
        dataset_root
        / "map-keyframes-aic25-b1"
        / "map-keyframes"
        / "L21_V001.csv"
    )
    mapping = pd.read_csv(mapping_path)
    mapping.loc[0, "pts_time"] = 0.06
    mapping.to_csv(mapping_path, index=False)

    frames_path = ingest_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )
    frames = pd.read_parquet(frames_path)
    report = json.loads(
        (output_root / "reports" / "extraction_report.json").read_text()
    )

    changed_timestamp = frames.loc[
        frames["frame_id"] == "L21_V001_00000007",
        "timestamp_ms",
    ].item()

    assert changed_timestamp == 60
    assert report["successful_videos"] == ["L21_V001"]
    assert report["skipped_videos"] == ["L21_V002"]
    assert report["created_frames"] == 3
    assert report["resumed_frames"] == 3


def test_bad_images_and_mapping_are_reported_without_aborting(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Report corrupt, missing, and malformed inputs as per-video failures."""

    dataset_root, output_root = aic_dataset
    keyframe_root = dataset_root / "Keyframes_L21" / "keyframes"
    (keyframe_root / "L21_V001" / "001.jpg").write_bytes(b"not an image")
    (keyframe_root / "L21_V001" / "0003.jpg").unlink()
    mapping_path = (
        dataset_root
        / "map-keyframes-aic25-b1"
        / "map-keyframes"
        / "L21_V002.csv"
    )
    pd.read_csv(mapping_path).drop(columns="frame_idx").to_csv(
        mapping_path,
        index=False,
    )

    inventory = inventory_corpus(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )
    frames_path = ingest_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )
    extraction = json.loads(
        (output_root / "reports" / "extraction_report.json").read_text()
    )

    assert inventory["counts"]["corrupt_images"] == 1
    assert inventory["mapping_coverage"]["missing_images"] == 1
    assert inventory["mapping_errors"][0]["video_id"] == "L21_V002"
    assert "frame_idx" in inventory["mapping_errors"][0]["error"]
    assert pd.read_parquet(frames_path).empty
    assert {failure["video_id"] for failure in extraction["failed_videos"]} == {
        "L21_V001",
        "L21_V002",
    }


def test_decreasing_frame_indexes_are_not_treated_as_collisions(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Reject decreasing source indexes instead of silently repairing them."""

    dataset_root, output_root = aic_dataset
    mapping_path = (
        dataset_root
        / "map-keyframes-aic25-b1"
        / "map-keyframes"
        / "L21_V001.csv"
    )
    mapping = pd.read_csv(mapping_path)
    mapping["frame_idx"] = [7, 6, 211]
    mapping.to_csv(mapping_path, index=False)

    inventory = inventory_corpus(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )
    ingest_dataset(dataset_root, output_root, "aic2025_s1_v2")
    extraction = json.loads(
        (output_root / "reports" / "extraction_report.json").read_text()
    )

    assert "Decreasing frame_idx" in inventory["mapping_errors"][0]["error"]
    assert extraction["failed_videos"][0]["video_id"] == "L21_V001"


def test_validation_accepts_canonical_metadata_and_writes_artifacts(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Validate a complete ingest and freeze its audit and checksum artifacts."""

    dataset_root, output_root = aic_dataset
    inventory_corpus(dataset_root, output_root, "aic2025_s1_v2")
    frames_path = ingest_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )

    report = validate_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
        deep=True,
        audit_limit=4,
        metadata_path=frames_path,
    )

    assert report["valid"] is True
    assert report["status"] == "passed"
    assert report["row_count"] == 6
    assert report["error_count"] == 0
    assert report["error_counts"] == {}
    assert report["audit_rows"] == 4
    assert {
        Path(path).name for path in report["outputs"].values()
    } == {
        "validation_report.json",
        "corpus_report.md",
        "audit_samples.csv",
        "mapping_collisions.csv",
        "checksums.sha256",
    }
    assert all(Path(path).is_file() for path in report["outputs"].values())
    checksum_lines = (output_root / "checksums.sha256").read_text().splitlines()
    assert len(checksum_lines) == 10
    assert any("output/metadata/frames.parquet" in line for line in checksum_lines)


def test_validation_reports_contract_and_mapping_failures(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Reject duplicate IDs, invalid records, missing files, and bad timestamps."""

    dataset_root, output_root = aic_dataset
    frames_path = ingest_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )
    frames = pd.read_parquet(frames_path)
    frames.loc[0, "timestamp_ms"] = 41
    frames.loc[1, "image_path"] = str(output_root / "missing.jpg")
    frames.loc[2, "width"] = 0
    frames = pd.concat([frames, frames.iloc[[0]]], ignore_index=True)
    frames.to_parquet(frames_path, index=False)

    report = validate_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
        metadata_path=frames_path,
    )

    assert report["valid"] is False
    assert report["status"] == "failed"
    assert report["error_count"] > 0
    assert {
        "duplicate_frame_id",
        "duplicate_video_frame",
        "record_schema",
        "image_missing",
        "mapping_timestamp",
    }.issubset(report["error_counts"])
    saved = json.loads(
        (output_root / "reports" / "validation_report.json").read_text()
    )
    assert saved["valid"] is False


def test_validation_reports_missing_frame_record_column(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Report a missing required metadata column without crashing validation."""

    dataset_root, output_root = aic_dataset
    frames_path = ingest_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
    )
    frames = pd.read_parquet(frames_path).drop(columns="height")
    frames.to_parquet(frames_path, index=False)

    report = validate_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
        metadata_path=frames_path,
    )

    assert report["valid"] is False
    assert report["error_counts"]["missing_columns"] == 1
    assert report["error_counts"]["record_schema"] == 6


def test_prepare_dataset_runs_inventory_ingest_and_validation(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Run the public pipeline end to end with a round-robin frame limit."""

    dataset_root, output_root = aic_dataset

    frames_path = prepare_dataset(
        dataset_root,
        output_root,
        "aic2025_s1_v2",
        limit=4,
        thumbnail_max_edge=40,
    )
    frames = pd.read_parquet(frames_path)
    validation = json.loads(
        (output_root / "reports" / "validation_report.json").read_text()
    )

    assert frames["frame_id"].tolist() == [
        "L21_V001_00000007",
        "L21_V001_00000081",
        "L21_V002_00000004",
        "L21_V002_00000099",
    ]
    assert validation["valid"] is True
    assert validation["row_count"] == 4
    assert (output_root / "reports" / "corpus_inventory.json").is_file()
    assert (output_root / "reports" / "extraction_report.json").is_file()


def test_cli_prepares_and_validates_from_environment(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Run the public CLI with environment-based paths and version."""

    dataset_root, output_root = aic_dataset
    project_root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": str(project_root / "src"),
        "HCMAI_DATASET_ROOT": str(dataset_root),
        "HCMAI_DATA_ROOT": str(output_root),
        "HCMAI_DATASET_VERSION": "aic2025_s1_v2",
    }
    prepare = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--limit",
            "2",
            "--thumbnail-max-edge",
            "24",
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    frames_path = output_root / "metadata" / "frames.parquet"
    metadata_before_validation = frames_path.read_bytes()
    validate = subprocess.run(
        [sys.executable, "scripts/prepare_data.py", "--validate-only"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert prepare.returncode == 0, prepare.stderr
    assert "Metadata ready" in prepare.stdout
    assert validate.returncode == 0, validate.stderr
    assert "Validation completed" in validate.stdout
    assert frames_path.read_bytes() == metadata_before_validation


def test_cli_returns_failure_when_preparation_is_invalid(
    aic_dataset: tuple[Path, Path],
) -> None:
    """Return non-zero and suppress success output after failed validation."""

    dataset_root, output_root = aic_dataset
    project_root = Path(__file__).resolve().parents[1]
    image_path = (
        dataset_root
        / "Keyframes_L21"
        / "keyframes"
        / "L21_V001"
        / "001.jpg"
    )
    image_path.unlink()
    environment = {
        **os.environ,
        "PYTHONPATH": str(project_root / "src"),
        "HCMAI_DATASET_ROOT": str(dataset_root),
        "HCMAI_DATA_ROOT": str(output_root),
        "HCMAI_DATASET_VERSION": "aic2025_s1_v2",
    }

    completed = subprocess.run(
        [sys.executable, "scripts/prepare_data.py"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Metadata ready" not in completed.stdout
    assert "Data preparation failed" in completed.stderr
