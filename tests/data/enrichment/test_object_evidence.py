"""Behavior tests for strict BTC object evidence import."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pydantic import ValidationError

from hcmai.common.schemas import ObjectDetection, ObjectEvidence, ProcessingStatus
from hcmai.data.enrichment.objects.config import ObjectConfig
from hcmai.data.enrichment.objects.importer import import_objects


def _frames(tmp_path: Path, count: int = 1) -> Path:
    rows = [
        {
            "frame_id": f"L01_V001:{index:04d}",
            "video_id": "L01_V001",
            "frame_idx": 120 + index * 30,
            "keyframe_order": index + 1,
            "timestamp_ms": 4000 + index * 1000,
            "image_path": f"keyframes/L01_V001/{index:04d}.jpg",
            "width": 32,
            "height": 24,
        }
        for index in range(count)
    ]
    path = tmp_path / "frames.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_object(root: Path, stem: str, payload: object) -> None:
    path = root / "L01_V001" / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _config(tmp_path: Path, **updates: object) -> ObjectConfig:
    values = {
        "objects_root": tmp_path / "objects",
        "output_dir": tmp_path / "output",
    }
    values.update(updates)
    return ObjectConfig(**values)


def test_btc_object_import_preserves_counts_multiplicity_and_boxes(tmp_path):
    source = _frames(tmp_path)
    _write_object(
        tmp_path / "objects",
        "0000",
        {
            "detection_class_entities": [" Person ", "Person", "Car"],
            "detection_scores": [0.91, 0.83, 0.72],
            "detection_boxes": [
                [0.10, 0.10, 0.50, 0.30],
                [0.15, 0.40, 0.55, 0.60],
                [0.20, 0.65, 0.70, 0.95],
            ],
        },
    )

    report = import_objects(
        source,
        tmp_path / "objects",
        tmp_path / "output",
        _config(tmp_path),
        frame_store_id="btc-v1",
    )

    frames = pd.read_parquet(tmp_path / "output/frames.parquet")
    detections = pd.read_parquet(tmp_path / "output/detections.parquet")
    row = frames.iloc[0]
    assert json.loads(row["counts_json"]) == {"car": 1, "person": 2}
    assert row["summary"] == "person x2; car x1"
    assert row["detection_count"] == 3
    assert row["frame_id"] == "L01_V001:0000"
    assert row["video_id"] == "L01_V001"
    assert row["frame_idx"] == 120
    assert row["frame_store_id"] == "btc-v1"
    assert detections[
        ["frame_id", "video_id", "detection_index", "label"]
    ].values.tolist() == [
        ["L01_V001:0000", "L01_V001", 0, "person"],
        ["L01_V001:0000", "L01_V001", 1, "person"],
        ["L01_V001:0000", "L01_V001", 2, "car"],
    ]
    assert detections.iloc[0][
        ["x_min", "y_min", "x_max", "y_max"]
    ].tolist() == [0.10, 0.10, 0.30, 0.50]
    assert report["completed_frames"] == 1
    assert report["failed_frames"] == 0


def test_all_detections_are_retained_but_summary_is_thresholded(tmp_path):
    source = _frames(tmp_path)
    _write_object(
        tmp_path / "objects",
        "0000",
        {
            "detection_class_entities": ["Car", "Bicycle", "Person", "Car"],
            "detection_scores": [0.1, 0.8, 0.7, 0.6],
            "detection_boxes": [[0.0, 0.0, 1.0, 1.0]] * 4,
        },
    )

    import_objects(
        source,
        tmp_path / "objects",
        tmp_path / "output",
        _config(tmp_path, summary_min_confidence=0.5, max_summary_labels=2),
    )

    frames = pd.read_parquet(tmp_path / "output/frames.parquet")
    detections = pd.read_parquet(tmp_path / "output/detections.parquet")
    assert len(detections) == 4
    assert json.loads(frames.iloc[0]["counts_json"]) == {
        "bicycle": 1,
        "car": 1,
        "person": 1,
    }
    # Equal counts sort by descending maximum confidence, then label.
    assert frames.iloc[0]["summary"] == "bicycle x1; person x1"


def test_object_contract_allows_thresholded_counts_but_bounds_multiplicity():
    detections = [
        ObjectDetection(
            label="person",
            confidence=0.9,
            x_min=0,
            y_min=0,
            x_max=1,
            y_max=1,
        ),
        ObjectDetection(
            label="car",
            confidence=0.1,
            x_min=0,
            y_min=0,
            x_max=1,
            y_max=1,
        ),
    ]
    values = {
        "frame_id": "f1",
        "video_id": "v1",
        "frame_idx": 1,
        "detections": detections,
        "detection_count": 2,
        "artifact_version": "object-v1",
    }

    evidence = ObjectEvidence(**values, counts={"person": 1})

    assert evidence.counts == {"person": 1}
    with pytest.raises(ValidationError, match="raw detection multiplicity"):
        ObjectEvidence(**values, counts={"person": 2})
    with pytest.raises(ValidationError, match="raw detection multiplicity"):
        ObjectEvidence(**values, counts={"bicycle": 1})


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        {
            "detection_class_entities": ["Person"],
            "detection_scores": [],
            "detection_boxes": [],
        },
        {
            "detection_class_entities": ["Person"],
            "detection_scores": [float("nan")],
            "detection_boxes": [[0.0, 0.0, 1.0, 1.0]],
        },
        {
            "detection_class_entities": ["Person"],
            "detection_scores": [0.8],
            "detection_boxes": [[0.0, 0.0, 2.0, 1.0]],
        },
    ],
)
def test_malformed_object_json_fails_only_its_frame(tmp_path, payload):
    source = _frames(tmp_path, count=2)
    bad_path = tmp_path / "objects/L01_V001/0000.json"
    bad_path.parent.mkdir(parents=True)
    if isinstance(payload, str):
        bad_path.write_text(payload, encoding="utf-8")
    else:
        bad_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_object(
        tmp_path / "objects",
        "0001",
        {
            "detection_class_entities": [],
            "detection_scores": [],
            "detection_boxes": [],
        },
    )

    report = import_objects(
        source,
        tmp_path / "objects",
        tmp_path / "output",
        _config(tmp_path),
    )

    frames = pd.read_parquet(tmp_path / "output/frames.parquet")
    assert frames.frame_id.tolist() == ["L01_V001:0000", "L01_V001:0001"]
    assert frames.status.tolist() == [
        ProcessingStatus.FAILED.value,
        ProcessingStatus.COMPLETED.value,
    ]
    assert frames.iloc[0]["error_code"]
    assert 0 < len(frames.iloc[0]["error_message"]) <= 300
    assert frames.iloc[1]["detection_count"] == 0
    assert frames.iloc[1]["summary"] is None
    assert json.loads(frames.iloc[1]["counts_json"]) == {}
    assert report["completed_frames"] == 1
    assert report["failed_frames"] == 1


def test_missing_json_is_a_bounded_per_frame_failure(tmp_path):
    source = _frames(tmp_path)

    import_objects(
        source,
        tmp_path / "objects",
        tmp_path / "output",
        _config(tmp_path),
    )

    row = pd.read_parquet(tmp_path / "output/frames.parquet").iloc[0]
    assert row["status"] == ProcessingStatus.FAILED.value
    assert row["error_code"] == "FileNotFoundError"
    assert "0000.json" in row["error_message"]
    assert len(row["error_message"]) <= 300


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"artifact_version": ""}, "artifact_version"),
        ({"summary_min_confidence": -0.1}, "summary_min_confidence"),
        ({"summary_min_confidence": 1.1}, "summary_min_confidence"),
        ({"max_summary_labels": 0}, "max_summary_labels"),
    ],
)
def test_object_config_rejects_invalid_policy(tmp_path, updates, message):
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **updates)


def test_duplicate_canonical_frame_identity_is_rejected_before_publication(tmp_path):
    source = _frames(tmp_path)
    frames = pd.read_parquet(source)
    pd.concat([frames, frames], ignore_index=True).to_parquet(source, index=False)

    with pytest.raises(ValueError, match="duplicate frame_id"):
        import_objects(
            source,
            tmp_path / "objects",
            tmp_path / "output",
            _config(tmp_path),
        )

    assert not (tmp_path / "output/manifest.json").exists()
