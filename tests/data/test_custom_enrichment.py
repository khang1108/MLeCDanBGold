"""Tests for per-video custom enrichment inputs and handoff validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hcmai.data.ingestion.custom_enrichment import (
    materialize_video_enrichment_frames,
    write_enrichment_handoff,
)
from hcmai.data.ingestion.custom_frames import iter_native_frame_records
from tests.data.test_custom_frames import write_valid_native_bundle


def _canonical_rows(bundle: Path) -> list[dict[str, object]]:
    """Return exact identity rows from a staging bundle for specialist fixtures."""

    run_root = bundle.parent.parent
    return [
        {
            "frame_id": record.frame_id,
            "video_id": record.video_id,
            "frame_idx": record.frame_idx,
            "timestamp_ms": record.timestamp_ms,
            "frame_store_id": "custom-raw1fps-v1",
        }
        for record in iter_native_frame_records(bundle, run_root=run_root)
    ]


def _write_frame_artifact(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a simple frame-native specialist artifact with full identity fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_asr_artifact(path: Path, video_id: str) -> Path:
    """Write one timeline-native ASR artifact without forcing frame alignment."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "segment_id": f"{video_id}_segment_000000",
                "video_id": video_id,
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 500,
                "text": "fixture speech",
            }
        ]
    ).to_parquet(path, index=False)
    return path


def _valid_artifact_map(bundle: Path) -> dict[str, Path]:
    """Write one valid path per specialist modality for a handoff fixture."""

    rows = _canonical_rows(bundle)
    root = bundle / "enrichment"
    return {
        "caption": _write_frame_artifact(root / "caption.parquet", rows),
        "ocr": _write_frame_artifact(root / "ocr.parquet", rows),
        "objects": _write_frame_artifact(root / "objects.parquet", rows),
        "asr": _write_asr_artifact(root / "asr.parquet", "L01_V001"),
    }


def test_enrichment_variants_preserve_identity_but_switch_image_path(
    tmp_path: Path,
) -> None:
    """Keep canonical metadata stable while selecting durable or OCR image bytes."""

    bundle = write_valid_native_bundle(
        tmp_path,
        "L01_V001",
        count=2,
        status="enrichment_pending",
    )

    durable = materialize_video_enrichment_frames(
        bundle,
        tmp_path / "durable.parquet",
        image_variant="durable",
    )
    high_res = materialize_video_enrichment_frames(
        bundle,
        tmp_path / "ocr.parquet",
        image_variant="enrichment",
    )

    durable_rows = pd.read_parquet(durable)
    high_res_rows = pd.read_parquet(high_res)
    assert durable_rows[["frame_id", "frame_idx", "timestamp_ms"]].equals(
        high_res_rows[["frame_id", "frame_idx", "timestamp_ms"]]
    )
    assert durable_rows["image_path"].str.contains("/images/").all()
    assert high_res_rows["image_path"].str.contains("/enrichment_images/").all()


def test_handoff_rejects_artifact_frame_identity_mismatch(tmp_path: Path) -> None:
    """Reject a specialist file that cannot be joined to native frame identity."""

    bundle = write_valid_native_bundle(
        tmp_path,
        "L01_V001",
        count=2,
        status="enrichment_pending",
    )
    artifacts = _valid_artifact_map(bundle)
    artifacts["caption"] = _write_frame_artifact(
        bundle / "enrichment" / "bad-caption.parquet",
        [
            {
                "frame_id": "foreign-frame",
                "video_id": "L01_V001",
                "frame_idx": 0,
                "timestamp_ms": 0,
                "frame_store_id": "custom-raw1fps-v1",
            }
        ],
    )

    with pytest.raises(ValueError, match="caption frame identity"):
        write_enrichment_handoff(
            bundle,
            artifact_paths=artifacts,
            output_path=bundle / "enrichment" / "handoff.json",
            frame_store_id="custom-raw1fps-v1",
        )


def test_handoff_preserves_explicit_not_evaluated_status(tmp_path: Path) -> None:
    """Keep missing evaluation distinct from an invented negative artifact result."""

    bundle = write_valid_native_bundle(
        tmp_path,
        "L01_V001",
        count=1,
        status="enrichment_pending",
    )

    handoff = write_enrichment_handoff(
        bundle,
        artifact_paths={
            "caption": None,
            "ocr": None,
            "objects": None,
            "asr": None,
        },
        output_path=bundle / "enrichment" / "handoff.json",
        frame_store_id="custom-raw1fps-v1",
    )

    payload = json.loads(handoff.read_text(encoding="utf-8"))
    assert {entry["status"] for entry in payload["artifacts"].values()} == {
        "not_evaluated"
    }
    assert {entry["path"] for entry in payload["artifacts"].values()} == {""}


def _asr_artifact_ending_at(path: Path, video_id: str, end_ms: int) -> Path:
    """Write one ASR segment whose timeline ends at an exact millisecond."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "segment_id": f"{video_id}_segment_000000",
                "video_id": video_id,
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": end_ms,
                "text": "fixture speech",
            }
        ]
    ).to_parquet(path, index=False)
    return path


def test_handoff_tolerates_audio_stream_skew_but_rejects_a_different_cut(
    tmp_path: Path,
) -> None:
    """Accept millisecond stream skew past the decoded duration, reject a real overrun."""

    bundle = write_valid_native_bundle(
        tmp_path,
        "L01_V001",
        count=2,
        status="enrichment_pending",
    )
    artifacts = _valid_artifact_map(bundle)
    artifacts["asr"] = _asr_artifact_ending_at(
        bundle / "enrichment" / "asr-skewed.parquet", "L01_V001", 2062
    )
    write_enrichment_handoff(
        bundle,
        artifact_paths=artifacts,
        output_path=bundle / "enrichment" / "handoff.json",
        frame_store_id="custom-raw1fps-v1",
    )

    artifacts["asr"] = _asr_artifact_ending_at(
        bundle / "enrichment" / "asr-overrun.parquet", "L01_V001", 5000
    )
    with pytest.raises(ValueError, match="exceeds native video duration"):
        write_enrichment_handoff(
            bundle,
            artifact_paths=artifacts,
            output_path=bundle / "enrichment" / "handoff.json",
            frame_store_id="custom-raw1fps-v1",
        )
