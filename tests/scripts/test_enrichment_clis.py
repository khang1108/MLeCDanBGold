"""Exit semantics for independently runnable enrichment CLIs."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from hcmai.data.enrichment.ocr.config import OCRConfig
from hcmai.data.enrichment.object_detection import ObjectDetectionConfig
from scripts import detect_objects as object_cli
from scripts import generate_ocr_enrichment as ocr_cli


def _job(tmp_path: Path) -> SimpleNamespace:
    """Return the minimum validated-shape job used by both CLI boundaries."""

    return SimpleNamespace(
        frames_path=tmp_path / "frames.parquet",
        data_root=tmp_path,
        frame_store_id="btc-v1",
        ocr=OCRConfig(checkpoint="fixture/ocr", revision="r1", device="cpu"),
        ocr_output_dir=tmp_path / "ocr",
        objects=ObjectDetectionConfig(model="fixture/yoloe.pt", device="cpu"),
        object_output_dir=tmp_path / "objects",
    )


def _dataset_args(tmp_path: Path) -> list[str]:
    """Return the complete runtime dataset contract required by each CLI."""

    return [
        "--version",
        "fixture-v1",
        "--source",
        "custom_raw_video",
        "--frame-store-id",
        "btc-v1",
        "--data-root",
        str(tmp_path),
        "--frames",
        str(tmp_path / "frames.parquet"),
        "--frame-store-output",
        str(tmp_path / "frame-store"),
    ]


@pytest.mark.parametrize("cli_name", ["ocr", "objects"])
def test_partial_frame_failures_exit_success_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cli_name: str,
) -> None:
    """Contained row failures are degraded artifacts, not command failures."""

    job = _job(tmp_path)
    caplog.set_level(logging.INFO)
    if cli_name == "ocr":
        monkeypatch.setattr(
            ocr_cli.EnrichmentJobConfig,
            "from_yaml",
            lambda path, **kwargs: job,
        )
        monkeypatch.setattr(
            ocr_cli.EnrichmentService,
            "generate_ocr",
            lambda *args, **kwargs: {
                "completed_frames": 2,
                "failed_frames": 1,
                "skipped_frames": 0,
            },
        )
        result = ocr_cli.main(
            [
                "--config",
                str(tmp_path / "enrichment.yaml"),
                "--app-config",
                str(tmp_path / "missing-app.yaml"),
                *_dataset_args(tmp_path),
            ]
        )
    else:
        monkeypatch.setattr(
            object_cli.EnrichmentJobConfig,
            "from_yaml",
            lambda path, **kwargs: job,
        )
        monkeypatch.setattr(
            object_cli,
            "run_yoloe",
            lambda *args, **kwargs: {
                "completed_frames": 2,
                "failed_frames": 1,
                "inference_completed_frames": 2,
                "inference_skipped_frames": 0,
            },
        )
        result = object_cli.main(
            [
                "--frames",
                str(tmp_path / "frames.parquet"),
                "--output",
                str(tmp_path / "objects"),
                *_dataset_args(tmp_path),
            ]
        )

    assert result == 0
    assert "DEGRADED" in caplog.text
    assert "failed=1" in caplog.text


@pytest.mark.parametrize("cli_name", ["ocr", "objects"])
def test_artifact_level_failure_remains_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_name: str,
) -> None:
    """Do not convert publication or artifact corruption into CLI success."""

    job = _job(tmp_path)

    def fail_publication(*args: object, **kwargs: object) -> dict[str, object]:
        raise OSError("artifact publication failed")

    if cli_name == "ocr":
        monkeypatch.setattr(
            ocr_cli.EnrichmentJobConfig,
            "from_yaml",
            lambda path, **kwargs: job,
        )
        monkeypatch.setattr(
            ocr_cli.EnrichmentService, "generate_ocr", fail_publication
        )
        arguments = [
            "--config",
            str(tmp_path / "enrichment.yaml"),
            "--app-config",
            str(tmp_path / "missing-app.yaml"),
            *_dataset_args(tmp_path),
        ]
        command = ocr_cli.main
    else:
        monkeypatch.setattr(
            object_cli.EnrichmentJobConfig,
            "from_yaml",
            lambda path, **kwargs: job,
        )
        monkeypatch.setattr(object_cli, "run_yoloe", fail_publication)
        arguments = [
            "--frames",
            str(tmp_path / "frames.parquet"),
            "--output",
            str(tmp_path / "objects"),
            *_dataset_args(tmp_path),
        ]
        command = object_cli.main

    with pytest.raises(OSError, match="artifact publication failed"):
        command(arguments)
