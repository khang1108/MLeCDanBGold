"""Tests for the resumable non-S3 custom preparation orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from scripts import prepare_custom_pipeline as pipeline
from tests.data.test_custom_frames import write_valid_native_bundle


def _argument_value(arguments: list[str], name: str) -> Path:
    """Return one path argument captured from a stage command."""

    return Path(arguments[arguments.index(name) + 1])


def test_custom_pipeline_coordinates_every_local_artifact_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run extraction through indexes without requiring BTC mapping or S3."""

    video_id = "L01_V001"
    run_root = tmp_path / "run"
    output_root = tmp_path / "output"
    bundle = write_valid_native_bundle(
        run_root,
        video_id,
        count=2,
        status="enrichment_pending",
    )
    source = run_root / "source" / f"{video_id}.part"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic-video")
    events: list[str] = []

    monkeypatch.setattr(
        pipeline.extract_custom_keyframes,
        "run",
        lambda args: {
            "failed": 0,
            "selected_video_ids": [video_id],
            "enrichment_ready_video_ids": [video_id],
            "published_video_ids": [],
        },
    )

    def fake_stage(script: str, raw_arguments: object) -> None:
        arguments = list(raw_arguments)  # type: ignore[arg-type]
        events.append(script)
        frames = pd.read_parquet(_argument_value(arguments, "--frames"))
        if script == "generate_enrichment.py":
            output = _argument_value(arguments, "--output")
            output.mkdir(parents=True, exist_ok=True)
            frames.assign(
                frame_store_id="custom-v1",
                status="completed",
            ).to_parquet(output / "captions.parquet", index=False)
        elif script == "generate_ocr_enrichment.py":
            output = _argument_value(arguments, "--output")
            output.mkdir(parents=True, exist_ok=True)
            frames.assign(
                frame_store_id="custom-v1",
                status="completed",
            ).to_parquet(output / "frames.parquet", index=False)
        elif script == "detect_objects.py":
            output = _argument_value(arguments, "--output")
            output.mkdir(parents=True, exist_ok=True)
            frames.assign(
                frame_store_id="custom-v1",
                status="completed",
            ).to_parquet(output / "frames.parquet", index=False)
        elif script == "prepare_transcripts.py":
            output = _argument_value(arguments, "--output") / "L01"
            output.mkdir(parents=True, exist_ok=True)
            transcript = output / f"{video_id}.parquet"
            pd.DataFrame(
                [{"video_id": video_id, "start_ms": 0, "end_ms": 500}]
            ).to_parquet(transcript, index=False)
            transcript.with_suffix(".manifest.json").write_text(
                json.dumps({"video_id": video_id}), encoding="utf-8"
            )
        elif script == "build_frame_context.py":
            output = _argument_value(arguments, "--output")
            output.mkdir(parents=True, exist_ok=True)
            (output / "frame_context_v1.parquet").write_bytes(b"context")
        elif script == "build_retrieval_indexes.py":
            output = _argument_value(arguments, "--output-root")
            output.mkdir(parents=True, exist_ok=True)
            (output / "build_report.json").write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )

    monkeypatch.setattr(pipeline, "_run_python", fake_stage)
    monkeypatch.setattr(pipeline, "mark_video_enriched", lambda *args: None)

    def publish(*args: object) -> None:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest["status"] = "published"
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        destination = run_root / "published" / video_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(bundle), destination)

    monkeypatch.setattr(pipeline, "mark_video_published", publish)
    monkeypatch.setattr(pipeline, "cleanup_video", lambda *args: None)

    args = pipeline.parse_args(
        [
            "--media-info-dir",
            str(tmp_path / "media-info"),
            "--run-root",
            str(run_root),
            "--output-root",
            str(output_root),
            "--native-executable",
            "/bin/true",
            "--version",
            "custom-dataset-v1",
            "--source",
            "custom_raw_video_1fps",
            "--frame-store-id",
            "custom-v1",
            "--video-id",
            video_id,
        ]
    )

    report = pipeline.run(args)

    assert report["status"] == "passed"
    assert report["frame_count"] == 2
    assert events == [
        "generate_enrichment.py",
        "generate_ocr_enrichment.py",
        "detect_objects.py",
        "prepare_transcripts.py",
        "build_frame_context.py",
        "build_retrieval_indexes.py",
    ]
    assert (output_root / "frame_store" / "frames.parquet").is_file()
    assert (output_root / "prepare_report.json").is_file()
