"""Tests for the resumable non-S3 custom preparation orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd
import pytest

from scripts import prepare_custom_pipeline as pipeline
from tests.data.test_custom_frames import write_valid_native_bundle


def test_finalize_batch_chunk_size_is_configurable_from_cli(tmp_path: Path) -> None:
    """Expose a positive committed-batch ceiling for RAM-bounded finalize."""

    args = pipeline.parse_args(
        [
            "finalize",
            "--run-root",
            str(tmp_path / "run"),
            "--output-root",
            str(tmp_path / "output"),
            "--version",
            "custom-dataset-v1",
            "--frame-store-id",
            "custom-v1",
            "--transcripts-root",
            str(tmp_path / "transcripts"),
            "--asr-index-root",
            str(tmp_path / "asr-index"),
            "--archive-url",
            "https://example.com/Videos_L01_a.zip",
            "--finalize-batch-chunk-size",
            "32",
        ]
    )

    assert args.finalize_batch_chunk_size == 32


def test_finalize_batch_chunk_size_rejects_zero(tmp_path: Path) -> None:
    """Reject a zero-sized chunk before finalization touches artifacts."""

    with pytest.raises(SystemExit):
        pipeline.parse_args(
            [
                "finalize",
                "--run-root",
                str(tmp_path / "run"),
                "--output-root",
                str(tmp_path / "output"),
                "--version",
                "custom-dataset-v1",
                "--frame-store-id",
                "custom-v1",
                "--transcripts-root",
                str(tmp_path / "transcripts"),
                "--asr-index-root",
                str(tmp_path / "asr-index"),
                "--archive-url",
                "https://example.com/Videos_L01_a.zip",
                "--finalize-batch-chunk-size",
                "0",
            ]
        )


def _argument_value(arguments: list[str], name: str) -> Path:
    """Return one path argument captured from a stage command."""

    return Path(arguments[arguments.index(name) + 1])


def _pipeline_args(tmp_path: Path, media_info_url: str) -> argparse.Namespace:
    """Parse a minimal auto-download invocation for metadata bootstrap tests."""

    return pipeline.parse_args(
        [
            "--media-info-url",
            media_info_url,
            "--run-root",
            str(tmp_path / "run"),
            "--output-root",
            str(tmp_path / "output"),
            "--version",
            "custom-dataset-v1",
            "--frame-store-id",
            "custom-v1",
            "--limit",
            "1",
        ]
    )


def test_pipeline_downloads_and_reuses_default_media_info_layout(
    tmp_path: Path,
) -> None:
    """Missing local metadata downloads once and resolves media-info/*.json."""

    source_archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(source_archive, "w") as archive:
        archive.writestr(
            "media-info/L01_V001.json",
            json.dumps(
                {
                    "watch_url": "https://youtube.com/watch?v=synthetic",
                    "length": 3,
                }
            ),
        )
    args = _pipeline_args(tmp_path, source_archive.as_uri())

    first = pipeline._resolve_media_info_dir(args)
    source_archive.unlink()
    second = pipeline._resolve_media_info_dir(args)

    assert first == second
    assert first.name == "media-info"
    assert (first / "L01_V001.json").is_file()
    assert (
        tmp_path / "run" / "input" / pipeline.MEDIA_INFO_ARCHIVE_NAME
    ).is_file()


def test_pipeline_rejects_media_info_zip_path_traversal(tmp_path: Path) -> None:
    """Organizer metadata bootstrap must not write outside its extraction root."""

    source_archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source_archive, "w") as archive:
        archive.writestr("../escaped.json", "{}")
    args = _pipeline_args(tmp_path, source_archive.as_uri())

    with pytest.raises(ValueError, match="unsafe media-info ZIP member"):
        pipeline._resolve_media_info_dir(args)

    assert not (tmp_path / "run" / "input" / "escaped.json").exists()


def test_pipeline_forwards_yt_dlp_authentication_options(tmp_path: Path) -> None:
    """Top-level cookies/runtime options must reach the extraction CLI contract."""

    cookies = tmp_path / "youtube.cookies.txt"
    args = pipeline.parse_args(
        [
            "--run-root",
            str(tmp_path / "run"),
            "--output-root",
            str(tmp_path / "output"),
            "--version",
            "custom-dataset-v1",
            "--frame-store-id",
            "custom-v1",
            "--yt-dlp-cookies",
            str(cookies),
            "--yt-dlp-js-runtime",
            "node",
            "--limit",
            "1",
        ]
    )

    extraction = pipeline._extraction_args(args)

    assert extraction.yt_dlp_cookies == cookies
    assert extraction.yt_dlp_js_runtime == "node"


def test_pipeline_surfaces_and_persists_native_extraction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The top-level command must retain the native per-video root cause."""

    video_id = "L01_V001"
    media_info_dir = tmp_path / "media-info"
    media_info_dir.mkdir()
    (media_info_dir / f"{video_id}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        pipeline.extract_custom_keyframes,
        "run",
        lambda args: {
            "failed": 1,
            "selected_video_ids": [video_id],
            "failures": [
                {
                    "video_id": video_id,
                    "error": "yt-dlp exited with status 1",
                }
            ],
        },
    )
    args = pipeline.parse_args(
        [
            "--media-info-dir",
            str(media_info_dir),
            "--run-root",
            str(tmp_path / "run"),
            "--output-root",
            str(tmp_path / "output"),
            "--native-executable",
            "/bin/true",
            "--version",
            "custom-dataset-v1",
            "--frame-store-id",
            "custom-v1",
            "--limit",
            "1",
        ]
    )

    with pytest.raises(
        RuntimeError,
        match=r"L01_V001: yt-dlp exited with status 1.*extraction_report.json",
    ):
        pipeline.run(args)

    report = json.loads(
        (tmp_path / "run" / "input" / "extraction_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["failures"][0]["video_id"] == video_id


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
    media_info_dir = tmp_path / "media-info"
    media_info_dir.mkdir()
    (media_info_dir / f"{video_id}.json").write_text("{}", encoding="utf-8")

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
            str(media_info_dir),
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
