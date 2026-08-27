"""Exercise custom extraction CLIs and the local native/Python release gate.

The acceptance test uses a synthetic source video and explicit temporary paths.
It proves the complete offline lifecycle without invoking yt-dlp, models, a
cloud provider, or the organizer corpus.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import pandas as pd
from PIL import Image
import pytest

from hcmai.data.ingestion.custom_enrichment import (
    materialize_video_enrichment_frames,
    write_enrichment_handoff,
)
from hcmai.data.ingestion.custom_frames import (
    CustomFrameStoreConfig,
    iter_native_frame_records,
    materialize_custom_frame_store,
    validate_native_video_bundle,
)
from hcmai.data.ingestion.custom_manifest import (
    build_native_input_manifest,
    write_extraction_config,
)
from hcmai.data.ingestion.custom_state import (
    cleanup_video,
    mark_video_enriched,
    mark_video_published,
)
from hcmai.data.stores.frame import FrameStore
from scripts import (
    extract_custom_keyframes,
    materialize_custom_frames,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NATIVE_EXECUTABLE = _PROJECT_ROOT / "build" / "keyframes_extraction" / "keyframe_extractor"


def _make_synthetic_source(source_root: Path, video_id: str) -> Path:
    """Create the deterministic three-second local video used by the smoke gate.

    Args:
        source_root: Directory receiving the fixture ``{video_id}.mp4`` file.
        video_id: Canonical source ID encoded into the fixture filename.

    Returns:
        Existing path to the generated two-FPS MP4 source.

    Raises:
        subprocess.CalledProcessError: If local FFmpeg cannot encode the fixture.
    """

    source_root.mkdir(parents=True, exist_ok=True)
    source_path = source_root / f"{video_id}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=80x40:rate=2:duration=3",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            str(source_path),
        ],
        check=True,
        shell=False,
        capture_output=True,
        text=True,
    )
    return source_path


def _make_fake_yt_dlp(path: Path) -> Path:
    """Create a deterministic downloader double for the native network branch."""

    path.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import sys

output = sys.argv[sys.argv.index("--output") + 1].replace("%(ext)s", "mp4")
Path(output).parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(os.environ["HCMAI_TEST_DOWNLOAD_SOURCE"], output)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_synthetic_media_info(media_info_dir: Path, video_id: str) -> None:
    """Write the one metadata row consumed by the source-root extraction path.

    Args:
        media_info_dir: Directory receiving the organizer-shaped JSON record.
        video_id: Canonical video identifier that must match the source filename.

    Returns:
        None; writes a three-second metadata-only fixture.
    """

    media_info_dir.mkdir(parents=True, exist_ok=True)
    (media_info_dir / f"{video_id}.json").write_text(
        json.dumps(
            {
                "watch_url": f"https://youtube.com/watch?v=synthetic-{video_id}",
                "length": 3,
            }
        ),
        encoding="utf-8",
    )


def _write_asr_fixture(path: Path, video_id: str) -> Path:
    """Write a minimal valid timeline-native ASR artifact for lifecycle testing.

    Args:
        path: Parquet destination inside the selected staging bundle.
        video_id: Canonical source-video identity for the synthetic segment.

    Returns:
        Existing Parquet path with one valid sub-second ASR segment.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "segment_id": f"{video_id}_segment_000000",
                "video_id": video_id,
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 500,
                "text": "synthetic smoke speech",
            }
        ]
    ).to_parquet(path, index=False)
    return path


def _assert_readable_jpegs(run_root: Path, image_paths: list[str]) -> None:
    """Verify that every run-root-relative JPEG can be decoded by Pillow.

    Args:
        run_root: Native lifecycle root that owns the image paths.
        image_paths: Portable paths emitted through canonical FrameRecord rows.

    Returns:
        None; returns only if every JPEG header and byte stream is readable.
    """

    for relative_path in image_paths:
        image_path = run_root / relative_path
        with Image.open(image_path) as image:
            image.verify()


def _native_executable() -> Path:
    """Return the built repository-local native executable required by the gate.

    Returns:
        Existing ``keyframe_extractor`` executable in the standard build tree.

    Raises:
        AssertionError: If the caller did not build the native package first.
    """

    assert _NATIVE_EXECUTABLE.is_file(), (
        "build the native extractor before running this release gate: "
        "cmake --build build/keyframes_extraction --parallel"
    )
    return _NATIVE_EXECUTABLE


def test_extract_cli_downloads_selected_batch_and_prepares_enrichment_tables(
    tmp_path: Path,
    capsys: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download, extract, and prepare one bounded batch in a single command."""

    native_executable = _native_executable()
    media_info = tmp_path / "media-info"
    source_root = tmp_path / "source-fixtures"
    run_root = tmp_path / "run"
    for video_id in ("L01_V001", "L01_V002"):
        _write_synthetic_media_info(media_info, video_id)
        _make_synthetic_source(source_root, video_id)
    fake_yt_dlp = _make_fake_yt_dlp(tmp_path / "fake-yt-dlp")
    monkeypatch.setenv(
        "HCMAI_TEST_DOWNLOAD_SOURCE",
        str(source_root / "L01_V001.mp4"),
    )

    assert extract_custom_keyframes.main(
        [
            "--media-info-dir",
            str(media_info),
            "--run-root",
            str(run_root),
            "--native-executable",
            str(native_executable),
            "--frame-store-id",
            "custom-test-v1",
            "--yt-dlp-binary",
            str(fake_yt_dlp),
            "--limit",
            "1",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["selected_video_ids"] == ["L01_V001"]
    assert result["completed"] == 1
    assert result["failed"] == 0
    assert result["emitted_frame_count"] == 3
    assert result["enrichment_ready_video_ids"] == ["L01_V001"]
    assert not (run_root / "state" / "L01_V002.json").exists()
    assert (run_root / "source" / "L01_V001.part").is_file()
    enrichment_root = run_root / "staging" / "L01_V001" / "enrichment"
    durable = pd.read_parquet(enrichment_root / "durable_frames.parquet")
    ocr = pd.read_parquet(enrichment_root / "ocr_frames.parquet")
    assert durable["frame_id"].tolist() == ocr["frame_id"].tolist()
    assert durable["image_path"].tolist() != ocr["image_path"].tolist()


def test_materialize_cli_never_invokes_native_extraction(
    tmp_path: Path,
    capsys: object,
) -> None:
    """Materialize an already published bundle through metadata/Parquet only."""

    from tests.data.test_custom_frames import write_valid_native_bundle

    write_valid_native_bundle(tmp_path, "L01_V001", count=1)
    output_root = tmp_path / "corpus"

    assert materialize_custom_frames.main(
        [
            "--run-root",
            str(tmp_path),
            "--output-root",
            str(output_root),
            "--frame-store-id",
            "custom-test-v1",
            "--video-id",
            "L01_V001",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert Path(result["frames_path"]).is_file()
    assert result["frame_count"] == 1


def test_complete_local_native_python_smoke_path(tmp_path: Path) -> None:
    """Gate the complete custom lifecycle with a local synthetic three-second video.

    The fixture uses ``--source-root`` so it exercises the production native CLI
    and Python boundaries without downloader, model, cloud, or organizer-data
    access. Identity-only temporary tables stand in for specialist inference;
    their role is to verify handoff lineage and lifecycle sequencing.

    Args:
        tmp_path: Isolated pytest directory for the native run and source fixture.

    Returns:
        None; asserts the published custom FrameStore is readable after cleanup.
    """

    video_id = "L01_V001"
    native_executable = _native_executable()
    run_root = tmp_path / "run"
    media_info_dir = tmp_path / "media-info"
    source_root = tmp_path / "synthetic-source"
    _write_synthetic_media_info(media_info_dir, video_id)
    _make_synthetic_source(source_root, video_id)

    manifest_path = build_native_input_manifest(
        media_info_dir,
        run_root / "input" / "media_manifest.jsonl",
    )
    config_path = write_extraction_config(
        run_root / "input" / "extraction_config.json",
        run_root=run_root,
        native_executable=native_executable,
        frame_store_id="custom-smoke-v1",
        yt_dlp_binary="yt-dlp",
    )
    extraction = subprocess.run(
        [
            str(native_executable),
            "extract",
            "--manifest",
            str(manifest_path),
            "--run-root",
            str(run_root),
            "--config",
            str(config_path),
            "--source-root",
            str(source_root),
            "--fail-fast",
        ],
        check=True,
        shell=False,
        capture_output=True,
        text=True,
    )
    assert json.loads(extraction.stdout) == {
        "completed": 1,
        "emitted_frame_count": 3,
        "failed": 0,
        "pending": 0,
        "skipped": 0,
    }

    staging_bundle = run_root / "staging" / video_id
    staging_report = validate_native_video_bundle(
        staging_bundle,
        run_root=run_root,
        expected_status="enrichment_pending",
    )
    durable_records = list(
        iter_native_frame_records(
            staging_bundle,
            run_root=run_root,
            image_variant="durable",
        )
    )
    ocr_records = list(
        iter_native_frame_records(
            staging_bundle,
            run_root=run_root,
            image_variant="enrichment",
        )
    )
    native_rows = [
        json.loads(line)
        for line in (staging_bundle / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert staging_report.frame_count == 3
    assert [record.timestamp_ms for record in durable_records] == sorted(
        record.timestamp_ms for record in durable_records
    )
    assert [record.frame_id for record in durable_records] == [
        f"{video_id}_raw1fps_{sample_index:09d}"
        for sample_index in range(3)
    ]
    assert [row["frame_idx"] for row in native_rows] == [
        math.floor(math.ceil(float(row["avg_fps"])) * int(row["timestamp_ms"]) / 1_000)
        for row in native_rows
    ]
    _assert_readable_jpegs(
        run_root,
        [record.image_path for record in durable_records],
    )
    _assert_readable_jpegs(
        run_root,
        [record.image_path for record in ocr_records],
    )

    durable_path = materialize_video_enrichment_frames(
        staging_bundle,
        staging_bundle / "enrichment" / "durable_frames.parquet",
        image_variant="durable",
    )
    ocr_path = materialize_video_enrichment_frames(
        staging_bundle,
        staging_bundle / "enrichment" / "ocr_frames.parquet",
        image_variant="enrichment",
    )
    asr_path = _write_asr_fixture(
        staging_bundle / "enrichment" / "asr.parquet",
        video_id,
    )
    handoff_path = write_enrichment_handoff(
        staging_bundle,
        artifact_paths={
            "caption": durable_path,
            "ocr": ocr_path,
            "objects": durable_path,
            "asr": asr_path,
        },
        output_path=staging_bundle / "enrichment" / "handoff.json",
        frame_store_id="custom-smoke-v1",
    )
    mark_video_enriched(native_executable, run_root, video_id, handoff_path)
    mark_video_published(
        native_executable,
        run_root,
        video_id,
        staging_bundle / "manifest.json",
    )
    cleanup_video(native_executable, run_root, video_id)

    published_bundle = run_root / "published" / video_id
    published_report = validate_native_video_bundle(
        published_bundle,
        run_root=run_root,
        expected_status="published",
    )
    assert published_report.frame_count == 3
    assert not (run_root / "source" / f"{video_id}.part").exists()
    assert not staging_bundle.exists()
    assert not (published_bundle / "enrichment_images").exists()
    _assert_readable_jpegs(
        run_root,
        [record.image_path for record in iter_native_frame_records(published_bundle, run_root=run_root)],
    )

    frames_path = materialize_custom_frame_store(
        CustomFrameStoreConfig(
            run_root=run_root,
            output_root=run_root / "corpus",
            frame_store_id="custom-smoke-v1",
            selected_video_ids=(video_id,),
        )
    )
    frame_store = FrameStore(frames_path)
    final_records = list(frame_store.iter_frames())
    assert len(frame_store) == 3
    assert [record.timestamp_ms for record in final_records] == [
        record.timestamp_ms for record in durable_records
    ]
    assert all(record.image_path.startswith(f"published/{video_id}/images/") for record in final_records)
