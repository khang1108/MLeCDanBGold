"""Tests for deterministic metadata-only custom extraction preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from offline.ingestion.custom_manifest import (
    build_native_input_manifest,
    write_extraction_config,
)


def _write_media_info(path: Path, *, watch_url: str, length: object) -> None:
    """Write the minimum organizer media-info record used by a fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"watch_url": watch_url, "length": length}),
        encoding="utf-8",
    )


def test_build_native_input_manifest_is_sorted_and_strict(tmp_path: Path) -> None:
    """Sort source filenames and serialize the native JSONL contract exactly."""

    media_info = tmp_path / "media-info"
    _write_media_info(
        media_info / "L01_V002.json",
        watch_url="https://youtube.com/watch?v=b",
        length=4,
    )
    _write_media_info(
        media_info / "L01_V001.json",
        watch_url="https://youtube.com/watch?v=a",
        length=3,
    )

    output = build_native_input_manifest(media_info, tmp_path / "input.jsonl")

    assert output.read_text(encoding="utf-8").splitlines() == [
        '{"video_id":"L01_V001","watch_url":"https://youtube.com/watch?v=a","metadata_length_s":3}',
        '{"video_id":"L01_V002","watch_url":"https://youtube.com/watch?v=b","metadata_length_s":4}',
    ]


@pytest.mark.parametrize(
    ("length", "message"),
    [(True, "integer"), (3.0, "integer"), (-1, "non-negative")],
)
def test_manifest_rejects_non_integral_or_negative_lengths(
    tmp_path: Path,
    length: object,
    message: str,
) -> None:
    """Reject values that could otherwise be silently truncated for planning."""

    media_info = tmp_path / "media-info"
    _write_media_info(
        media_info / "L01_V001.json",
        watch_url="https://youtube.com/watch?v=a",
        length=length,
    )

    with pytest.raises(ValueError, match=message):
        build_native_input_manifest(media_info, tmp_path / "input.jsonl")


def test_manifest_rejects_duplicate_urls_and_blank_url(tmp_path: Path) -> None:
    """Reject ambiguous or unusable source acquisition records before native work."""

    duplicate_root = tmp_path / "duplicate"
    _write_media_info(
        duplicate_root / "L01_V001.json",
        watch_url="https://youtube.com/watch?v=same",
        length=3,
    )
    _write_media_info(
        duplicate_root / "L01_V002.json",
        watch_url="https://youtube.com/watch?v=same",
        length=4,
    )
    with pytest.raises(ValueError, match="duplicate watch_url"):
        build_native_input_manifest(duplicate_root, tmp_path / "duplicate.jsonl")

    blank_root = tmp_path / "blank"
    _write_media_info(
        blank_root / "L01_V001.json",
        watch_url="  ",
        length=3,
    )
    with pytest.raises(ValueError, match="watch_url"):
        build_native_input_manifest(blank_root, tmp_path / "blank.jsonl")


def test_write_extraction_config_hashes_its_canonical_payload(tmp_path: Path) -> None:
    """Keep native config provenance reproducible and independent of JSON spacing."""

    output = write_extraction_config(
        tmp_path / "input" / "extraction_config.json",
        run_root=tmp_path / "run",
        native_executable=tmp_path / "build" / "keyframe_extractor",
        frame_store_id="custom-raw1fps-v1",
        yt_dlp_binary="yt-dlp",
    )

    config = json.loads(output.read_text(encoding="utf-8"))
    operational_fields = {
        "config_hash",
        "yt_dlp_cookies_path",
        "yt_dlp_js_runtime",
        "disk_reserve_bytes",
    }
    payload = {
        key: value for key, value in config.items() if key not in operational_fields
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert config["config_hash"] == hashlib.sha256(encoded).hexdigest()
    assert config["sample_period_ms"] == 1_000
    assert config["durable_long_edge"] == 1_024
    assert config["durable_jpeg_quality"] == 92
    assert config["enrichment_jpeg_quality"] == 95
    assert config["write_enrichment_images"] is True
    assert config["extractor_version"] == "hcmai-keyframes-extractor/0.1.0"
    assert config["yt_dlp_cookies_path"] is None
    assert config["yt_dlp_js_runtime"] is None

    authenticated_output = write_extraction_config(
        tmp_path / "input" / "authenticated_extraction_config.json",
        run_root=tmp_path / "run",
        native_executable=tmp_path / "build" / "keyframe_extractor",
        frame_store_id="custom-raw1fps-v1",
        yt_dlp_binary="yt-dlp",
        yt_dlp_cookies_path=tmp_path / "secrets" / "youtube.cookies.txt",
        yt_dlp_js_runtime="node",
    )
    authenticated = json.loads(authenticated_output.read_text(encoding="utf-8"))
    assert authenticated["config_hash"] == config["config_hash"]
    assert authenticated["yt_dlp_cookies_path"].endswith("youtube.cookies.txt")
    assert authenticated["yt_dlp_js_runtime"] == "node"
