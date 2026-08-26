"""Tests for metadata-only custom extraction command-line preparation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import prepare_custom_extraction


def test_prepare_cli_writes_inputs_and_reports_metadata_only_stats(
    tmp_path: Path,
    capsys: object,
) -> None:
    """Create a native manifest/config without invoking a downloader or decoder."""

    media_info = tmp_path / "media-info"
    media_info.mkdir()
    (media_info / "L01_V001.json").write_text(
        json.dumps(
            {
                "watch_url": "https://youtube.com/watch?v=a",
                "length": 3,
            }
        ),
        encoding="utf-8",
    )
    (media_info / "L01_V002.json").write_text(
        json.dumps(
            {
                "watch_url": "https://youtube.com/watch?v=b",
                "length": 4,
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"

    assert prepare_custom_extraction.main(
        [
            "--media-info-dir",
            str(media_info),
            "--run-root",
            str(run_root),
            "--native-executable",
            str(tmp_path / "keyframe_extractor"),
            "--frame-store-id",
            "custom-test-v1",
            "--yt-dlp-binary",
            "yt-dlp",
        ]
    ) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["video_count"] == 2
    assert result["unique_url_count"] == 2
    assert result["metadata_length_seconds"] == 7
    assert result["sample_period_ms"] == 1_000
    assert Path(result["manifest_path"]).is_file()
    assert Path(result["config_path"]).is_file()
