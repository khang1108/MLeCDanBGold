"""Tests for the explicit Filter catalog build command."""

from __future__ import annotations

import json

from pathlib import Path

import pandas as pd

from scripts.build_filter_catalog import main


def test_build_filter_catalog_cli_prints_machine_readable_report(
    tmp_path: Path,
    capsys,
) -> None:
    """Let operators record catalog identity without starting online models."""

    frames_path = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "L21_V001_000001",
                "video_id": "L21_V001",
                "frame_idx": 25,
                "timestamp_ms": 1000,
                "image_path": "keyframes/L21_V001/1.jpg",
            }
        ]
    ).to_parquet(frames_path, index=False)
    output_path = tmp_path / "filter.sqlite"

    exit_code = main(
        [
            "--frames", str(frames_path),
            "--output", str(output_path),
            "--catalog-version", "cli-fixture-v1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["catalog_version"] == "cli-fixture-v1"
    assert payload["frame_count"] == 1
    assert payload["output_size_bytes"] > 0
    assert payload["availability"]["caption"] is False
    assert output_path.is_file()
