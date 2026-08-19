"""Tests for bounded audio extraction diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from hcmai.data.corpus_build import audio


def test_extract_flac_reports_ffmpeg_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the decoder reason when ffmpeg rejects one staged video."""

    class Result:
        returncode = 1
        stderr = b"Stream map '0:a:0' matches no streams."
        stdout = b""

    def fake_run(*args: object, **kwargs: object) -> Result:
        return Result()

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="matches no streams"):
        audio.extract_flac(
            tmp_path / "L23_V023.mp4",
            tmp_path / "audio.flac",
            16_000,
        )
