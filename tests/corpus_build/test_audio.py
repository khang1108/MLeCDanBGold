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


def test_extract_flac_declares_format_for_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final ``.partial`` suffix must not hide the FLAC container."""

    calls: list[list[str]] = []

    class Result:
        returncode = 1
        stderr = b"decoder stopped"
        stdout = b""

    def fake_run(command: list[str], **kwargs: object) -> Result:
        calls.append(command)
        return Result()

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="decoder stopped"):
        audio.extract_flac(
            tmp_path / "L23_V022.mp4",
            tmp_path / "audio.flac",
            16_000,
        )

    assert "-f" in calls[0]
    assert calls[0][calls[0].index("-f") + 1] == "flac"
