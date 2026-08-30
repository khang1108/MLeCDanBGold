"""Tests for safe Python wrappers around native custom lifecycle commands."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from offline.ingestion.custom_state import (
    cleanup_video,
    mark_video_enriched,
    mark_video_published,
)


def test_mark_enriched_invokes_native_command_without_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pass every user-controlled argument as an explicit subprocess argv item."""

    calls: list[dict[str, object]] = []

    def fake_run(
        argv: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        """Capture a native command without executing an external process."""

        calls.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    mark_video_enriched(
        native_executable=tmp_path / "keyframe_extractor",
        run_root=tmp_path / "run",
        video_id="L01_V001",
        handoff_path=tmp_path / "handoff.json",
    )

    assert calls[0]["shell"] is False
    assert calls[0]["check"] is True
    assert calls[0]["capture_output"] is True
    assert calls[0]["text"] is True
    assert calls[0]["argv"] == [
        str(tmp_path / "keyframe_extractor"),
        "state",
        "mark-enriched",
        "--run-root",
        str(tmp_path / "run"),
        "--video-id",
        "L01_V001",
        "--artifacts",
        str(tmp_path / "handoff.json"),
    ]


def test_publication_and_cleanup_use_their_guarded_native_subcommands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retain separate manifest and cleanup transitions without direct JSON writes."""

    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Record each safe command-line vector for assertion."""

        assert kwargs["shell"] is False
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    executable = tmp_path / "keyframe_extractor"
    run_root = tmp_path / "run"
    mark_video_published(
        executable,
        run_root,
        "L01_V001",
        run_root / "staging" / "L01_V001" / "manifest.json",
    )
    cleanup_video(executable, run_root, "L01_V001")

    assert calls == [
        [
            str(executable),
            "state",
            "mark-published",
            "--run-root",
            str(run_root),
            "--video-id",
            "L01_V001",
            "--manifest",
            str(run_root / "staging" / "L01_V001" / "manifest.json"),
        ],
        [
            str(executable),
            "state",
            "cleanup",
            "--run-root",
            str(run_root),
            "--video-id",
            "L01_V001",
        ],
    ]


def test_wrapper_reports_native_stderr_and_rejects_blank_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep native diagnostics actionable without accepting a broad target path."""

    def fail_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Simulate a failed native predecessor check with captured stderr."""

        raise subprocess.CalledProcessError(1, argv, output="", stderr="wrong state")

    monkeypatch.setattr(subprocess, "run", fail_run)
    with pytest.raises(RuntimeError, match="wrong state"):
        cleanup_video(tmp_path / "keyframe_extractor", tmp_path / "run", "L01_V001")
    with pytest.raises(ValueError, match="video_id"):
        cleanup_video(tmp_path / "keyframe_extractor", tmp_path / "run", "  ")
