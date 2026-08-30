"""Invoke native custom-extraction lifecycle transitions through explicit argv.

This module is a narrow Python convenience boundary. It validates operator
inputs and delegates all state mutation, predecessor checks, idempotency, and
filesystem cleanup to the C++ executable; it never reads or edits state JSON.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _require_non_blank(value: str | Path, *, field_name: str) -> str:
    """Return a non-blank CLI argument without normalizing its user-visible path.

    Args:
        value: Path-like or string command argument.
        field_name: Argument name used in the validation error.

    Returns:
        Original string representation after confirming it is non-blank.

    Raises:
        ValueError: If the argument is blank or whitespace-only.
    """

    rendered = str(value)
    if not rendered.strip():
        raise ValueError(f"{field_name} must not be blank")
    return rendered


def _run_native_state(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one native state command without shell interpolation.

    Args:
        argv: Complete executable-and-argument vector for the native command.

    Returns:
        Completed process result with captured text streams on success.

    Raises:
        RuntimeError: If the native executable exits non-zero, including its
            captured stderr (and stdout when stderr is empty) for diagnosis.
    """

    try:
        return subprocess.run(
            argv,
            check=True,
            shell=False,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        diagnostic = (error.stderr or error.stdout or "").strip()
        suffix = f": {diagnostic}" if diagnostic else ""
        raise RuntimeError(
            "native custom extraction state command failed"
            f" (exit_code={error.returncode}){suffix}"
        ) from error


def mark_video_enriched(
    native_executable: str | Path,
    run_root: str | Path,
    video_id: str,
    handoff_path: str | Path,
) -> subprocess.CompletedProcess[str]:
    """Ask the C++ runner to accept a validated per-video enrichment handoff.

    Args:
        native_executable: Built ``keyframe_extractor`` executable path.
        run_root: Native run root owning the target video state.
        video_id: Canonical selected video identifier.
        handoff_path: Validated compact JSON handoff path inside staging.

    Returns:
        Completed native process result after an accepted or idempotent command.

    Raises:
        ValueError: If any command identity or path argument is blank.
        RuntimeError: If the native predecessor/provenance command fails.
    """

    executable = _require_non_blank(native_executable, field_name="native_executable")
    root = _require_non_blank(run_root, field_name="run_root")
    identifier = _require_non_blank(video_id, field_name="video_id")
    handoff = _require_non_blank(handoff_path, field_name="handoff_path")
    return _run_native_state(
        [
            executable,
            "state",
            "mark-enriched",
            "--run-root",
            root,
            "--video-id",
            identifier,
            "--artifacts",
            handoff,
        ]
    )


def mark_video_published(
    native_executable: str | Path,
    run_root: str | Path,
    video_id: str,
    manifest_path: str | Path,
) -> subprocess.CompletedProcess[str]:
    """Ask the C++ runner to atomically publish an enriched staging bundle.

    Args:
        native_executable: Built ``keyframe_extractor`` executable path.
        run_root: Native run root owning the target video state.
        video_id: Canonical selected video identifier.
        manifest_path: Staging native manifest required by the native guard.

    Returns:
        Completed native process result after a published or idempotent command.

    Raises:
        ValueError: If any command identity or path argument is blank.
        RuntimeError: If native manifest/provenance checks reject publication.
    """

    executable = _require_non_blank(native_executable, field_name="native_executable")
    root = _require_non_blank(run_root, field_name="run_root")
    identifier = _require_non_blank(video_id, field_name="video_id")
    manifest = _require_non_blank(manifest_path, field_name="manifest_path")
    return _run_native_state(
        [
            executable,
            "state",
            "mark-published",
            "--run-root",
            root,
            "--video-id",
            identifier,
            "--manifest",
            manifest,
        ]
    )


def cleanup_video(
    native_executable: str | Path,
    run_root: str | Path,
    video_id: str,
) -> subprocess.CompletedProcess[str]:
    """Ask the C++ runner to remove only post-publication temporary artifacts.

    Args:
        native_executable: Built ``keyframe_extractor`` executable path.
        run_root: Native run root owning the target video state.
        video_id: Canonical selected video identifier.

    Returns:
        Completed native process result after a cleaned or idempotent command.

    Raises:
        ValueError: If any command identity argument is blank.
        RuntimeError: If native publication-state checks or cleanup fail.
    """

    executable = _require_non_blank(native_executable, field_name="native_executable")
    root = _require_non_blank(run_root, field_name="run_root")
    identifier = _require_non_blank(video_id, field_name="video_id")
    return _run_native_state(
        [
            executable,
            "state",
            "cleanup",
            "--run-root",
            root,
            "--video-id",
            identifier,
        ]
    )


__all__ = ["cleanup_video", "mark_video_enriched", "mark_video_published"]
