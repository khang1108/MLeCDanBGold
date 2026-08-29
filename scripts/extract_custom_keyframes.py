#!/usr/bin/env python3
"""Download and extract a bounded custom-keyframe batch from media-info JSON.

This command composes the existing metadata and native extraction contracts. It
does not implement downloading or decoding itself: the native extractor invokes
yt-dlp without a shell, retains each source for ASR, and writes resumable
per-video state. Successful staging bundles are validated and projected into
durable and OCR-specific FrameRecord tables for downstream enrichment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

from tqdm.auto import tqdm

from hcmai.data.ingestion import (
    build_native_input_manifest,
    materialize_video_enrichment_frames,
    validate_native_video_bundle,
    write_extraction_config,
)


def _positive_int(value: str) -> int:
    """Parse a strictly positive batch limit for argparse."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse paths, lineage, and an explicit bounded video selection.

    Args:
        argv: Optional argument sequence for tests; ``None`` reads process args.

    Returns:
        Validated argparse namespace. Exactly one selection mode is required.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--media-info-dir",
        type=Path,
        required=True,
        help="Directory containing organizer {video_id}.json metadata files",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Isolated root for input, source, state, staging, and publication",
    )
    parser.add_argument(
        "--native-executable",
        type=Path,
        default=Path("build/keyframes_extraction/keyframe_extractor"),
    )
    parser.add_argument("--frame-store-id", default="custom-raw1fps-v1")
    parser.add_argument("--yt-dlp-binary", default="yt-dlp")
    parser.add_argument(
        "--yt-dlp-cookies",
        type=Path,
        help="Netscape-format cookie file used for authenticated downloads.",
    )
    parser.add_argument(
        "--yt-dlp-js-runtime",
        help="yt-dlp JavaScript runtime token, for example deno or node.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Use existing {video_id}.mp4 files instead of yt-dlp; intended for "
            "offline pilots and deterministic tests"
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--video-id",
        action="append",
        help="Extract this video ID; repeat for an explicit batch",
    )
    selection.add_argument(
        "--limit",
        type=_positive_int,
        help="Extract the first N manifest videos in deterministic order",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Explicitly authorize every video in the metadata folder",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the selected batch after the first failed video",
    )
    return parser.parse_args(argv)


def _resolve_executable(value: str | Path, *, label: str) -> Path:
    """Resolve an executable path or command name before any source mutation."""

    raw = str(value).strip()
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        resolved = candidate.resolve()
    else:
        discovered = shutil.which(raw)
        if discovered is None:
            raise FileNotFoundError(f"{label} executable is unavailable: {raw}")
        resolved = Path(discovered).resolve()
    if not os.access(resolved, os.X_OK):
        raise PermissionError(f"{label} is not executable: {resolved}")
    return resolved


def _manifest_video_ids(manifest_path: Path) -> tuple[str, ...]:
    """Read canonical video IDs from the just-generated native manifest."""

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    video_ids = tuple(str(row["video_id"]) for row in rows)
    if not video_ids or len(video_ids) != len(set(video_ids)):
        raise ValueError("generated extraction manifest has invalid video identities")
    return video_ids


def _select_video_ids(
    manifest_video_ids: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[str, ...]:
    """Apply an explicit selection without changing manifest ordering."""

    if args.video_id:
        requested = tuple(args.video_id)
        if len(requested) != len(set(requested)):
            raise ValueError("--video-id values must be unique")
        unknown = sorted(set(requested).difference(manifest_video_ids))
        if unknown:
            raise ValueError(
                "requested video IDs are absent from metadata: " + ", ".join(unknown)
            )
        requested_set = set(requested)
        return tuple(
            video_id
            for video_id in manifest_video_ids
            if video_id in requested_set
        )
    if args.limit is not None:
        return manifest_video_ids[: args.limit]
    if args.all:
        return manifest_video_ids
    raise ValueError("one extraction selection mode is required")


def _native_summary(result: subprocess.CompletedProcess[str]) -> dict[str, int]:
    """Parse and validate the native extractor's machine-readable summary."""

    try:
        value: Any = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("native extractor returned invalid JSON output") from error
    fields = ("completed", "failed", "skipped", "pending", "emitted_frame_count")
    if not isinstance(value, dict) or any(
        isinstance(value.get(field), bool)
        or not isinstance(value.get(field), int)
        or int(value[field]) < 0
        for field in fields
    ):
        raise RuntimeError("native extractor returned an invalid summary contract")
    return {field: int(value[field]) for field in fields}


def _native_failure_diagnostic(
    result: subprocess.CompletedProcess[str],
    run_root: Path,
    video_id: str,
) -> str:
    """Recover a native root cause from its durable failed state, else stderr."""

    state_path = run_root / "state" / f"{video_id}.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        state = None
    if isinstance(state, dict):
        diagnostic = state.get("error")
        if isinstance(diagnostic, str) and diagnostic.strip():
            return diagnostic.strip()
    return result.stderr.strip() or "native extraction failed"


def _prepare_enrichment_tables(run_root: Path, video_id: str) -> dict[str, str]:
    """Validate one staging bundle and write its two image-variant tables."""

    staging_bundle = run_root / "staging" / video_id
    validate_native_video_bundle(
        staging_bundle,
        run_root=run_root,
        expected_status="enrichment_pending",
    )
    enrichment_root = staging_bundle / "enrichment"
    durable_path = materialize_video_enrichment_frames(
        staging_bundle,
        enrichment_root / "durable_frames.parquet",
        image_variant="durable",
    )
    ocr_path = materialize_video_enrichment_frames(
        staging_bundle,
        enrichment_root / "ocr_frames.parquet",
        image_variant="enrichment",
    )
    return {
        "video_id": video_id,
        "durable_frames_path": str(durable_path),
        "ocr_frames_path": str(ocr_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare metadata, run bounded native extraction, and return readiness.

    Args:
        args: Validated extraction command arguments.

    Returns:
        JSON-safe extraction and enrichment-input summary.
    """
    native_executable = _resolve_executable(
        args.native_executable,
        label="native extractor",
    )
    source_root = args.source_root.expanduser().resolve() if args.source_root else None
    if source_root is not None and not source_root.is_dir():
        raise NotADirectoryError(source_root)
    yt_dlp_binary = (
        str(_resolve_executable(args.yt_dlp_binary, label="yt-dlp"))
        if source_root is None
        else str(args.yt_dlp_binary)
    )
    yt_dlp_cookies = (
        args.yt_dlp_cookies.expanduser().resolve()
        if args.yt_dlp_cookies is not None
        else None
    )
    if yt_dlp_cookies is not None and not yt_dlp_cookies.is_file():
        raise FileNotFoundError(f"yt-dlp cookie file is unavailable: {yt_dlp_cookies}")
    yt_dlp_js_runtime = (
        args.yt_dlp_js_runtime.strip()
        if args.yt_dlp_js_runtime is not None
        else None
    )
    if args.yt_dlp_js_runtime is not None and not yt_dlp_js_runtime:
        raise ValueError("--yt-dlp-js-runtime must not be blank")

    run_root = args.run_root.expanduser().resolve()
    input_root = run_root / "input"
    manifest_path = build_native_input_manifest(
        args.media_info_dir,
        input_root / "media_manifest.jsonl",
    )
    config_path = write_extraction_config(
        input_root / "extraction_config.json",
        run_root=run_root,
        native_executable=native_executable,
        frame_store_id=args.frame_store_id,
        yt_dlp_binary=yt_dlp_binary,
        yt_dlp_cookies_path=yt_dlp_cookies,
        yt_dlp_js_runtime=yt_dlp_js_runtime,
    )
    expected_version = str(
        json.loads(config_path.read_text(encoding="utf-8"))["extractor_version"]
    )
    version = subprocess.run(
        [str(native_executable), "--version"],
        check=True,
        shell=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != expected_version:
        raise RuntimeError(
            f"native extractor version mismatch: expected {expected_version}, got {version}"
        )

    selected_video_ids = _select_video_ids(
        _manifest_video_ids(manifest_path),
        args,
    )
    totals = {
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "emitted_frame_count": 0,
    }
    enrichment_inputs: list[dict[str, str]] = []
    published_video_ids: list[str] = []
    failures: list[dict[str, str]] = []

    for video_id in tqdm(
        selected_video_ids, desc="Extracting keyframes", unit="video", dynamic_ncols=True
    ):
        command = [
            str(native_executable),
            "extract",
            "--manifest",
            str(manifest_path),
            "--run-root",
            str(run_root),
            "--config",
            str(config_path),
            "--video-id",
            video_id,
            "--fail-fast",
        ]
        if source_root is not None:
            command.extend(("--source-root", str(source_root)))
        result = subprocess.run(
            command,
            check=False,
            shell=False,
            capture_output=True,
            text=True,
        )
        try:
            summary = _native_summary(result)
            if result.returncode != 0 or summary["failed"]:
                diagnostic = _native_failure_diagnostic(
                    result,
                    run_root,
                    video_id,
                )
                raise RuntimeError(diagnostic)

            for field in ("completed", "skipped", "emitted_frame_count"):
                totals[field] += summary[field]
            staging_bundle = run_root / "staging" / video_id
            published_bundle = run_root / "published" / video_id
            if staging_bundle.is_dir():
                enrichment_inputs.append(
                    _prepare_enrichment_tables(run_root, video_id)
                )
            elif published_bundle.is_dir():
                validate_native_video_bundle(
                    published_bundle,
                    run_root=run_root,
                    expected_status="published",
                )
                published_video_ids.append(video_id)
            else:
                raise RuntimeError(
                    "native extraction succeeded without a staging or published bundle"
                )
        except Exception as error:  # noqa: BLE001 - aggregate per-video failures
            totals["failed"] += 1
            failures.append({"video_id": video_id, "error": str(error)})
            if args.fail_fast:
                break

    output = {
        **totals,
        "selected_video_ids": list(selected_video_ids),
        "enrichment_ready_video_ids": [
            item["video_id"] for item in enrichment_inputs
        ],
        "published_video_ids": published_video_ids,
        "enrichment_inputs": enrichment_inputs,
        "failures": failures,
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
        "run_root": str(run_root),
    }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Run extraction, print its machine-readable summary, and map failures."""

    output = run(parse_args(argv))
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if output["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
