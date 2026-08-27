#!/usr/bin/env python3
"""Run custom video preparation from media metadata through local indexes.

This command coordinates the existing native extractor, specialist enrichment,
native publication, canonical FrameStore materialization, FrameContext, and
retrieval-index builders. It intentionally does not upload to S3. Model-heavy
stages remain separate subprocesses so one stage releases RAM/VRAM before the
next stage starts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Literal, Sequence

import pandas as pd

from hcmai.common.schemas import FrameRecord
from hcmai.common.utils.io import atomic_write, read_json, write_json, write_parquet
from hcmai.data.ingestion import (
    CustomFrameStoreConfig,
    cleanup_video,
    iter_native_frame_records,
    mark_video_enriched,
    mark_video_published,
    materialize_custom_frame_store,
    write_enrichment_handoff,
)
from scripts import extract_custom_keyframes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "prepare.yaml"
DEFAULT_APP_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"


def _positive_int(value: str) -> int:
    """Parse one strictly positive CLI integer."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than or equal to 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one explicit custom-corpus preparation contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--app-config", type=Path, default=DEFAULT_APP_CONFIG)
    parser.add_argument("--media-info-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--native-executable",
        type=Path,
        default=Path("build/keyframes_extraction/keyframe_extractor"),
    )
    parser.add_argument("--yt-dlp-binary", default="yt-dlp")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source", default="custom_raw_video_1fps")
    parser.add_argument("--frame-store-id", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--video-id", action="append")
    selection.add_argument("--limit", type=_positive_int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument(
        "--keep-temporary",
        action="store_true",
        help="Keep native source and OCR scratch files after a passed index build.",
    )
    return parser.parse_args(argv)


def _extraction_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the native extraction namespace without shell interpolation."""

    values = [
        "--media-info-dir",
        str(args.media_info_dir),
        "--run-root",
        str(args.run_root),
        "--native-executable",
        str(args.native_executable),
        "--frame-store-id",
        args.frame_store_id,
        "--yt-dlp-binary",
        args.yt_dlp_binary,
    ]
    if args.source_root is not None:
        values.extend(("--source-root", str(args.source_root)))
    if args.video_id:
        for video_id in args.video_id:
            values.extend(("--video-id", video_id))
    elif args.limit is not None:
        values.extend(("--limit", str(args.limit)))
    else:
        values.append("--all")
    if args.fail_fast:
        values.append("--fail-fast")
    return extract_custom_keyframes.parse_args(values)


def _run_python(script: str, arguments: Sequence[str]) -> None:
    """Run one model-heavy stage in an isolated Python process."""

    environment = os.environ.copy()
    current = environment.get("PYTHONPATH")
    repository_paths = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}"
    environment["PYTHONPATH"] = (
        f"{repository_paths}:{current}" if current else repository_paths
    )
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        shell=False,
    )


def _bundle_root(run_root: Path, video_id: str) -> Path:
    """Return one staging or published native bundle without guessing identity."""

    staging = run_root / "staging" / video_id
    if staging.is_dir():
        return staging
    published = run_root / "published" / video_id
    if published.is_dir():
        return published
    raise FileNotFoundError(f"native bundle is unavailable for {video_id}")


def _materialize_batch_frames(
    run_root: Path,
    video_ids: tuple[str, ...],
    output: Path,
    *,
    image_variant: Literal["durable", "enrichment"],
) -> Path:
    """Publish one deterministic temporary FrameStore over selected bundles."""

    records = [
        record
        for video_id in sorted(video_ids)
        for record in iter_native_frame_records(
            _bundle_root(run_root, video_id),
            run_root=run_root,
            image_variant=image_variant,
        )
    ]
    expected_ids = [record.frame_id for record in records]
    if output.is_file():
        existing = pd.read_parquet(output)
        if existing["frame_id"].astype(str).tolist() == expected_ids:
            return output
        raise ValueError("temporary FrameStore selection changed inside one run")

    table = pd.DataFrame(
        [record.model_dump(mode="python") for record in records],
        columns=list(FrameRecord.model_fields),
    )
    atomic_write(output, lambda path: write_parquet(table, path, index=False))
    return output


def _dataset_arguments(
    args: argparse.Namespace,
    frames_path: Path,
    frame_store_root: Path,
) -> list[str]:
    """Return the complete shared dataset CLI contract for one stage."""

    return [
        "--version",
        args.version,
        "--source",
        args.source,
        "--frame-store-id",
        args.frame_store_id,
        "--data-root",
        str(args.run_root),
        "--frames",
        str(frames_path),
        "--frame-store-output",
        str(frame_store_root),
    ]


def _stage_video_aliases(run_root: Path, video_ids: tuple[str, ...]) -> Path:
    """Expose extension-bearing hard links for transcript source discovery."""

    output = run_root / "pipeline" / "videos"
    output.mkdir(parents=True, exist_ok=True)
    for video_id in video_ids:
        source = run_root / "source" / f"{video_id}.part"
        destination = output / f"{video_id}.mp4"
        if destination.is_file() and source.is_file() and (
            destination.stat().st_size == source.stat().st_size
        ):
            continue
        if not source.is_file():
            raise FileNotFoundError(f"retained native source is unavailable: {source}")
        destination.unlink(missing_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return output


def _transcript_path(root: Path, video_id: str) -> Path:
    """Return the grouped transcript path owned by TranscriptService."""

    return root / video_id.split("_", maxsplit=1)[0] / f"{video_id}.parquet"


def _transcripts_complete(root: Path, video_ids: tuple[str, ...]) -> bool:
    """Return whether every selected video has a committed transcript pair."""

    return all(
        (path := _transcript_path(root, video_id)).is_file()
        and path.with_suffix(".manifest.json").is_file()
        for video_id in video_ids
    )


def _require_complete_frame_artifact(
    path: Path,
    expected_frame_ids: list[str],
    label: str,
) -> pd.DataFrame:
    """Reject a specialist artifact containing failed or missing frame rows."""

    table = pd.read_parquet(path)
    actual_ids = table["frame_id"].astype(str).tolist()
    if actual_ids != expected_frame_ids:
        raise ValueError(f"{label} artifact does not cover canonical frame order")
    if "status" not in table:
        raise ValueError(f"{label} artifact is missing processing status")
    if set(table["status"].astype(str)) != {"completed"}:
        raise RuntimeError(f"{label} artifact contains non-completed frame evidence")
    return table


def _write_video_projection(
    table: pd.DataFrame,
    frame_ids: list[str],
    output: Path,
) -> Path:
    """Write one exact per-video specialist projection for native handoff."""

    order = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    selected = table[table["frame_id"].astype(str).isin(order)].copy()
    selected["_order"] = selected["frame_id"].astype(str).map(order)
    selected = selected.sort_values("_order").drop(columns="_order")
    if selected["frame_id"].astype(str).tolist() != frame_ids:
        raise ValueError("specialist projection changed per-video frame identity")
    atomic_write(output, lambda path: write_parquet(selected, path, index=False))
    return output


def _completed_report(
    report_path: Path,
    args: argparse.Namespace,
    selected: tuple[str, ...],
) -> dict[str, Any] | None:
    """Return a complete matching run report, otherwise require normal resume."""

    if not report_path.is_file():
        return None
    value = read_json(report_path)
    if not isinstance(value, dict):
        return None
    identity = {
        "status": "passed",
        "dataset_version": args.version,
        "dataset_source": args.source,
        "frame_store_id": args.frame_store_id,
        "selected_video_ids": list(selected),
    }
    if any(value.get(name) != expected for name, expected in identity.items()):
        return None
    for raw_path in value.get("required_artifacts", []):
        if not Path(str(raw_path)).exists():
            return None
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute or resume every non-S3 custom preparation stage."""

    run_root = args.run_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    native = args.native_executable.expanduser()
    args.native_executable = (
        native if native.is_absolute() else PROJECT_ROOT / native
    ).resolve()
    args.config = args.config.expanduser().resolve()
    args.app_config = args.app_config.expanduser().resolve()
    args.run_root = run_root
    args.output_root = output_root
    extraction = extract_custom_keyframes.run(_extraction_args(args))
    if extraction["failed"]:
        raise RuntimeError("custom extraction contains failed videos")
    selected = tuple(sorted(str(value) for value in extraction["selected_video_ids"]))
    if not selected:
        raise ValueError("custom preparation requires at least one selected video")

    report_path = output_root / "prepare_report.json"
    completed = _completed_report(report_path, args, selected)
    if completed is not None:
        return completed

    pipeline_root = run_root / "pipeline"
    frame_store_root = output_root / "frame_store"
    enrichment_root = output_root / "enrichment"
    caption_root = enrichment_root / "captions"
    ocr_root = enrichment_root / "ocr"
    object_root = enrichment_root / "objects"
    transcript_root = enrichment_root / "transcripts"
    context_root = enrichment_root / "context"
    index_root = output_root / "indexes"
    durable_frames = _materialize_batch_frames(
        run_root,
        selected,
        pipeline_root / "durable_frames.parquet",
        image_variant="durable",
    )
    ocr_frames = _materialize_batch_frames(
        run_root,
        selected,
        pipeline_root / "ocr_frames.parquet",
        image_variant="enrichment",
    )
    expected_ids = pd.read_parquet(durable_frames)["frame_id"].astype(str).tolist()

    common = _dataset_arguments(args, durable_frames, frame_store_root)
    _run_python(
        "generate_enrichment.py",
        [
            "--config",
            str(args.config),
            "--app-config",
            str(args.app_config),
            "--output",
            str(caption_root),
            *common,
        ],
    )
    _run_python(
        "generate_ocr_enrichment.py",
        [
            "--config",
            str(args.config),
            "--app-config",
            str(args.app_config),
            "--output",
            str(ocr_root),
            *_dataset_arguments(args, ocr_frames, frame_store_root),
        ],
    )
    _run_python(
        "detect_objects.py",
        ["--config", str(args.config), "--output", str(object_root), *common],
    )

    if not _transcripts_complete(transcript_root, selected):
        videos_root = _stage_video_aliases(run_root, selected)
        transcript_arguments = [
            "--config",
            str(args.config),
            "--videos-root",
            str(videos_root),
            "--output",
            str(transcript_root),
            "--frame-enrichment-output",
            str(enrichment_root / "asr" / "frame_enrichment.parquet"),
            *common,
        ]
        if args.no_diarization:
            transcript_arguments.append("--no-diarization")
        _run_python("prepare_transcripts.py", transcript_arguments)

    caption_table = _require_complete_frame_artifact(
        caption_root / "captions.parquet", expected_ids, "Caption"
    )
    ocr_table = _require_complete_frame_artifact(
        ocr_root / "frames.parquet", expected_ids, "OCR"
    )
    object_table = _require_complete_frame_artifact(
        object_root / "frames.parquet", expected_ids, "Object"
    )

    for video_id in selected:
        bundle = _bundle_root(run_root, video_id)
        if bundle.parent.name == "published":
            continue
        records = list(iter_native_frame_records(bundle, run_root=run_root))
        frame_ids = [record.frame_id for record in records]
        handoff_root = bundle / "enrichment" / "handoff_artifacts"
        handoff_root.mkdir(parents=True, exist_ok=True)
        handoff = write_enrichment_handoff(
            bundle,
            artifact_paths={
                "caption": _write_video_projection(
                    caption_table, frame_ids, handoff_root / "caption.parquet"
                ),
                "ocr": _write_video_projection(
                    ocr_table, frame_ids, handoff_root / "ocr.parquet"
                ),
                "objects": _write_video_projection(
                    object_table, frame_ids, handoff_root / "objects.parquet"
                ),
                "asr": _transcript_path(transcript_root, video_id),
            },
            output_path=bundle / "enrichment" / "handoff.json",
            frame_store_id=args.frame_store_id,
        )
        mark_video_enriched(
            args.native_executable, run_root, video_id, handoff
        )
        mark_video_published(
            args.native_executable,
            run_root,
            video_id,
            bundle / "manifest.json",
        )

    frames_path = materialize_custom_frame_store(
        CustomFrameStoreConfig(
            run_root=run_root,
            output_root=frame_store_root,
            frame_store_id=args.frame_store_id,
            selected_video_ids=selected,
        )
    )
    canonical = pd.read_parquet(frames_path)
    frame_count = len(canonical)
    video_count = int(canonical["video_id"].nunique())
    final_common = _dataset_arguments(args, frames_path, frame_store_root)
    context_path = context_root / "frame_context_v1.parquet"
    _run_python(
        "build_frame_context.py",
        [
            "--config",
            str(args.config),
            "--captions",
            str(caption_root / "captions.parquet"),
            "--ocr-frames",
            str(ocr_root / "frames.parquet"),
            "--object-frames",
            str(object_root / "frames.parquet"),
            "--output",
            str(context_root),
            *final_common,
        ],
    )
    _run_python(
        "build_retrieval_indexes.py",
        [
            "--stage",
            "all",
            "--config",
            str(args.config),
            "--model-config",
            str(args.config),
            "--frame-manifest",
            str(frame_store_root / "manifest.json"),
            "--context",
            str(context_path),
            "--transcripts",
            str(transcript_root),
            "--expected-video-count",
            str(video_count),
            "--expected-frame-count",
            str(frame_count),
            "--output-root",
            str(index_root),
            *final_common,
        ],
    )

    build_report_path = index_root / "build_report.json"
    build_report = read_json(build_report_path)
    if not isinstance(build_report, dict) or build_report.get("status") != "passed":
        raise RuntimeError("index build did not publish a passed validation report")

    if not args.keep_temporary:
        for video_id in selected:
            cleanup_video(args.native_executable, run_root, video_id)
        shutil.rmtree(pipeline_root / "videos", ignore_errors=True)

    report = {
        "status": "passed",
        "dataset_version": args.version,
        "dataset_source": args.source,
        "frame_store_id": args.frame_store_id,
        "selected_video_ids": list(selected),
        "video_count": video_count,
        "frame_count": frame_count,
        "required_artifacts": [
            str(frames_path),
            str(caption_root / "captions.parquet"),
            str(ocr_root / "frames.parquet"),
            str(object_root / "frames.parquet"),
            str(context_path),
            str(build_report_path),
        ],
    }
    atomic_write(report_path, lambda path: write_json(report, path))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the custom preparation pipeline and print its final report."""

    report = run(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
