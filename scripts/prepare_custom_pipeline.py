#!/usr/bin/env python3
"""Run the local A6000/100GB custom corpus pipeline in resumable subcommands.

Subcommands:
  preflight        Validate local prerequisites and the requested archive
                    work window without downloading anything.
  process-archive  Resume every archive in the requested work window through
                    committed local batches (extraction, Caption, OCR,
                    Objects, FrameContext, visual/context embeddings, and the
                    three batch indexes).
  status           Report local archive/batch state and the recommended next
                    offset. Read-only.
  finalize         Compact every committed batch into the final corpus once
                    the complete frozen archive plan is cleaned.

This CLI has no video selector, yt-dlp option, or cloud destination option:
every archive URL in the ordered plan is processed in canonical groups of at
most eight videos (see docs/superpowers/plans/2026-08-28-a6000-100gb-custom-
pipeline.md), and publication to any remote store is an operator-owned,
out-of-band step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import Request, urlopen
import zipfile

import pandas as pd

from hcmai.common.config import EncoderConfig
from hcmai.common.utils.io import atomic_write, read_json, read_yaml_section, write_json, write_parquet
from offline.ingestion.custom_pipeline.asr import ASRReuseBundle, validate_asr_source
from offline.ingestion.custom_pipeline.config import (
    ArchivePlan,
    ArchiveWorkWindow,
    CustomPipelineConfig,
)
from offline.ingestion.custom_pipeline.contracts import RunIdentity
from offline.ingestion.custom_pipeline.runner import (
    BatchArtifacts,
    RunnerContext,
    finalize_pipeline,
    pipeline_status,
    preflight_pipeline,
    process_archive,
)
from offline.ingestion.custom_pipeline.state import PipelineStateStore, VideoStage
from offline.ingestion import (
    cleanup_video,
    iter_native_frame_records,
    mark_video_enriched,
    mark_video_published,
    write_enrichment_handoff,
)
from scripts import extract_custom_keyframes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "prepare.yaml"
DEFAULT_APP_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
DEFAULT_MEDIA_INFO_URL = "https://aic-data.ledo.io.vn/media-info-aic25-b1.zip"
MEDIA_INFO_ARCHIVE_NAME = "media-info-aic25-b1.zip"
MAX_MEDIA_INFO_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEDIA_INFO_EXTRACTED_BYTES = 256 * 1024 * 1024
MAX_MEDIA_INFO_MEMBERS = 50_000


# ---------------------------------------------------------------------------
# Media-info bootstrap (unchanged from the prior monolithic CLI: safe,
# atomic, and independent of archive/batch orchestration).
# ---------------------------------------------------------------------------


def _discover_media_info_dir(root: Path) -> Path:
    """Find the unique extracted directory containing media-info JSON files."""

    preferred = root / "media-info"
    if preferred.is_dir() and any(preferred.glob("*.json")):
        return preferred.resolve()
    if root.is_dir() and any(root.glob("*.json")):
        return root.resolve()

    candidates = sorted(
        {path.parent.resolve() for path in root.rglob("*.json") if path.is_file()},
        key=str,
    )
    if not candidates:
        raise ValueError(f"media-info archive contains no JSON files: {root}")
    if len(candidates) != 1:
        raise ValueError("media-info archive contains multiple JSON directories")
    return candidates[0]


def _download_media_info_archive(url: str, output: Path) -> None:
    """Download the bounded media-info ZIP through an atomic local file."""

    request = Request(url, headers={"User-Agent": "HCMAI/2026 media-info bootstrap"})

    def write_download(temporary: Path) -> None:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as target:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > MAX_MEDIA_INFO_ARCHIVE_BYTES:
                raise ValueError("media-info archive exceeds the download size limit")

            downloaded = 0
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > MAX_MEDIA_INFO_ARCHIVE_BYTES:
                    raise ValueError("media-info archive exceeds the download size limit")
                target.write(chunk)
            if downloaded == 0:
                raise ValueError("media-info archive download is empty")

    atomic_write(output, write_download)


def _safe_extract_media_info_archive(archive: Path, output: Path) -> Path:
    """Atomically extract a bounded ZIP without links or escaping members."""

    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if not members or len(members) > MAX_MEDIA_INFO_MEMBERS:
            raise ValueError("media-info archive has an invalid member count")
        extracted_bytes = sum(member.file_size for member in members)
        if extracted_bytes > MAX_MEDIA_INFO_EXTRACTED_BYTES:
            raise ValueError("media-info archive exceeds the extraction size limit")

        validated: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
        for member in members:
            normalized = member.filename.replace("\\", "/")
            relative = PurePosixPath(normalized)
            parts = tuple(part for part in relative.parts if part not in ("", "."))
            unix_mode = member.external_attr >> 16
            if relative.is_absolute() or not parts or ".." in parts or stat.S_ISLNK(unix_mode):
                raise ValueError(f"unsafe media-info ZIP member: {member.filename}")
            validated.append((member, parts))

        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".media-info-extract-", dir=output.parent) as temporary_value:
            temporary = Path(temporary_value) / "content"
            temporary.mkdir()
            for member, parts in validated:
                destination = temporary.joinpath(*parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as compressed, destination.open("wb") as target:
                    shutil.copyfileobj(compressed, target, length=1024 * 1024)

            media_info_dir = _discover_media_info_dir(temporary)
            relative_media_info = media_info_dir.relative_to(temporary)
            if output.exists():
                raise FileExistsError(f"invalid existing media-info extraction must be removed: {output}")
            temporary.replace(output)

    return (output / relative_media_info).resolve()


def _resolve_media_info_dir(media_info_dir: Path | None, media_info_url: str, run_root: Path) -> Path:
    """Return explicit metadata or bootstrap the default resumable archive."""

    if media_info_dir is not None:
        resolved = media_info_dir.expanduser().resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        if not any(resolved.glob("*.json")):
            raise ValueError(f"media-info directory contains no JSON files: {resolved}")
        return resolved

    input_root = run_root / "input"
    extraction_root = input_root / "media-info-aic25-b1"
    if extraction_root.is_dir():
        return _discover_media_info_dir(extraction_root)

    archive = input_root / MEDIA_INFO_ARCHIVE_NAME
    if not archive.is_file():
        _download_media_info_archive(media_info_url, archive)
    return _safe_extract_media_info_archive(archive, extraction_root)


# ---------------------------------------------------------------------------
# Isolated local specialist-stage subprocesses.
# ---------------------------------------------------------------------------


def _run_python(script: str, arguments: Sequence[str]) -> None:
    """Run one model-heavy stage in an isolated Python process."""

    import os

    environment = os.environ.copy()
    current = environment.get("PYTHONPATH")
    repository_paths = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}"
    environment["PYTHONPATH"] = f"{repository_paths}:{current}" if current else repository_paths
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


def _transcript_path(root: Path, video_id: str) -> Path:
    """Return the grouped transcript path owned by TranscriptService."""

    return root / video_id.split("_", maxsplit=1)[0] / f"{video_id}.parquet"


def _require_complete_frame_artifact(path: Path, expected_frame_ids: list[str], label: str) -> pd.DataFrame:
    """Reject a specialist artifact containing failed or missing frame rows."""

    table = pd.read_parquet(path)
    actual_ids = table["frame_id"].astype(str).tolist()
    if actual_ids != expected_frame_ids:
        raise ValueError(f"{label} artifact does not cover the batch's canonical frame order")
    if "status" not in table:
        raise ValueError(f"{label} artifact is missing processing status")
    if set(table["status"].astype(str)) != {"completed"}:
        raise RuntimeError(f"{label} artifact contains non-completed frame evidence")
    return table


def _write_video_projection(table: pd.DataFrame, frame_ids: list[str], output: Path) -> Path:
    """Write one exact per-video specialist projection for the native handoff."""

    order = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    selected = table[table["frame_id"].astype(str).isin(order)].copy()
    selected["_order"] = selected["frame_id"].astype(str).map(order)
    selected = selected.sort_values("_order").drop(columns="_order")
    if selected["frame_id"].astype(str).tolist() != frame_ids:
        raise ValueError("specialist projection changed per-video frame identity")
    atomic_write(output, lambda path: write_parquet(selected, path, index=False))
    return output


def _load_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one optional child parquet, or an empty table with the given columns."""

    if path.is_file():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=columns)


def _materialize_frames_for_videos(
    run_root: Path,
    video_ids: list[str],
    output: Path,
    *,
    image_variant: str,
) -> Path:
    """Publish a deterministic per-batch FrameStore slice over selected bundles."""

    from offline.ingestion.models import FrameArtifact

    records = [
        record
        for video_id in sorted(video_ids)
        for record in iter_native_frame_records(
            _bundle_root(run_root, video_id), run_root=run_root, image_variant=image_variant
        )
    ]
    table = pd.DataFrame(
        [record.model_dump(mode="python") for record in records],
        columns=list(FrameArtifact.model_fields),
    )
    atomic_write(output, lambda path: write_parquet(table, path, index=False))
    return output


def _dataset_arguments(args: argparse.Namespace, frames_path: Path, frame_store_root: Path) -> list[str]:
    """Return the shared dataset CLI contract used by every specialist stage."""

    return [
        "--version", args.version,
        "--source", args.source,
        "--frame-store-id", args.frame_store_id,
        "--data-root", str(args.run_root),
        "--frames", str(frames_path),
        "--frame-store-output", str(frame_store_root),
    ]


def _resolve_native_executable(args: argparse.Namespace) -> Path:
    """Resolve the native extractor executable relative to the repo root."""

    native = args.native_executable.expanduser()
    return (native if native.is_absolute() else PROJECT_ROOT / native).resolve()


def _load_encoder_config(config_path: Path, section: str) -> EncoderConfig:
    """Load one pinned encoder configuration from the shared models section."""

    models = read_yaml_section(config_path, "models")
    return EncoderConfig.from_dict(models[section])


def _encode_visual_vectors(image_paths: list[Path], config: EncoderConfig):
    """Encode durable images into L2-normalized visual vectors."""

    from hcmai.common.utils.image import load_image
    from hcmai.retrieval.embedding.adapters.siglip import SigLIPAdapter

    adapter = SigLIPAdapter(config)
    images = [load_image(path, mode="RGB") for path in image_paths]
    return adapter.encode_images(images)


def _encode_context_vectors(texts: list[str], config: EncoderConfig):
    """Encode FrameContext text into L2-normalized context vectors."""

    from hcmai.retrieval.embedding.adapters.bge import BGEAdapter

    adapter = BGEAdapter(config)
    return adapter.encode_text(texts)


# ---------------------------------------------------------------------------
# Batch identity (RunIdentity) helpers.
# ---------------------------------------------------------------------------


def _media_info_digest(media_info_dir: Path) -> str:
    """Hash every organizer media-info filename/size deterministically."""

    entries = sorted((path.name, path.stat().st_size) for path in media_info_dir.glob("*.json"))
    payload = json.dumps(entries, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_config_fingerprint(config_path: Path) -> str:
    """Hash the artifact-shaping part of the ``custom_pipeline`` config section.

    Scheduling and stage batch sizes are excluded: they change only how work is
    chunked across hosts, never the produced artifacts, so retuning them per
    GPU must not invalidate a resumable run.
    """

    raw = dict(read_yaml_section(config_path, "custom_pipeline"))
    for throughput_key in ("scheduling", "stage_batches"):
        raw.pop(throughput_key, None)
    payload = json.dumps(raw, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_revisions(config_path: Path) -> dict[str, str]:
    """Collect every pinned model revision that affects local artifact identity."""

    enrichment = read_yaml_section(config_path, "enrichment")
    models = read_yaml_section(config_path, "models")
    return {
        "caption": enrichment["caption"]["revision"],
        "ocr": enrichment["ocr"]["revision"],
        "asr": enrichment["transcript"]["asr"]["revision"],
        "visual_embedding": models["visual_embedding"]["revision"],
        "evidence_embedding": models["evidence_embedding"]["revision"],
    }


def _asr_lineage_digest(asr_index_root: Path) -> str:
    """Hash the persisted, checksum-validated ASR index's provenance metadata."""

    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

    index = SegmentDenseIndex.load(asr_index_root)
    payload = json.dumps(index.metadata.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_run_identity(args: argparse.Namespace, plan: ArchivePlan) -> RunIdentity:
    """Build the immutable identity that must match to resume local state."""

    return RunIdentity(
        version=args.version,
        source=args.source,
        frame_store_id=args.frame_store_id,
        media_info_digest=_media_info_digest(args.media_info_dir),
        archive_plan_digest=plan.digest,
        artifact_config_fingerprint=_artifact_config_fingerprint(args.config),
        model_revisions=_model_revisions(args.config),
        asr_lineage_digest=_asr_lineage_digest(args.asr_index_root),
    )


# ---------------------------------------------------------------------------
# The real per-batch specialist + embedding pipeline (produce_batch_artifacts).
# ---------------------------------------------------------------------------


def _make_produce_batch_artifacts(
    args: argparse.Namespace, state_store: PipelineStateStore
) -> Callable[[str, list[str], list[Path]], BatchArtifacts]:
    """Build the callback that runs real local stages for one batch.

    Runs native extraction, then Caption/OCR/Objects as isolated
    subprocesses, then FrameContext, then encodes visual/context vectors
    in-process. Advances each video's state from ``source_ready`` through
    ``embeddings_complete`` as each real stage completes, and performs the
    native per-video enriched/published/cleanup handoff so native scratch is
    reclaimed once a video's specialist evidence is durable.
    """

    run_root = args.run_root.expanduser().resolve()
    native_executable = _resolve_native_executable(args)
    visual_encoder_config = _load_encoder_config(args.config, "visual_embedding")
    context_encoder_config = _load_encoder_config(args.config, "evidence_embedding")

    def _produce(batch_id: str, video_ids: list[str], source_paths: list[Path]) -> BatchArtifacts:
        for video_id in video_ids:
            state_store.advance_video(video_id, VideoStage.SOURCE_READY)

        source_root = source_paths[0].parent
        extraction_argv = [
            "--media-info-dir", str(args.media_info_dir),
            "--run-root", str(run_root),
            "--native-executable", str(native_executable),
            "--frame-store-id", args.frame_store_id,
            # yt-dlp is never invoked here: --source-root always wins for
            # archive-sourced videos, but the native CLI still requires this flag.
            "--yt-dlp-binary", "yt-dlp",
            "--source-root", str(source_root),
            "--fail-fast",
        ]
        for video_id in video_ids:
            extraction_argv.extend(("--video-id", video_id))
        extraction = extract_custom_keyframes.run(extract_custom_keyframes.parse_args(extraction_argv))
        if extraction["failed"]:
            raise RuntimeError(f"native extraction failed for batch {batch_id}: {extraction.get('failures')}")

        for video_id in video_ids:
            state_store.advance_video(video_id, VideoStage.EXTRACTED)

        batch_root = run_root / "active" / "batch" / batch_id
        pipeline_root = batch_root / "pipeline"
        frame_store_root = batch_root / "frame_store"
        durable_frames = _materialize_frames_for_videos(
            run_root, video_ids, pipeline_root / "durable_frames.parquet", image_variant="durable"
        )
        ocr_frames_path = _materialize_frames_for_videos(
            run_root, video_ids, pipeline_root / "ocr_frames.parquet", image_variant="enrichment"
        )
        frame_ids = pd.read_parquet(durable_frames)["frame_id"].astype(str).tolist()

        caption_root = pipeline_root / "enrichment" / "captions"
        ocr_root = pipeline_root / "enrichment" / "ocr"
        object_root = pipeline_root / "enrichment" / "objects"
        context_root = pipeline_root / "enrichment" / "context"

        common = _dataset_arguments(args, durable_frames, frame_store_root)
        _run_python(
            "generate_enrichment.py",
            [
                "--config", str(args.config),
                "--app-config", str(args.app_config),
                "--output", str(caption_root),
                "--execution-backend", "local",
                *common,
            ],
        )
        for video_id in video_ids:
            state_store.advance_video(video_id, VideoStage.CAPTIONED)

        # NOTE: generate_ocr_enrichment.py does not yet expose a local
        # execution backend; this still requires a reachable inference
        # gateway (see --app-config) until that CLI is updated.
        _run_python(
            "generate_ocr_enrichment.py",
            [
                "--config", str(args.config),
                "--app-config", str(args.app_config),
                "--output", str(ocr_root),
                *_dataset_arguments(args, ocr_frames_path, frame_store_root),
            ],
        )
        for video_id in video_ids:
            state_store.advance_video(video_id, VideoStage.OCR_COMPLETE)

        _run_python(
            "detect_objects.py",
            ["--config", str(args.config), "--output", str(object_root), *common],
        )
        for video_id in video_ids:
            state_store.advance_video(video_id, VideoStage.OBJECTS_COMPLETE)

        caption_table = _require_complete_frame_artifact(caption_root / "captions.parquet", frame_ids, "Caption")
        ocr_table = _require_complete_frame_artifact(ocr_root / "frames.parquet", frame_ids, "OCR")
        object_table = _require_complete_frame_artifact(object_root / "frames.parquet", frame_ids, "Object")

        _run_python(
            "build_frame_context.py",
            [
                "--config", str(args.config),
                "--captions", str(caption_root / "captions.parquet"),
                "--ocr-frames", str(ocr_root / "frames.parquet"),
                "--object-frames", str(object_root / "frames.parquet"),
                "--output", str(context_root),
                *common,
            ],
        )
        context_table = _require_complete_frame_artifact(
            context_root / "frame_context_v1.parquet", frame_ids, "Context"
        )
        for video_id in video_ids:
            state_store.advance_video(video_id, VideoStage.CONTEXT_COMPLETE)

        durable_table = pd.read_parquet(durable_frames)
        image_paths = [
            path if (path := Path(str(value))).is_absolute() else run_root / path
            for value in durable_table["image_path"]
        ]
        visual_vectors = _encode_visual_vectors(image_paths, visual_encoder_config)
        context_texts = [
            str(value) if value is not None else "" for value in context_table["context_text"]
        ]
        context_vectors = _encode_context_vectors(context_texts, context_encoder_config)
        for video_id in video_ids:
            state_store.advance_video(video_id, VideoStage.EMBEDDINGS_COMPLETE)

        base = durable_table[["frame_id", "video_id", "frame_idx", "timestamp_ms"]].reset_index(drop=True)
        mapping = base.assign(embedding_index=range(len(base)))

        frame_native_tables = {
            "caption": caption_table.reset_index(drop=True),
            "ocr_frames": ocr_table.reset_index(drop=True),
            "object_frames": object_table.reset_index(drop=True),
            "context": context_table.reset_index(drop=True),
        }
        child_tables = {
            "ocr_regions": _load_or_empty(ocr_root / "regions.parquet", ["frame_id", "video_id"]),
            "object_detections": _load_or_empty(object_root / "detections.parquet", ["frame_id", "video_id"]),
        }

        # Native per-video handoff/publish/cleanup so native scratch is
        # reclaimed once a video's specialist evidence is durable.
        for video_id in video_ids:
            bundle = _bundle_root(run_root, video_id)
            if bundle.parent.name == "published":
                continue
            records = list(iter_native_frame_records(bundle, run_root=run_root))
            video_frame_ids = [record.frame_id for record in records]
            handoff_root = bundle / "enrichment" / "handoff_artifacts"
            handoff_root.mkdir(parents=True, exist_ok=True)
            handoff = write_enrichment_handoff(
                bundle,
                artifact_paths={
                    "caption": _write_video_projection(caption_table, video_frame_ids, handoff_root / "caption.parquet"),
                    "ocr": _write_video_projection(ocr_table, video_frame_ids, handoff_root / "ocr.parquet"),
                    "objects": _write_video_projection(object_table, video_frame_ids, handoff_root / "objects.parquet"),
                    "asr": _transcript_path(args.transcripts_root, video_id),
                },
                output_path=bundle / "enrichment" / "handoff.json",
                frame_store_id=args.frame_store_id,
            )
            mark_video_enriched(native_executable, run_root, video_id, handoff)
            mark_video_published(native_executable, run_root, video_id, bundle / "manifest.json")
            cleanup_video(native_executable, run_root, video_id)

        return BatchArtifacts(
            frames_table=base,
            frame_native_tables=frame_native_tables,
            child_tables=child_tables,
            visual_vectors=visual_vectors,
            visual_mapping=mapping,
            context_vectors=context_vectors,
            context_mapping=mapping,
        )

    return _produce


def _make_asr_bundle_factory(args: argparse.Namespace) -> Callable[[list[str]], ASRReuseBundle]:
    """Validate and return reusable ASR evidence for exactly one batch's videos."""

    evidence_encoder = _load_encoder_config(args.config, "evidence_embedding")

    def _factory(video_ids: list[str]) -> ASRReuseBundle:
        return validate_asr_source(
            args.transcripts_root,
            args.asr_index_root,
            video_ids,
            evidence_encoder=evidence_encoder,
        )

    return _factory


# ---------------------------------------------------------------------------
# CLI parsing.
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    """Parse one strictly positive CLI integer with an actionable error."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options common to every subcommand."""

    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--app-config", type=Path, default=DEFAULT_APP_CONFIG)
    parser.add_argument("--media-info-dir", type=Path)
    parser.add_argument("--media-info-url", default=DEFAULT_MEDIA_INFO_URL)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--native-executable", type=Path, default=Path("build/keyframes-extraction/keyframe_extractor")
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--source", default="custom_raw_video_1fps")
    parser.add_argument("--frame-store-id", required=True)
    parser.add_argument("--transcripts-root", type=Path, required=True)
    parser.add_argument("--asr-index-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional path to also write the JSON report.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument(
        "--archive-url",
        dest="archive_urls",
        action="append",
        required=True,
        help="One organizer ZIP URL; repeat in the runbook's exact fixed order.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-offset-gap", action="store_true")
    parser.add_argument("--batch-offset", type=int, default=0)
    parser.add_argument("--batch-limit", type=int, default=None)
    parser.add_argument(
        "--finalize-batch-chunk-size",
        type=_positive_int,
        default=16,
        help=(
            "Maximum committed batches read per finalization chunk "
            "(default: 16; use 32 on hosts with more RAM)."
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one explicit local pipeline subcommand."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("preflight", "Validate local prerequisites without downloading anything."),
        ("process-archive", "Resume every archive in the work window through committed batches."),
        ("status", "Report local pipeline state."),
        ("finalize", "Compact every committed batch once the full plan is cleaned."),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        _add_shared_arguments(subparser)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Subcommand handlers.
# ---------------------------------------------------------------------------


def _resolve_shared_paths(args: argparse.Namespace) -> None:
    """Resolve every path argument once, in place, before any subcommand runs."""

    args.run_root = args.run_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.app_config = args.app_config.expanduser().resolve()
    args.transcripts_root = args.transcripts_root.expanduser().resolve()
    args.asr_index_root = args.asr_index_root.expanduser().resolve()
    args.media_info_dir = _resolve_media_info_dir(args.media_info_dir, args.media_info_url, args.run_root)


def _build_runner_context(args: argparse.Namespace, custom_config: CustomPipelineConfig) -> RunnerContext:
    """Build the shared local runner context from resolved CLI arguments."""

    return RunnerContext(
        run_root=args.run_root,
        artifacts_root=args.output_root,
        native_executable=_resolve_native_executable(args),
        disk_budget=custom_config.disk,
        scheduling=custom_config.scheduling,
    )


def _cmd_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate local prerequisites and the requested work window."""

    custom_config = CustomPipelineConfig.from_yaml(args.config)
    context = _build_runner_context(args, custom_config)
    plan = ArchivePlan.from_urls(args.archive_urls)
    window = ArchiveWorkWindow(offset=args.offset, limit=args.limit)
    report = preflight_pipeline(context, plan, window)
    return {
        "command": "preflight",
        "ok": report.ok,
        "problems": list(report.problems),
        "native_executable_found": report.native_executable_found,
        "ffmpeg_found": report.ffmpeg_found,
        "curl_found": report.curl_found,
        "measured_cpus": report.measured_cpus,
        "free_bytes": report.disk.free_bytes,
        "active_bytes": report.disk.active_bytes,
        "archive_plan_size": report.archive_plan_size,
        "work_window": report.work_window,
    }


def _cmd_process_archive(args: argparse.Namespace) -> dict[str, Any]:
    """Resume every archive in the requested work window."""

    custom_config = CustomPipelineConfig.from_yaml(args.config)
    context = _build_runner_context(args, custom_config)
    plan = ArchivePlan.from_urls(args.archive_urls)
    window = ArchiveWorkWindow(offset=args.offset, limit=args.limit)

    state_store = PipelineStateStore(context.run_root)
    state_store.create_or_resume_run(
        _build_run_identity(args, plan), window, allow_offset_gap=args.allow_offset_gap
    )

    produce_batch_artifacts = _make_produce_batch_artifacts(args, state_store)
    asr_bundle_factory = _make_asr_bundle_factory(args)
    visual_encoder = _load_encoder_config(args.config, "visual_embedding")
    context_encoder = _load_encoder_config(args.config, "evidence_embedding")

    committed_batches: dict[str, list[str]] = {}
    for entry in window.select(plan):
        committed_batches[entry.archive_id] = process_archive(
            context,
            state_store,
            entry,
            produce_batch_artifacts,
            asr_bundle_factory,
            dataset_version=args.version,
            visual_model_name=visual_encoder.model_name,
            visual_model_revision=visual_encoder.revision,
            context_model_name=context_encoder.model_name,
            context_model_revision=context_encoder.revision,
            batch_offset=args.batch_offset,
            batch_limit=args.batch_limit,
        )
    return {"command": "process-archive", "committed_batches": committed_batches}


def _cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    """Report read-only local pipeline state."""

    custom_config = CustomPipelineConfig.from_yaml(args.config)
    context = _build_runner_context(args, custom_config)
    plan = ArchivePlan.from_urls(args.archive_urls)
    state_store = PipelineStateStore(context.run_root)
    status = pipeline_status(context, state_store, plan)
    return {"command": "status", **status}


def _cmd_finalize(args: argparse.Namespace) -> dict[str, Any]:
    """Compact every committed batch once the full plan is cleaned."""

    custom_config = CustomPipelineConfig.from_yaml(args.config)
    context = _build_runner_context(args, custom_config)
    plan = ArchivePlan.from_urls(args.archive_urls)
    state_store = PipelineStateStore(context.run_root)
    report = finalize_pipeline(
        context,
        state_store,
        plan,
        context.artifacts_root / "batches",
        args.run_root,
        context.artifacts_root,
        dataset_version=args.version,
        batch_chunk_size=args.finalize_batch_chunk_size,
    )
    return {"command": "finalize", **report}


_HANDLERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "preflight": _cmd_preflight,
    "process-archive": _cmd_process_archive,
    "status": _cmd_status,
    "finalize": _cmd_finalize,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one local pipeline subcommand and print its JSON report."""

    from hcmai.common.utils.logging import configure_logging

    args = parse_args(argv)
    configure_logging(args.log_level)
    _resolve_shared_paths(args)

    report = _HANDLERS[args.command](args)
    if args.report is not None:
        atomic_write(args.report, lambda path: write_json(report, path))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
