"""Caption generation and artifact serialization pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import math
from numbers import Integral
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Sequence, cast

import pandas as pd
from tqdm.auto import tqdm

from hcmai.common.config import AppConfig
from hcmai.common.utils.image import load_image
from hcmai.common.utils.io import (
    atomic_write,
    read_json,
    write_json,
    write_parquet,
)
from offline.enrichment.bundle import publish_staged_bundle
from offline.enrichment.dataset_cli import add_dataset_arguments, dataset_overrides
from offline.enrichment.models import FrameEnrichment, ProcessingStatus

from .adapter import VLLMCaptionAdapter
from .models import (
    CaptionAdapter,
    CaptionConfig,
    CaptionEvidence,
    CaptionJobConfig,
    DEFAULT_ENRICHMENT_CONFIG,
    ENRICHMENT_VERSION,
)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def _null_scalar(value: object) -> bool:
    return value is None or value is pd.NA or (
        isinstance(value, float) and math.isnan(value)
    )


def valid_caption(
    data: dict[str, Any],
    *,
    artifact_version: str,
    model_name: str,
    frame_store_id: str | None,
) -> CaptionEvidence | None:
    """Return a reusable completed CaptionEvidence row, if valid."""
    try:
        if (
            not isinstance(data.get("frame_id"), str)
            or not data["frame_id"]
            or data["frame_id"].strip() != data["frame_id"]
            or not isinstance(data.get("video_id"), str)
            or not data["video_id"]
            or data["video_id"].strip() != data["video_id"]
            or isinstance(data.get("frame_idx"), bool)
            or not isinstance(data.get("frame_idx"), Integral)
            or isinstance(data.get("timestamp_ms"), bool)
            or not isinstance(data.get("timestamp_ms"), Integral)
        ):
            return None
        values = dict(data)
        for field in (
            "text",
            "frame_store_id",
            "model_revision",
            "error_code",
            "error_message",
        ):
            if _null_scalar(values.get(field)):
                values[field] = None
        row = CaptionEvidence.model_validate(values)
    except Exception:
        return None

    reusable = (
        row.status == ProcessingStatus.COMPLETED
        and row.artifact_version == artifact_version
        and row.model_name == model_name
        and (frame_store_id is None or row.frame_store_id == frame_store_id)
        and bool((row.text or "").strip())
    )
    return row if reusable else None


def _legacy_projection(row: CaptionEvidence) -> FrameEnrichment:
    return FrameEnrichment(
        frame_id=row.frame_id,
        frame_store_id=row.frame_store_id,
        caption=row.text if row.status == ProcessingStatus.COMPLETED else None,
        enrichment_version=row.artifact_version,
        objects=[],
        model_name=row.model_name,
        status=row.status,
        error_message=row.error_message,
    )


def write_caption_artifacts(
    output: Path,
    order: list[str],
    rows: dict[str, CaptionEvidence],
    failures: dict[str, dict[str, str]],
    manifest: dict[str, Any],
) -> None:
    """Stage, validate, and publish the complete Caption bundle."""
    missing_rows = [frame_id for frame_id in order if frame_id not in rows]
    if missing_rows:
        raise ValueError(
            "caption rows do not cover canonical order: "
            + ", ".join(missing_rows[:5])
        )
    frame_table = pd.DataFrame(
        [rows[frame_id].model_dump(mode="json") for frame_id in order],
        columns=list(CaptionEvidence.model_fields),
    )
    projection_table = pd.DataFrame(
        [
            _legacy_projection(rows[frame_id]).model_dump(mode="json")
            for frame_id in order
        ],
        columns=list(FrameEnrichment.model_fields),
    )
    failure_rows = [
        failures[frame_id] for frame_id in order if frame_id in failures
    ]

    output.mkdir(parents=True, exist_ok=True)
    published = (
        output / "captions.parquet",
        output / "failures.json",
        output / "frame_enrichment.parquet",
        output / "manifest.json",
    )
    staged = tuple(
        path.with_name(f".{path.name}.staged") for path in published
    )
    try:
        atomic_write(
            staged[0],
            lambda path: write_parquet(frame_table, path, index=False),
        )
        atomic_write(staged[1], lambda path: write_json(failure_rows, path))
        atomic_write(
            staged[2],
            lambda path: write_parquet(projection_table, path, index=False),
        )
        atomic_write(staged[3], lambda path: write_json(manifest, path))
        publish_staged_bundle(staged, published)
    finally:
        for path in staged:
            if path.exists():
                path.unlink()


def build_manifest(
    config: CaptionConfig,
    frames_path: Path,
    root: Path,
    rows: dict[str, CaptionEvidence],
    captioner: CaptionAdapter,
    started: datetime,
    elapsed: float,
    latencies: list[float],
    skipped: int,
    retried: int,
    *,
    frame_store_id: str | None = None,
) -> dict[str, Any]:
    complete = sum(
        row.status == ProcessingStatus.COMPLETED for row in rows.values()
    )
    ordered = sorted(latencies)

    def percentile(part: float) -> float:
        return ordered[round((len(ordered) - 1) * part)] if ordered else 0.0

    return {
        "artifact_version": config.enrichment_version,
        "source_artifact": "captions.parquet",
        ENRICHMENT_VERSION: config.enrichment_version,
        "dataset_version": config.dataset_version,
        "input_parquet_path": str(frames_path),
        "dataset_root": str(root),
        "frame_store_id": frame_store_id,
        "model_checkpoint": config.model_checkpoint,
        "resolved_model_revision": captioner.resolved_revision,
        "prompt": config.prompt,
        "decoding": config.decoding,
        "device": config.device,
        "precision": config.precision,
        "dtype": config.dtype,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "input_frame_count": len(rows),
        "completed_count": complete,
        "failed_count": len(rows) - complete,
        "skipped_count": skipped,
        "retried_count": retried,
        "start_time": started.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "elapsed_time_sec": elapsed,
        "throughput_images_per_sec": (complete - skipped) / elapsed if elapsed else 0.0,
        "batch_latency_ms_p50": percentile(0.5),
        "batch_latency_ms_p95": percentile(0.95),
        "effective_configuration": asdict(config),
        "git_commit": _git_commit(),
    }


def guard_resume(
    path: Path,
    old: dict[str, Any],
    config: CaptionConfig,
    root: Path,
    resolved_revision: str | None = None,
    frame_store_id: str | None = None,
) -> None:
    if not path.exists():
        return
    if not old:
        raise ValueError("Cannot safely resume: manifest.json is missing")
    if old.get(ENRICHMENT_VERSION) != config.enrichment_version:
        return
    previous, current = old.get("effective_configuration"), asdict(config)
    if not isinstance(previous, dict):
        raise ValueError("Cannot safely resume: effective configuration is missing")
    throughput_only = {"batch_size", "write_interval"}
    changed = [
        key
        for key, value in current.items()
        if key not in throughput_only and previous.get(key) != value
    ]
    if old.get("dataset_root") != str(root):
        changed.append("dataset_root")
    if resolved_revision is not None and old.get("resolved_model_revision") != resolved_revision:
        changed.append("resolved_model_revision")
    if frame_store_id is not None and old.get("frame_store_id") != frame_store_id:
        changed.append("frame_store_id")
    if changed:
        fields = ", ".join(sorted(set(changed)))
        raise ValueError(
            f"Cannot resume {config.enrichment_version!r}: changed {fields}; "
            "use a new enrichment_version or output directory"
        )


def resume_rows(
    frames: list[dict[str, Any]],
    path: Path,
    config: CaptionConfig,
    frame_store_id: str | None = None,
) -> tuple[dict[str, CaptionEvidence], list[dict[str, Any]], int, int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    if path.exists():
        try:
            prior = cast(
                list[dict[str, Any]], pd.read_parquet(path).to_dict(orient="records")
            )
        except Exception as error:
            message = str(error).strip()[:200] or type(error).__name__
            raise RuntimeError(f"Cannot resume corrupted Parquet {path}: {message}") from error
        for data in prior:
            groups.setdefault(str(data.get("frame_id")), []).append(data)

    rows: dict[str, CaptionEvidence] = {}
    todo: list[dict[str, Any]] = []
    skipped = retried = 0
    for frame in frames:
        frame_id = str(frame["frame_id"])
        old = groups.get(frame_id, [])
        row = (
            valid_caption(
                old[0],
                artifact_version=config.enrichment_version,
                model_name=config.model_checkpoint,
                frame_store_id=frame_store_id,
            )
            if len(old) == 1
            else None
        )
        if row is not None and (
            row.video_id != str(frame["video_id"])
            or row.frame_idx != int(frame["frame_idx"])
            or row.timestamp_ms != int(frame["timestamp_ms"])
        ):
            row = None
        if row is not None:
            rows[frame_id] = row
            skipped += 1
            continue
        retried += int(bool(old))
        rows[frame_id] = CaptionEvidence(
            frame_id=frame_id,
            video_id=str(frame["video_id"]),
            frame_idx=int(frame["frame_idx"]),
            timestamp_ms=int(frame["timestamp_ms"]),
            frame_store_id=frame_store_id,
            artifact_version=config.enrichment_version,
            model_name=config.model_checkpoint,
            status=ProcessingStatus.PENDING,
        )
        todo.append(frame)
    return rows, todo, skipped, retried


def _load_frame_image(frame: dict[str, Any], config: CaptionConfig, root: Path) -> Any:
    try:
        path = Path(str(frame["image_path"])).expanduser()
        image = load_image(path if path.is_absolute() else root / path, mode="RGB")
        image.thumbnail((config.image_size, config.image_size))
        return image
    except Exception as error:  # noqa: BLE001
        return error


def _failure(
    frame: dict[str, Any],
    config: CaptionConfig,
    *,
    frame_store_id: str | None,
    resolved_revision: str | None,
    stage: str,
    error_code: str,
    error_message: str,
) -> tuple[CaptionEvidence, dict[str, str]]:
    message = error_message.strip()[:300] or error_code
    code = error_code[:100]
    row = CaptionEvidence(
        frame_id=str(frame["frame_id"]),
        video_id=str(frame["video_id"]),
        frame_idx=int(frame["frame_idx"]),
        timestamp_ms=int(frame["timestamp_ms"]),
        frame_store_id=frame_store_id,
        artifact_version=config.enrichment_version,
        model_name=config.model_checkpoint,
        model_revision=resolved_revision,
        status=ProcessingStatus.FAILED,
        error_code=code,
        error_message=message,
    )
    detail = {
        "frame_id": row.frame_id,
        "artifact_version": config.enrichment_version,
        "processing_stage": stage,
        "exception_category": code,
        "error_code": code,
        "error_message": message,
    }
    return row, detail


def run_batches(
    todo: list[dict[str, Any]],
    order: list[str],
    rows: dict[str, CaptionEvidence],
    failures: dict[str, dict[str, str]],
    captioner: CaptionAdapter,
    config: CaptionConfig,
    output: Path,
    root: Path,
    checkpoint_manifest: dict[str, Any],
    *,
    frame_store_id: str | None,
    resolved_revision: str | None,
    image_workers: int = 1,
) -> list[float]:
    latencies: list[float] = []
    since_write = 0
    progress = tqdm(
        total=len(order),
        initial=len(order) - len(todo),
        desc="Generating captions",
        unit="frame",
        dynamic_ncols=True,
    )
    for start in range(0, len(todo), config.batch_size):
        chunk = todo[start : start + config.batch_size]
        valid: list[tuple[dict[str, Any], Any]] = []

        if image_workers > 1:
            with ThreadPoolExecutor(max_workers=image_workers) as executor:
                loaded = list(
                    executor.map(
                        lambda frame: _load_frame_image(frame, config, root),
                        chunk,
                    )
                )
        else:
            loaded = [_load_frame_image(frame, config, root) for frame in chunk]

        for frame, result in zip(chunk, loaded):
            if isinstance(result, Exception):
                row, detail = _failure(
                    frame,
                    config,
                    frame_store_id=frame_store_id,
                    resolved_revision=resolved_revision,
                    stage="image_loading",
                    error_code="corrupted_image",
                    error_message=str(result),
                )
                rows[row.frame_id] = row
                failures[row.frame_id] = detail
            else:
                valid.append((frame, result))

        if valid:
            began = perf_counter()
            try:
                captions = captioner.caption_batch([img for _, img in valid])
                elapsed_ms = (perf_counter() - began) * 1_000
                latencies.append(elapsed_ms)
                for (frame, _), text in zip(valid, captions):
                    rows[str(frame["frame_id"])] = CaptionEvidence(
                        frame_id=str(frame["frame_id"]),
                        video_id=str(frame["video_id"]),
                        frame_idx=int(frame["frame_idx"]),
                        timestamp_ms=int(frame["timestamp_ms"]),
                        frame_store_id=frame_store_id,
                        artifact_version=config.enrichment_version,
                        model_name=config.model_checkpoint,
                        model_revision=resolved_revision,
                        status=ProcessingStatus.COMPLETED,
                        text=str(text),
                    )
            except Exception as error:  # noqa: BLE001
                for frame, _ in valid:
                    row, detail = _failure(
                        frame,
                        config,
                        frame_store_id=frame_store_id,
                        resolved_revision=resolved_revision,
                        stage="inference",
                        error_code="inference_error",
                        error_message=str(error),
                    )
                    rows[row.frame_id] = row
                    failures[row.frame_id] = detail

        progress.update(len(chunk))
        progress.set_postfix({"failed": len(failures)})
        since_write += len(chunk)
        if since_write >= config.write_interval:
            write_caption_artifacts(output, order, rows, failures, checkpoint_manifest)
            since_write = 0

    progress.close()
    return latencies


def generate_captions(
    frames_path: str | Path,
    output_dir: str | Path,
    config: CaptionConfig,
    captioner: CaptionAdapter | None = None,
    *,
    dataset_root: str | Path = ".",
    frame_store_id: str | None = None,
    image_workers: int = 1,
) -> dict[str, Any]:
    from offline.artifact_readers import FrameArtifactReader

    started, began, frames_path = datetime.now(timezone.utc), perf_counter(), Path(frames_path)
    root = Path(dataset_root).expanduser().resolve()

    frames = [asdict(frame) for frame in FrameArtifactReader.load(frames_path).iter_frames()]
    order = [str(frame["frame_id"]) for frame in frames]

    if len(order) != len(set(order)):
        raise ValueError("input frames contain duplicate frame_id values")

    output = Path(output_dir)
    captioner = captioner or VLLMCaptionAdapter(config)
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "manifest.json"
    captions_path = output / "captions.parquet"
    old = read_json(manifest_path) if manifest_path.exists() else {}

    guard_resume(captions_path, old, config, root, frame_store_id=frame_store_id)
    rows, todo, skipped, retried = resume_rows(frames, captions_path, config, frame_store_id)
    resolved_revision = captioner.resolve_revision()
    guard_resume(
        captions_path,
        old,
        config,
        root,
        resolved_revision,
        frame_store_id=frame_store_id,
    )

    failures: dict[str, dict[str, str]] = {}
    checkpoint_manifest = {
        "status": "in_progress",
        "started_at": started.isoformat(),
        "total_frames": len(order),
        "frame_store_id": frame_store_id,
        "model_checkpoint": config.model_checkpoint,
        "resolved_model_revision": resolved_revision,
    }

    latencies = run_batches(
        todo,
        order,
        rows,
        failures,
        captioner,
        config,
        output,
        root,
        checkpoint_manifest,
        frame_store_id=frame_store_id,
        resolved_revision=resolved_revision,
        image_workers=image_workers,
    )
    manifest = build_manifest(
        config,
        frames_path,
        root,
        rows,
        captioner,
        started,
        perf_counter() - began,
        latencies,
        skipped,
        retried,
        frame_store_id=frame_store_id,
    )
    write_caption_artifacts(output, order, rows, failures, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_ENRICHMENT_CONFIG)
    parser.add_argument("--app-config", default="configs/baseline.yaml")
    add_dataset_arguments(parser)
    parser.add_argument("--output")
    parser.add_argument(
        "--execution-backend",
        choices=("vllm", "local"),
        default="vllm",
        help="Run captioning via standalone vLLM serving ('vllm', default).",
    )
    parser.add_argument(
        "--vllm-url",
        default=None,
        help="Custom base URL for vLLM VLM serving (defaults to HCMAI_VLM_CAPTION_URL or http://localhost:8001/v1).",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.image_workers < 1:
        parser.error("--image-workers must be positive")

    dataset = dataset_overrides(args)
    job = (
        CaptionJobConfig.from_yaml(args.config, dataset=dataset)
        if dataset is not None
        else CaptionJobConfig.from_yaml(args.config)
    )
    caption_config = (
        replace(job.caption, batch_size=args.batch_size)
        if args.batch_size is not None
        else job.caption
    )

    if args.execution_backend == "vllm":
        captioner = VLLMCaptionAdapter(caption_config, base_url=args.vllm_url)
        manifest = generate_captions(
            args.frames or job.frames_path,
            args.output or job.output_dir,
            caption_config,
            captioner,
            dataset_root=args.data_root or job.dataset_root,
            frame_store_id=job.frame_store_id,
            image_workers=args.image_workers,
        )
        keys = "completed_count", "failed_count", "skipped_count", "retried_count"
        print({key: manifest[key] for key in keys})
        return 0

    raise NotImplementedError("Only --execution-backend vllm is supported.")


if __name__ == "__main__":
    raise SystemExit(main())