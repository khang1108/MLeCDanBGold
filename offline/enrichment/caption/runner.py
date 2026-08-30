"""Batch execution for independently materialized caption evidence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm.auto import tqdm

from hcmai.common.schemas import CaptionEvidence, ProcessingStatus
from hcmai.common.utils.image import load_image
from offline.enrichment.caption.artifacts import write_caption_artifacts
from offline.enrichment.caption.config import CaptionConfig
from offline.enrichment.caption.models.contracts import CaptionAdapter


def _load_frame_image(frame: dict[str, Any], config: CaptionConfig, root: Path) -> Any:
    """Load and thumbnail one frame's image, or return the raised exception."""

    try:
        path = Path(str(frame["image_path"])).expanduser()
        image = load_image(path if path.is_absolute() else root / path, mode="RGB")
        image.thumbnail((config.image_size, config.image_size))
        return image
    except Exception as error:  # noqa: BLE001 - surfaced as a per-frame failure row
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
    """Run caption batches while retaining a typed row for every frame.

    ``image_workers`` only parallelizes local disk image decoding/thumbnailing;
    it never changes image content, order, or the resulting caption rows.
    """

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
        if image_workers > 1 and len(chunk) > 1:
            with ThreadPoolExecutor(max_workers=image_workers) as pool:
                loaded = list(
                    pool.map(lambda frame: _load_frame_image(frame, config, root), chunk)
                )
        else:
            loaded = [_load_frame_image(frame, config, root) for frame in chunk]
        for frame, outcome in zip(chunk, loaded):
            frame_id = str(frame["frame_id"])
            if isinstance(outcome, Exception):
                rows[frame_id], failures[frame_id] = _failure(
                    frame,
                    config,
                    frame_store_id=frame_store_id,
                    resolved_revision=resolved_revision,
                    stage="image_load",
                    error_code=type(outcome).__name__,
                    error_message=str(outcome),
                )
            else:
                valid.append((frame, outcome))
        if valid:
            began = perf_counter()
            try:
                results = captioner.caption_batch([image for _, image in valid])
                if len(results) != len(valid):
                    raise ValueError("caption backend returned the wrong result count")
            except Exception as error:
                results = [error] * len(valid)
            latencies.append((perf_counter() - began) * 1000)
            for (frame, _), result in zip(valid, results):
                frame_id = str(frame["frame_id"])
                if isinstance(result, Exception):
                    rows[frame_id], failures[frame_id] = _failure(
                        frame,
                        config,
                        frame_store_id=frame_store_id,
                        resolved_revision=resolved_revision,
                        stage="model",
                        error_code=type(result).__name__,
                        error_message=str(result),
                    )
                elif result is None or not str(result).strip():
                    rows[frame_id], failures[frame_id] = _failure(
                        frame,
                        config,
                        frame_store_id=frame_store_id,
                        resolved_revision=resolved_revision,
                        stage="model",
                        error_code="EmptyCaption",
                        error_message="caption model returned empty text",
                    )
                else:
                    rows[frame_id] = CaptionEvidence(
                        frame_id=frame_id,
                        video_id=str(frame["video_id"]),
                        frame_idx=int(frame["frame_idx"]),
                        timestamp_ms=int(frame["timestamp_ms"]),
                        text=str(result).strip(),
                        frame_store_id=frame_store_id,
                        artifact_version=config.enrichment_version,
                        model_name=config.model_checkpoint,
                        model_revision=resolved_revision,
                        status=ProcessingStatus.COMPLETED,
                    )
        since_write += len(chunk)
        if since_write >= config.write_interval:
            write_caption_artifacts(
                output,
                order,
                rows,
                failures,
                checkpoint_manifest,
            )
            since_write = 0
        progress.update(len(chunk))
        progress.set_postfix(failed=len(failures), refresh=False)
    progress.close()
    return latencies
