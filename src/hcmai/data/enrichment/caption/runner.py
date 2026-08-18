"""Batch execution for independently materialized caption evidence."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm.auto import tqdm

from hcmai.common.schemas import CaptionEvidence, ProcessingStatus
from hcmai.common.utils.image import load_image
from hcmai.data.enrichment.caption.artifacts import write_caption_artifacts
from hcmai.data.enrichment.caption.config import CaptionConfig
from hcmai.data.enrichment.caption.models.contracts import CaptionAdapter


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
    *,
    frame_store_id: str | None,
    resolved_revision: str | None,
) -> list[float]:
    """Run caption batches while retaining a typed row for every frame."""

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
        for frame in chunk:
            frame_id = str(frame["frame_id"])
            try:
                path = Path(str(frame["image_path"])).expanduser()
                image = load_image(
                    path if path.is_absolute() else root / path, mode="RGB"
                )
                image.thumbnail((config.image_size, config.image_size))
                valid.append((frame, image))
            except Exception as error:
                rows[frame_id], failures[frame_id] = _failure(
                    frame,
                    config,
                    frame_store_id=frame_store_id,
                    resolved_revision=resolved_revision,
                    stage="image_load",
                    error_code=type(error).__name__,
                    error_message=str(error),
                )
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
                        text=str(result).strip(),
                        frame_store_id=frame_store_id,
                        artifact_version=config.enrichment_version,
                        model_name=config.model_checkpoint,
                        model_revision=resolved_revision,
                        status=ProcessingStatus.COMPLETED,
                    )
        since_write += len(chunk)
        if since_write >= config.write_interval:
            write_caption_artifacts(output, order, rows, failures)
            since_write = 0
        progress.update(len(chunk))
        progress.set_postfix(failed=len(failures), refresh=False)
    progress.close()
    return latencies
