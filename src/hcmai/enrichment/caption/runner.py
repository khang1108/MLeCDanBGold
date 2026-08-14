"""Batch execution for caption enrichment."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm.auto import tqdm

from hcmai.common.schemas import FrameEnrichment, ProcessingStatus
from hcmai.common.utils.image import load_image
from hcmai.enrichment.caption.artifacts import write_caption_artifacts
from hcmai.enrichment.caption.models.contracts import CaptionAdapter
from hcmai.enrichment.caption.config import CaptionConfig, ENRICHMENT_VERSION


def _failure(
    frame_id: str, config: CaptionConfig, stage: str, error: Exception
) -> tuple[FrameEnrichment, dict[str, str]]:
    message = str(error).strip()[:300] or type(error).__name__
    row = FrameEnrichment.model_validate(
        {
            "frame_id": frame_id,
            "model_name": config.model_checkpoint,
            "enrichment_version": config.enrichment_version,
            "status": ProcessingStatus.FAILED,
            "error_message": message,
        }
    )
    detail = {
        "frame_id": frame_id,
        ENRICHMENT_VERSION: config.enrichment_version,
        "processing_stage": stage,
        "exception_category": type(error).__name__,
        "error_message": message,
    }
    return row, detail


def run_batches(
    todo: list[dict[str, Any]],
    order: list[str],
    rows: dict[str, FrameEnrichment],
    failures: dict[str, dict[str, str]],
    captioner: CaptionAdapter,
    config: CaptionConfig,
    output: Path,
    root: Path,
) -> list[float]:
    latencies, since_write = [], 0
    progress = tqdm(
        total=len(order),
        initial=len(order) - len(todo),
        desc="Generating captions",
        unit="frame",
        dynamic_ncols=True,
    )
    for start in range(0, len(todo), config.batch_size):
        chunk, valid = todo[start : start + config.batch_size], []
        for frame in chunk:
            frame_id = frame["frame_id"]
            try:
                path = Path(str(frame["image_path"])).expanduser()
                image = load_image(path if path.is_absolute() else root / path, mode="RGB")
                image.thumbnail((config.image_size, config.image_size))
                valid.append((frame_id, image))
            except Exception as error:
                rows[frame_id], failures[frame_id] = _failure(
                    frame_id, config, "image_load", error
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
            for (frame_id, _), result in zip(valid, results):
                if isinstance(result, Exception) or result is None or not str(result).strip():
                    cause = (
                        result
                        if isinstance(result, Exception)
                        else ValueError("empty caption")
                    )
                    rows[frame_id], failures[frame_id] = _failure(
                        frame_id, config, "model", cause
                    )
                else:
                    rows[frame_id] = FrameEnrichment.model_validate(
                        {
                            "frame_id": frame_id,
                            "caption": str(result).strip(),
                            "model_name": config.model_checkpoint,
                            "enrichment_version": config.enrichment_version,
                        }
                    )
        since_write += len(chunk)
        if since_write >= config.write_interval:
            write_caption_artifacts(output, order, rows, failures)
            since_write = 0
        progress.update(len(chunk))
        progress.set_postfix(failed=len(failures), refresh=False)
    progress.close()
    return latencies
