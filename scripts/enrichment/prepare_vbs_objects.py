"""Run VBS object enrichment through the hosted model API and store artifacts locally."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from PIL import Image

from hcmai.common.utils.logging import configure_logging, get_logger
from llm.pipeline import LLMService
from offline.artifact_readers import FrameAssetError, OfflineFrameAssetResolver
from offline.enrichment.object_detection import (
    materialize_object_artifacts,
    pending_frames,
    publish_raw_json,
)
from offline.enrichment.pipeline import EnrichmentJobConfig

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/vbs_prepare.yaml")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than or equal to 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--inference-url",
        default=os.getenv("HCMAI_INFERENCE_BASE_URL", "http://127.0.0.1:8100"),
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    job = EnrichmentJobConfig.from_yaml(args.config)
    config = job.objects

    data_root = args.data.expanduser().resolve()
    frames_path = Path(job.frames_path).expanduser().resolve()
    output = Path(job.object_output_dir).expanduser().resolve()
    raw_root = output / "raw"
    resolver = OfflineFrameAssetResolver(data_root)
    pending = pending_frames(frames_path, raw_root, args.limit)

    service = LLMService.remote(args.inference_url, timeout_seconds=args.timeout)
    completed = skipped = 0
    try:
        readiness = service.readiness()
        if not readiness.capabilities.objects:
            raise RuntimeError("remote inference service does not advertise object detection")
        status = readiness.models.get("objects")
        if status is None or not status.loaded:
            raise RuntimeError("remote object detection model is not loaded")
        if status.checkpoint and Path(status.checkpoint).name != Path(config.model).name:
            raise RuntimeError(
                "remote object model mismatch: "
                f"expected={config.model!r} actual={status.checkpoint!r}"
            )

        for start in range(0, len(pending), config.batch_size):
            selected: list[tuple[str, Path]] = []
            for video_id, image_path in pending[start : start + config.batch_size]:
                try:
                    selected.append((video_id, resolver.resolve_value(image_path)))
                except FrameAssetError as error:
                    skipped += 1
                    logger.warning("Skipping unavailable frame: %s", error)
            if not selected:
                continue

            images: list[Image.Image] = []
            try:
                for _, path in selected:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                item_ids = [f"{video_id}_{path.stem}" for video_id, path in selected]
                response = service.objects(
                    images,
                    min_confidence=config.min_confidence,
                    top_k=config.top_k,
                    item_ids=item_ids,
                )
                for (video_id, path), item in zip(selected, response.items, strict=True):
                    publish_raw_json(
                        raw_root / video_id / f"{path.stem}.json",
                        {
                            "detection_class_entities": list(item.labels),
                            "detection_scores": list(item.scores),
                            "detection_boxes": [list(box) for box in item.boxes],
                        },
                    )
                    completed += 1
            finally:
                for image in images:
                    image.close()
    finally:
        service.close()

    manifest = materialize_object_artifacts(
        frames_path,
        raw_root,
        output,
        config,
        frame_store_id=job.frame_store_id,
    )
    manifest.update(
        inference_backend="remote_api",
        inference_url=args.inference_url,
        inference_completed_frames=completed,
        inference_skipped_frames=skipped,
    )
    logger.info(
        "VBS object enrichment complete: completed=%d skipped=%d artifact_failed=%d output=%s",
        completed,
        skipped,
        manifest.get("failed_frames", 0),
        output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
