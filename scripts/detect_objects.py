"""Detect objects with YOLOE and publish the BTC object-JSON contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pyarrow.parquet as pq

from hcmai.common.config import resolve_dataset_root
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.data.assets import FrameAssetError, FrameAssetResolver

logger = get_logger(__name__)

DEFAULT_FRAMES = Path("artifacts/frame_store/frames.parquet")
DEFAULT_OUTPUT = Path("data/objects_yoloe")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the frame source, detector limits, and object JSON destination."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--model", default="yoloe-26l-seg-pf.pt")
    parser.add_argument("--min-confidence", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, help="Stop after N frames (smoke run)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def normalized_boxes(
    pixel_boxes: Sequence[Sequence[float]], width: int, height: int
) -> list[list[float]]:
    """Convert pixel ``x1,y1,x2,y2`` into the BTC unit ``ymin,xmin,ymax,xmax``.

    The importer unpacks ``ymin, xmin, ymax, xmax`` and rejects values outside
    ``[0, 1]`` into a silent failed row, so order is swapped and values clamped.
    """

    def unit(value: float) -> float:
        return min(1.0, max(0.0, value))

    return [
        [
            unit(y_min / height),
            unit(x_min / width),
            unit(y_max / height),
            unit(x_max / width),
        ]
        for x_min, y_min, x_max, y_max in pixel_boxes
    ]


def pending_frames(
    frames_path: Path, output_root: Path, limit: int | None
) -> list[tuple[str, str]]:
    """Return ``(video_id, image_path)`` pairs that have no published JSON."""

    table = pq.read_table(frames_path, columns=["video_id", "image_path"])
    pending: list[tuple[str, str]] = []
    for row in table.to_pylist():
        video_id = str(row["video_id"])
        image_path = str(row["image_path"])
        target = output_root / video_id / f"{Path(image_path).stem}.json"
        if target.is_file():
            continue
        pending.append((video_id, image_path))
        if limit is not None and len(pending) >= limit:
            break
    return pending


def publish(target: Path, payload: dict[str, list]) -> None:
    """Write one object JSON through a same-directory atomic rename."""

    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_suffix(".json.staged")
    staged.write_text(json.dumps(payload), encoding="utf-8")
    staged.replace(target)


def main(argv: Sequence[str] | None = None) -> int:
    """Run YOLOE over pending frames and publish BTC-shaped object JSON."""

    args = parse_args(argv)
    configure_logging(args.log_level)

    from ultralytics import YOLOE

    resolver = FrameAssetResolver(resolve_dataset_root(args.dataset_root))
    pending = pending_frames(args.frames, args.output, args.limit)
    logger.info(
        "Object detection started model=%s pending=%d output=%s",
        args.model,
        len(pending),
        args.output,
    )
    if not pending:
        return 0

    model = YOLOE(args.model)
    logger.info("Model vocabulary size=%d", len(model.names))

    completed = skipped = 0
    for start in range(0, len(pending), args.batch_size):
        batch: list[tuple[str, Path]] = []
        for video_id, image_path in pending[start : start + args.batch_size]:
            try:
                batch.append((video_id, resolver.resolve_value(image_path)))
            except FrameAssetError as error:
                skipped += 1
                logger.warning("Skipping unavailable frame: %s", error)
        if not batch:
            continue

        results = model.predict(
            [str(image) for _, image in batch],
            conf=args.min_confidence,
            device=args.device,
            verbose=False,
        )
        for (video_id, image), result in zip(batch, results):
            height, width = result.orig_shape
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                payload: dict[str, list] = {
                    "detection_class_entities": [],
                    "detection_scores": [],
                    "detection_boxes": [],
                }
            else:
                order = boxes.conf.argsort(descending=True)[: args.top_k]
                payload = {
                    "detection_class_entities": [
                        str(result.names[int(index)])
                        for index in boxes.cls[order].tolist()
                    ],
                    "detection_scores": [
                        min(1.0, max(0.0, float(score)))
                        for score in boxes.conf[order].tolist()
                    ],
                    "detection_boxes": normalized_boxes(
                        boxes.xyxy[order].tolist(), width, height
                    ),
                }
            publish(args.output / video_id / f"{image.stem}.json", payload)
            completed += 1

        if completed % 2000 < args.batch_size:
            logger.info("Progress completed=%d skipped=%d", completed, skipped)

    status = "DEGRADED" if skipped else "COMPLETE"
    log_result = logger.warning if skipped else logger.info
    log_result(
        "Object detection %s: completed=%d skipped=%d output=%s",
        status,
        completed,
        skipped,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
