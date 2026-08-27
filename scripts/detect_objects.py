"""Detect objects with YOLOE and publish canonical enrichment artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from hcmai.common.config import resolve_dataset_root, resolve_repository_path
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.data.enrichment.object_detection import (
    ObjectDetectionConfig,
    normalized_boxes,
    run_yoloe,
)

logger = get_logger(__name__)

DEFAULT_FRAMES = Path("artifacts/frame_store/frames.parquet")
DEFAULT_OUTPUT = Path("artifacts/enrichment/objects_yoloe")


def _positive_int(value: str) -> int:
    """Parse a CLI integer that must represent at least one item."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than or equal to 1")
    return parsed


def _confidence(value: str) -> float:
    """Parse a finite confidence threshold in the inclusive unit range."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and in the range [0, 1]")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse frame input, detector settings, and artifact destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Canonical object artifact directory; raw JSON is stored below raw/.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optional raw JSON directory used for resumable inference.",
    )
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--model", default="yoloe-26l-seg-pf.pt")
    parser.add_argument("--min-confidence", type=_confidence, default=0.20)
    parser.add_argument("--top-k", type=_positive_int, default=30)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--artifact-version",
        default="object-yoloe-v1",
        help="Version recorded in ObjectEvidence and the manifest.",
    )
    parser.add_argument(
        "--summary-min-confidence",
        type=_confidence,
        default=0.25,
    )
    parser.add_argument("--max-summary-labels", type=_positive_int, default=20)
    parser.add_argument("--frame-store-id")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Stop inference after N pending frames (smoke run; must be positive).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run YOLOE and publish raw plus canonical object enrichment."""

    args = parse_args(argv)
    configure_logging(args.log_level)
    config = ObjectDetectionConfig(
        model=args.model,
        min_confidence=args.min_confidence,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
        artifact_version=args.artifact_version,
        summary_min_confidence=args.summary_min_confidence,
        max_summary_labels=args.max_summary_labels,
    )
    frames = resolve_repository_path(args.frames)
    output = resolve_repository_path(args.output)
    raw_output = (
        resolve_repository_path(args.raw_output)
        if args.raw_output is not None
        else None
    )
    report = run_yoloe(
        frames,
        output,
        config,
        dataset_root=resolve_dataset_root(args.dataset_root),
        raw_output_root=raw_output,
        frame_store_id=args.frame_store_id,
        limit=args.limit,
    )
    failed = int(report.get("failed_frames", 0))
    skipped = int(report.get("inference_skipped_frames", 0))
    status = "DEGRADED" if failed or skipped else "COMPLETE"
    log_result = logger.warning if status == "DEGRADED" else logger.info
    log_result(
        "Object detection %s: inference_completed=%d inference_skipped=%d "
        "artifact_completed=%d artifact_failed=%d output=%s",
        status,
        report.get("inference_completed_frames", 0),
        skipped,
        report.get("completed_frames", 0),
        failed,
        output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
