"""Detect objects with YOLOE and publish canonical enrichment artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from hcmai.common.config import resolve_dataset_root, resolve_repository_path
from hcmai.common.utils.logging import configure_logging, get_logger
from offline.enrichment.dataset_cli import (
    add_dataset_arguments,
    apply_overrides,
    dataset_overrides,
)
from offline.enrichment.object_detection import run_yoloe
from offline.enrichment.pipeline import EnrichmentJobConfig

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/prepare.yaml")


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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_dataset_arguments(parser)
    parser.add_argument(
        "--output",
        type=Path,
        help="Canonical object artifact directory; raw JSON is stored below raw/.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optional raw JSON directory used for resumable inference.",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--vocab",
        type=Path,
        help="Text-prompt class list, one per line; omit to use the model vocabulary.",
    )
    parser.add_argument("--min-confidence", type=_confidence)
    parser.add_argument("--top-k", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--artifact-version",
        help="Version recorded in ObjectEvidence and the manifest.",
    )
    parser.add_argument(
        "--summary-min-confidence",
        type=_confidence,
    )
    parser.add_argument("--max-summary-labels", type=_positive_int)
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
    job = EnrichmentJobConfig.from_yaml(args.config, dataset=dataset_overrides(args))
    config = apply_overrides(
        job.objects,
        model=args.model,
        vocab_path=(
            str(resolve_repository_path(args.vocab)) if args.vocab is not None else None
        ),
        min_confidence=args.min_confidence,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
        artifact_version=args.artifact_version,
        summary_min_confidence=args.summary_min_confidence,
        max_summary_labels=args.max_summary_labels,
    )
    frames = resolve_repository_path(args.frames or job.frames_path)
    output = resolve_repository_path(args.output or job.object_output_dir)
    raw_output = (
        resolve_repository_path(args.raw_output)
        if args.raw_output is not None
        else None
    )
    report = run_yoloe(
        frames,
        output,
        config,
        dataset_root=resolve_dataset_root(args.data_root or job.data_root),
        raw_output_root=raw_output,
        frame_store_id=args.frame_store_id or job.frame_store_id,
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
