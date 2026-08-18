"""Import BTC-provided object JSON as structured frame evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hcmai.data.enrichment.objects.config import ObjectConfig
from hcmai.data.enrichment.pipeline import EnrichmentService


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse object paths, lineage, and deterministic summary policy."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--objects-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-store-id")
    parser.add_argument("--artifact-version", default="object-v1")
    parser.add_argument("--summary-min-confidence", type=float, default=0.25)
    parser.add_argument("--max-summary-labels", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build configuration and delegate object import to the service."""

    args = parse_args(argv)
    config = ObjectConfig(
        objects_root=args.objects_root,
        output_dir=args.output,
        artifact_version=args.artifact_version,
        summary_min_confidence=args.summary_min_confidence,
        max_summary_labels=args.max_summary_labels,
    )
    report = EnrichmentService.import_objects(
        args.frames,
        args.objects_root,
        args.output,
        config,
        frame_store_id=args.frame_store_id,
    )
    return 0 if report["failed_frames"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
