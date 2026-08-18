"""Import BTC-provided object JSON as structured frame evidence."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from hcmai.data.enrichment.pipeline import EnrichmentJobConfig, EnrichmentService


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/enrichment.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse object paths, lineage, and deterministic summary policy."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--objects-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame-store-id")
    parser.add_argument("--artifact-version")
    parser.add_argument("--summary-min-confidence", type=float)
    parser.add_argument("--max-summary-labels", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build configuration and delegate object import to the service."""

    args = parse_args(argv)
    job = EnrichmentJobConfig.from_yaml(args.config)
    objects_root = args.objects_root or job.objects_root
    output = args.output or job.object_output_dir
    config = replace(
        job.objects,
        objects_root=objects_root,
        output_dir=output,
        **{
            name: value
            for name, value in {
                "artifact_version": args.artifact_version,
                "summary_min_confidence": args.summary_min_confidence,
                "max_summary_labels": args.max_summary_labels,
            }.items()
            if value is not None
        },
    )
    report = EnrichmentService.import_objects(
        args.frames or job.frames_path,
        objects_root,
        output,
        config,
        frame_store_id=args.frame_store_id or job.frame_store_id,
    )
    return 0 if report["failed_frames"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
