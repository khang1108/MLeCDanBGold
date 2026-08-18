"""Build deterministic FrameContext V1 from materialized specialist evidence."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from hcmai.data.enrichment.pipeline import EnrichmentJobConfig, EnrichmentService


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/enrichment.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canonical, specialist, output, lineage, and serializer settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--ocr-frames", type=Path)
    parser.add_argument("--object-frames", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame-store-id")
    parser.add_argument("--context-version")
    parser.add_argument("--caption-token-budget", type=int)
    parser.add_argument("--ocr-token-budget", type=int)
    parser.add_argument("--object-token-budget", type=int)
    parser.add_argument("--min-ocr-quality", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build configuration and delegate context materialization to the service."""

    args = parse_args(argv)
    job = EnrichmentJobConfig.from_yaml(args.config)
    config = replace(
        job.context,
        **{
            name: value
            for name, value in {
                "context_version": args.context_version,
                "caption_token_budget": args.caption_token_budget,
                "ocr_token_budget": args.ocr_token_budget,
                "object_token_budget": args.object_token_budget,
                "min_ocr_quality": args.min_ocr_quality,
            }.items()
            if value is not None
        },
    )
    EnrichmentService.build_frame_context(
        args.frames or job.frames_path,
        args.captions or job.caption_output_dir / "captions.parquet",
        args.ocr_frames or job.ocr_output_dir / "frames.parquet",
        args.object_frames or job.object_output_dir / "frames.parquet",
        args.output or job.context_output_dir,
        config,
        frame_store_id=args.frame_store_id or job.frame_store_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
