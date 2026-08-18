"""Build deterministic FrameContext V1 from materialized specialist evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hcmai.data.enrichment.context import FrameContextConfig
from hcmai.data.enrichment.pipeline import EnrichmentService


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canonical, specialist, output, lineage, and serializer settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--ocr-frames", type=Path, required=True)
    parser.add_argument("--object-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-store-id")
    parser.add_argument("--context-version", default="frame-context-v1")
    parser.add_argument("--caption-token-budget", type=int, default=80)
    parser.add_argument("--ocr-token-budget", type=int, default=80)
    parser.add_argument("--object-token-budget", type=int, default=40)
    parser.add_argument("--min-ocr-quality", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build configuration and delegate context materialization to the service."""

    args = parse_args(argv)
    config = FrameContextConfig(
        context_version=args.context_version,
        caption_token_budget=args.caption_token_budget,
        ocr_token_budget=args.ocr_token_budget,
        object_token_budget=args.object_token_budget,
        min_ocr_quality=args.min_ocr_quality,
    )
    EnrichmentService.build_frame_context(
        args.frames,
        args.captions,
        args.ocr_frames,
        args.object_frames,
        args.output,
        config,
        frame_store_id=args.frame_store_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
