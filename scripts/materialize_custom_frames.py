#!/usr/bin/env python3
"""Materialize validated published custom frames without decoding or inference.

This command only checks native published bundles and writes the canonical
Parquet/manifest pair. It never invokes yt-dlp, FFmpeg, enrichment models, or
remote compute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from offline.ingestion.custom_frames import (
    CustomFrameStoreConfig,
    materialize_custom_frame_store,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse native run, output, lineage, and optional video-ID filters."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-store-id", required=True)
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Published video ID to materialize; omit to select every published bundle",
    )
    return parser.parse_args(argv)


def _published_video_ids(run_root: Path) -> tuple[str, ...]:
    """Discover a deterministic full-corpus selection from published directories."""

    published_root = run_root / "published"
    video_ids = (
        tuple(sorted(path.name for path in published_root.iterdir() if path.is_dir()))
        if published_root.is_dir()
        else ()
    )
    if not video_ids:
        raise ValueError("run_root contains no published video bundles")
    return video_ids


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize selected native bundles and print a JSON summary."""

    args = parse_args(argv)
    selected_video_ids = tuple(args.video_id) or _published_video_ids(args.run_root)
    output = materialize_custom_frame_store(
        CustomFrameStoreConfig(
            run_root=args.run_root,
            output_root=args.output_root,
            frame_store_id=args.frame_store_id,
            selected_video_ids=selected_video_ids,
        )
    )
    manifest = json.loads((args.output_root / "manifest.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "frames_path": str(output),
                "frame_count": manifest["frame_count"],
                "video_count": manifest["video_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
