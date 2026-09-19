"""Prepare the canonical local VBS frame store from data/videos/*.mp4."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from offline.vbs.frames import FrameBuildConfig, build_frames


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--sample-period-ms", type=int, default=1_000)
    parser.add_argument("--long-edge", type=int, default=1_024)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--frame-store-id", default="vbs-local-v1")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_frames(
        args.data,
        config=FrameBuildConfig(
            sample_period_ms=args.sample_period_ms,
            long_edge=args.long_edge,
            jpeg_quality=args.jpeg_quality,
            frame_store_id=args.frame_store_id,
            resume=not args.no_resume,
        ),
        video_ids=set(args.video_id) if args.video_id else None,
        limit=args.limit,
    )
    print(f"Videos: {report.video_count}")
    print(f"Frames: {report.frame_count}")
    print(f"Reused videos: {report.reused_video_count}")
    print(f"Frames artifact: {report.frames_path}")
    print(f"Manifest: {report.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
