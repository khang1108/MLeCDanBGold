"""Prepare canonical V3C/VBS frames from external local MP4 files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from offline.config import VBSConfig
from offline.corpus.frames import FrameBuildConfig, build_frames


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vbs_prepare.yaml"))
    parser.add_argument("--videos-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--sample-period-ms", type=int, default=1_000)
    parser.add_argument("--long-edge", type=int, default=1_024)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--frame-store-id", default="v3c-v1")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = VBSConfig.from_yaml(args.config)
    videos_root = args.videos_root or settings.paths.videos_root
    work_root = args.work_root or settings.paths.work_root
    report = build_frames(
        work_root,
        videos_root=videos_root,
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
