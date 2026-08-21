"""Build one adaptive FrameStore from raw videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from hcmai.data.pipeline import DataService
from hcmai.data.preprocessing import PreprocessingConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the small preprocessing command interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare FrameStore and print a compact result summary."""

    args = parse_args(argv)
    try:
        config = PreprocessingConfig.from_yaml(args.config)
        output = DataService.prepare_adaptive(
            args.config,
            resume=not args.no_resume,
            limit=args.limit,
        )
        data = DataService.load(output)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    frames = tuple(data.iter_frames())
    print(f"Videos: {len({frame.video_id for frame in frames})}")
    print(f"Frames: {len(frames)}")
    print(f"Output: {output}")
    if config.s3 is not None:
        artifacts_prefix = config.s3.artifacts_prefix_for_run(args.limit)
        print(
            "Published: "
            f"s3://{config.s3.bucket}/{artifacts_prefix}/latest.json"
        )
    print("Status: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
