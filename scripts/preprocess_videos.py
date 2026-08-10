"""Build one adaptive FrameStore from raw videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from hcmai.data import FrameStore
from hcmai.data.preprocessing import PreprocessingConfig, prepare_frame_store


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
        output = prepare_frame_store(
            config,
            resume=not args.no_resume,
            limit=args.limit,
        )
        store = FrameStore.load(output)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    frames = tuple(store.iter_frames())
    print(f"Videos: {len({frame.video_id for frame in frames})}")
    print(f"Frames: {len(frames)}")
    print(f"Output: {output}")
    print("Status: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
