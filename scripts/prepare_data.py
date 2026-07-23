"""Build canonical ``frames.parquet`` from an AIC dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from hcmai.data import FrameStore, prepare_frames


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the canonical data-builder arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build and report the canonical Parquet artifact."""

    args = parse_args(argv)
    try:
        output = prepare_frames(args.dataset_root, args.output)
        frames = FrameStore.load(output)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Videos: {len({row.video_id for row in frames.iter_frames()})}")
    print(f"Frames: {len(tuple(frames.iter_frames()))}")
    print(f"Output: {output}")
    print("Status: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
