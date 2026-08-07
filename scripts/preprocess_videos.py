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
    parser.add_argument("--config", type=Path)
    parser.add_argument("--videos-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def _config(args: argparse.Namespace) -> PreprocessingConfig:
    """Load YAML or environment settings, then apply CLI path overrides."""

    initial = {
        name: value
        for name, value in {
            "videos_root": args.videos_root,
            "output_root": args.output,
        }.items()
        if value is not None
    }
    config = (
        PreprocessingConfig.from_yaml(args.config)
        if args.config
        else PreprocessingConfig(**initial)
    )
    updates = {
        "videos_root": args.videos_root,
        "output_root": args.output,
        "limit": args.limit,
        "resume": not args.no_resume if args.no_resume else None,
    }
    values = config.model_dump()
    values.update({name: value for name, value in updates.items() if value is not None})
    return PreprocessingConfig.model_validate(values)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare FrameStore and print a compact result summary."""

    try:
        output = prepare_frame_store(_config(parse_args(argv)))
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
