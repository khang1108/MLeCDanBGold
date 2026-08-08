"""Move nested AIC keyframe folders into one canonical directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw"))
    parser.add_argument("--destination", type=Path, default=Path("data/keyframes"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned moves; do not change files.",
    )
    return parser.parse_args()


def find_video_directories(source: Path) -> list[Path]:
    """Find video folders under every ``Keyframes_*/keyframes`` directory."""
    roots = sorted(
        path for path in source.glob("Keyframes_*/keyframes") if path.is_dir()
    )
    if not roots:
        raise FileNotFoundError(
            f"No Keyframes_*/keyframes directories found under {source}"
        )

    videos: list[Path] = []
    for root in roots:
        videos.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    if not videos:
        raise FileNotFoundError(f"No video directories found under {source}")
    return videos


def build_plan(source: Path, destination: Path) -> list[tuple[Path, Path]]:
    videos = find_video_directories(source)
    plan = [(video, destination / video.name) for video in videos]
    names = [target.name for _, target in plan]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        joined = ", ".join(duplicate_names[:5])
        raise FileExistsError(f"Duplicate video folders found: {joined}")

    collisions = [target for _, target in plan if target.exists()]
    if collisions:
        joined = ", ".join(str(path) for path in collisions[:5])
        raise FileExistsError(
            f"Destination already contains video folder(s): {joined}. "
            "Move them aside before retrying."
        )
    return plan


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    try:
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source}")
        plan = build_plan(source, destination)
        print(f"Found {len(plan)} video folder(s).")
        for current, target in plan:
            print(f"{current} -> {target}")
        if args.dry_run:
            print("Status: DRY-RUN")
            return 0

        destination.mkdir(parents=True, exist_ok=True)
        for current, target in plan:
            shutil.move(str(current), str(target))
        print(f"Moved {len(plan)} video folder(s) to {destination}")
        print("Status: PASSED")
        return 0
    except (FileExistsError, FileNotFoundError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
