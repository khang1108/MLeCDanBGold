"""Build and validate the canonical BTC frame store from enrichment config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from hcmai.data.pipeline import DataService


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the BTC-native data preparation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Enrichment YAML containing the dataset BTC ingestion mapping",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional root for frame asset resolution after preparation",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build and report the configured canonical Parquet artifact."""

    args = parse_args(argv)
    try:
        output = DataService.prepare(args.config)
        data = DataService.load(output, dataset_root=args.dataset_root)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    frames = tuple(data.iter_frames())
    print(f"Videos: {len({row.video_id for row in frames})}")
    print(f"Frames: {len(frames)}")
    print(f"Output: {output}")
    print("Status: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
