"""Prepare canonical frame metadata from a mounted AIC dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from hcmai.data import (
    inventory_corpus,
    prepare_dataset,
    validate_dataset,
)


def _positive_int(value: str) -> int:
    """Parse a positive command-line integer."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse data preparation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("HCMAI_DATASET_ROOT"),
    )
    parser.add_argument(
        "--output-root",
        default=os.getenv("HCMAI_DATA_ROOT"),
    )
    parser.add_argument(
        "--dataset-version",
        default=os.getenv("HCMAI_DATASET_VERSION"),
    )
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--thumbnail-max-edge",
        type=_positive_int,
        default=320,
    )
    parser.add_argument("--no-resume", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--inventory-only", action="store_true")
    action.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    missing = [
        name
        for name, value in (
            ("--dataset-root", args.dataset_root),
            ("--output-root", args.output_root),
            ("--dataset-version", args.dataset_version),
        )
        if not value
    ]
    if missing:
        parser.error(f"missing required configuration: {', '.join(missing)}")

    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run inventory, ingestion, or validation."""

    args = parse_args(argv)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    if args.inventory_only:
        inventory_corpus(
            dataset_root,
            output_root,
            args.dataset_version,
            limit=args.limit,
        )
        print("Inventory completed")
        return 0

    if args.validate_only:
        report = validate_dataset(
            dataset_root,
            output_root,
            args.dataset_version,
            deep=True,
        )
        print("Validation completed")
        return 0 if report.get("valid", False) else 1

    try:
        frames_path = prepare_dataset(
            dataset_root,
            output_root,
            args.dataset_version,
            limit=args.limit,
            resume=not args.no_resume,
            thumbnail_max_edge=args.thumbnail_max_edge,
            deep_validation=True,
        )
    except ValueError as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Metadata ready: {frames_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
