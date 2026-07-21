"""Prepare canonical frame metadata from a mounted AIC dataset.

This CLI script is the main entry point for the offline data pipeline.
It supports three exclusive modes:

* **Full pipeline** (default): inventory → ingest → validate.
* ``--inventory-only``: inspect the dataset without writing metadata.
* ``--validate-only``: re-validate previously ingested metadata.

All three required paths can be supplied via environment variables so
that repeated invocations omit the flags:

.. code-block:: bash

    export HCMAI_DATASET_ROOT=/mnt/aic/dataset
    export HCMAI_DATA_ROOT=data/aic2025
    export HCMAI_DATASET_VERSION=aic2025_s1_v2

    PYTHONPATH=src python scripts/prepare_data.py --limit 100
    PYTHONPATH=src python scripts/prepare_data.py
    PYTHONPATH=src python scripts/prepare_data.py --validate-only

Exit codes:
    0: Pipeline completed successfully (or inventory finished).
    1: Validation failed or an unrecoverable error occurred.
"""

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
    """Parse and validate a positive integer from a CLI string argument.

    Used as an ``argparse`` ``type`` callback so that invalid values are
    rejected before ``parse_args`` returns.

    Args:
        value: Raw string supplied on the command line.

    Returns:
        The parsed integer, guaranteed to be greater than zero.

    Raises:
        argparse.ArgumentTypeError: If the string cannot be parsed as an
            integer or the resulting value is less than one.
        ValueError: If ``value`` cannot be converted to ``int``.
    """

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate data-preparation command-line arguments.

    Reads ``--dataset-root``, ``--output-root``, and
    ``--dataset-version`` from the command line or their corresponding
    environment variables.  Exits with an error message if any of the
    three required values are missing.

    Args:
        argv: Explicit argument list used instead of ``sys.argv[1:]``.
            Pass ``None`` (default) to read from the process arguments.

    Returns:
        Populated ``argparse.Namespace`` with all parsed arguments.

    Raises:
        SystemExit: If required arguments are missing or ``--help`` is
            requested.
    """

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
    """Run the data pipeline in one of three exclusive modes.

    Dispatches to ``inventory_corpus``, ``validate_dataset``, or the
    full ``prepare_dataset`` pipeline based on the parsed CLI flags.
    Prints a short status message to stdout on success and an error
    description to stderr on failure.

    Args:
        argv: Explicit argument list forwarded to ``parse_args``.  Pass
            ``None`` (default) to read from ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` if validation failed or an unrecoverable
        error was raised during data preparation.
    """

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
