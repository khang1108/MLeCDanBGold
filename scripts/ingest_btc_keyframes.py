#!/usr/bin/env python3
"""Ingest BTC-provided keyframes into the canonical frame store."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from hcmai.data.ingestion import BTCIngestionConfig, import_btc_frame_store


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--btc-root",
        type=Path,
        default=Path("data"),
        help="BTC root containing metadata/frames.parquet",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root used to resolve relative BTC image paths",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/btc-keyframes-v1/artifacts/frame_store"),
        help="Directory for canonical frame-store artifacts",
    )
    parser.add_argument(
        "--frame-store-id",
        default="btc-keyframes-v1",
        help="Canonical frame-store lineage identifier",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = import_btc_frame_store(
        BTCIngestionConfig(
            btc_root=args.btc_root,
            data_root=args.data_root,
            output_root=args.output_root,
            frame_store_id=args.frame_store_id,
        )
    )
    print(output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
