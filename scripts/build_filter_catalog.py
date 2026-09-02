"""Command-line entrypoint for atomic offline Filter catalog builds."""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Sequence

from offline.filtering.builder import FilterCatalogBuildConfig, build_filter_catalog


def _parser() -> argparse.ArgumentParser:
    """Define explicit artifact inputs without importing online services."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--video-metadata", type=Path)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--ocr", type=Path)
    parser.add_argument("--objects", type=Path)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/filter/filter_catalog.sqlite"),
    )
    parser.add_argument("--catalog-version", required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build one catalog and print its machine-readable publication report."""

    arguments = _parser().parse_args(argv)
    report = build_filter_catalog(
        FilterCatalogBuildConfig(
            frames_path=arguments.frames,
            video_metadata_path=arguments.video_metadata,
            caption_path=arguments.captions,
            ocr_path=arguments.ocr,
            object_counts_path=arguments.objects,
            transcripts_path=arguments.transcripts,
            output_path=arguments.output,
            catalog_version=arguments.catalog_version,
            batch_size=arguments.batch_size,
        )
    )
    print(json.dumps(report.to_json_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
