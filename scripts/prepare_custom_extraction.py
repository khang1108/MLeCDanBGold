#!/usr/bin/env python3
"""Prepare deterministic custom-extraction inputs without downloading videos.

The command reads organizer media-info JSON only. It emits the native input
manifest and extractor configuration under one run root; extraction starts only
when the separate C++ executable is invoked later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hcmai.data.ingestion.custom_manifest import (
    build_native_input_manifest,
    write_extraction_config,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse metadata-only custom extraction preparation arguments.

    Args:
        argv: Optional argument sequence for tests; ``None`` reads process args.

    Returns:
        Namespace containing source metadata and generated-input locations.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--media-info-dir",
        type=Path,
        default=Path("data/media-info-aic25-b1/media-info"),
        help="Directory containing organizer {video_id}.json media-info records",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/custom-raw1fps-v1"),
        help="Isolated root for generated input, state, staging, and publication",
    )
    parser.add_argument(
        "--native-executable",
        type=Path,
        default=Path("build/keyframes_extraction/keyframe_extractor"),
        help="Native executable path recorded in extraction config provenance",
    )
    parser.add_argument(
        "--frame-store-id",
        default="custom-raw1fps-v1",
        help="Separate canonical lineage identifier for the custom corpus",
    )
    parser.add_argument(
        "--yt-dlp-binary",
        default="yt-dlp",
        help="Explicit yt-dlp executable or command name for the later native run",
    )
    return parser.parse_args(argv)


def _manifest_statistics(manifest_path: Path) -> dict[str, int]:
    """Compute deterministic preparation statistics from generated JSONL.

    Args:
        manifest_path: Complete JSONL manifest produced during this command.

    Returns:
        Video count, unique URL count, and metadata duration total in seconds.

    Raises:
        ValueError: If a just-written JSONL row has an unexpected shape.
    """

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("generated native manifest contains a non-object row")
    urls = {str(row["watch_url"]) for row in rows}
    return {
        "video_count": len(rows),
        "unique_url_count": len(urls),
        "metadata_length_seconds": sum(int(row["metadata_length_s"]) for row in rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Generate manifest/config inputs and print a compact machine-readable summary.

    Args:
        argv: Optional arguments for programmatic tests; ``None`` reads process args.

    Returns:
        Zero after successful metadata-only preparation.
    """

    args = parse_args(argv)
    input_root = args.run_root / "input"
    manifest_path = build_native_input_manifest(
        args.media_info_dir,
        input_root / "media_manifest.jsonl",
    )
    config_path = write_extraction_config(
        input_root / "extraction_config.json",
        run_root=args.run_root,
        native_executable=args.native_executable,
        frame_store_id=args.frame_store_id,
        yt_dlp_binary=args.yt_dlp_binary,
    )
    result = {
        **_manifest_statistics(manifest_path),
        "sample_period_ms": 1_000,
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
