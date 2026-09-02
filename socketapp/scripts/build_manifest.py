#!/usr/bin/env python3
"""Build an explicit SocketApp video manifest from a local video directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from socketapp.catalog import CatalogError, VideoCatalog


def main(argv: list[str] | None = None) -> int:
    """Discover videos and atomically write a canonical-ID manifest."""

    parser = argparse.ArgumentParser(
        description="Create a SocketApp JSON manifest from local video files."
    )
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        catalog = VideoCatalog(args.video_root)
    except CatalogError as error:
        parser.error(str(error))

    root = catalog.root
    records = [
        {
            "video_id": entry.video_id,
            "path": entry.path.relative_to(root).as_posix(),
            "mime_type": entry.media_type,
        }
        for entry in catalog.entries()
    ]
    payload = {"version": 1, "videos": records}
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"wrote {len(records)} videos to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
