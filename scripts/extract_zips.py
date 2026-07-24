"""Extract every zip archive found under ``data/``.

The script scans the data directory recursively, extracts each ``.zip`` archive
into the data root, and keeps going until no new zip files appear. That makes it
safe for archives that unpack into nested folders containing more archives.
After a successful extraction, the source archive is removed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Directory to scan for zip files and extract into",
    )
    return parser.parse_args(argv)


def _safe_target(root: Path, member_name: str) -> Path:
    """Resolve a zip member path and reject path traversal attempts."""

    target = (root / member_name).resolve()
    root_resolved = root.resolve()
    if root_resolved not in target.parents and target != root_resolved:
        raise ValueError(f"Refusing to extract outside data dir: {member_name}")
    return target


def extract_zip_archive(archive_path: Path, output_dir: Path) -> int:
    """Extract one zip archive into ``output_dir`` and return the file count."""

    extracted_files = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = _safe_target(output_dir, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted_files += 1
    return extracted_files


def extract_all_zips(data_dir: Path) -> int:
    """Extract all zip files under ``data_dir`` until no new archives remain."""

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    seen: set[Path] = set()
    total_archives = 0
    total_files = 0

    while True:
        archives = sorted(
            path
            for path in data_dir.rglob("*.zip")
            if path.is_file() and path not in seen
        )
        if not archives:
            break

        for archive_path in archives:
            extracted = extract_zip_archive(archive_path, data_dir)
            archive_path.unlink()
            seen.add(archive_path)
            total_archives += 1
            total_files += extracted
            print(f"Extracted {archive_path} -> {extracted} file(s), removed archive")

    print(f"Archives processed: {total_archives}")
    print(f"Files extracted   : {total_files}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    try:
        return extract_all_zips(args.data_dir)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
