"""Download dataset files from Google Drive excluding specific folders (e.g. Videos, features).

This script recursively scans a Google Drive folder, excludes unwanted directories
(e.g., 'Videos', 'features'), and downloads files into a target local directory
with a tqdm progress bar and automatic retry/resume logic.

Usage:
    PYTHONPATH=src python scripts/download_dataset.py
    PYTHONPATH=src python scripts/download_dataset.py --output-dir data/aic2025_raw
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import gdown
from tqdm import tqdm

DEFAULT_FOLDER_URL = "https://drive.google.com/drive/u/0/folders/10Z6VSb6lRJr7IXF_UPKVI7gqd8vqCtkZ"
DEFAULT_EXCLUDE_FOLDERS = ("videos")


def patch_gdown_tree_parser(exclude_folders: tuple[str, ...]) -> None:
    """Patch gdown's folder parser to skip specified directory names and log progress.

    Args:
        exclude_folders: Tuple of lower-cased directory names to skip during
            Google Drive folder traversal.
    """
    globs = gdown.download_folder.__globals__
    parse_embedded = globs["_parse_embedded_folder_view"]
    GoogleDriveFile = globs["_GoogleDriveFile"]

    scanned_count = 0

    def custom_parse_link(sess, folder_id, quiet=False, verify=True):
        nonlocal scanned_count
        folder_name, children = parse_embedded(
            sess=sess, folder_id=folder_id, verify=verify
        )

        gdrive_file = GoogleDriveFile(
            id=folder_id,
            name=folder_name,
            type=GoogleDriveFile.TYPE_FOLDER,
        )

        for child_id, child_name, child_type in children:
            child_name_lower = child_name.lower().strip()
            if child_name_lower in exclude_folders:
                tqdm.write(f"[SKIP DIRECTORY] Excluding '{child_name}' (ID: {child_id})")
                continue

            if child_type != GoogleDriveFile.TYPE_FOLDER:
                gdrive_file.children.append(
                    GoogleDriveFile(
                        id=child_id,
                        name=child_name,
                        type=child_type,
                    )
                )
                continue

            scanned_count += 1
            tqdm.write(f"[Phase 1 #{scanned_count}] Traversing folder: {child_name}")
            child = custom_parse_link(
                sess=sess,
                folder_id=child_id,
                quiet=quiet,
                verify=verify,
            )
            gdrive_file.children.append(child)
        return gdrive_file

    globs["_download_and_parse_google_drive_link"] = custom_parse_link


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for downloading dataset.

    Args:
        argv: Optional sequence of argument strings.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder-url",
        default=os.getenv("HCMAI_GDRIVE_URL", DEFAULT_FOLDER_URL),
        help="Google Drive folder URL",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("HCMAI_DATASET_ROOT", "data/raw_dataset"),
        help="Target local directory for downloaded files",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_FOLDERS),
        help="List of folder names to skip (case-insensitive)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts per file on rate-limit/network failure",
    )
    return parser.parse_args(argv)


def download_dataset(
    folder_url: str,
    output_dir: str | Path,
    exclude: Sequence[str] = DEFAULT_EXCLUDE_FOLDERS,
    max_retries: int = 5,
) -> int:
    """Traverse and download files from Google Drive with tqdm progress tracking.

    Args:
        folder_url: Google Drive folder URL.
        output_dir: Local destination path.
        exclude: Sequence of directory names to exclude.
        max_retries: Number of retry attempts per file.

    Returns:
        0 on success, non-zero on failure.
    """
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if isinstance(exclude, str):
        exclude = (exclude,)
    exclude_normalized = tuple(f.lower().strip() for f in exclude)

    print("=" * 60)
    print("Google Drive Dataset Downloader")
    print(f"Target directory : {output_path}")
    print(f"Excluded folders : {', '.join(exclude_normalized)}")
    print("=" * 60)

    patch_gdown_tree_parser(exclude_normalized)

    print("\nPhase 1: Traversing Google Drive folder structure...")
    files_to_download = gdown.download_folder(
        url=folder_url,
        output=str(output_path),
        skip_download=True,
        quiet=True,
    )

    total_items = len(files_to_download)
    print(f"Found {total_items} items (files & directories).\n")

    print("Phase 2: Downloading files with tqdm progress tracking...\n")
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    pbar = tqdm(files_to_download, desc="Downloading", unit="item")

    for file_item in pbar:
        file_id = file_item.id
        rel_path = file_item.path
        local_path = Path(file_item.local_path)

        if file_id is None:
            # Directory entry
            local_path.mkdir(parents=True, exist_ok=True)
            continue

        if local_path.exists() and local_path.stat().st_size > 0:
            skipped_count += 1
            pbar.set_postfix(
                downloaded=downloaded_count,
                skipped=skipped_count,
                failed=failed_count,
            )
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"https://drive.google.com/uc?id={file_id}"
        success = False

        for attempt in range(1, max_retries + 1):
            try:
                gdown.download(
                    url=url,
                    output=str(local_path),
                    quiet=True,
                    resume=True,
                )
                if local_path.exists() and local_path.stat().st_size > 0:
                    downloaded_count += 1
                    success = True
                    break
            except Exception as e:
                pbar.write(
                    f"Attempt {attempt}/{max_retries} failed for {rel_path}: {e}"
                )
                if attempt < max_retries:
                    wait_sec = attempt * 5
                    time.sleep(wait_sec)

        if not success:
            failed_count += 1
            pbar.write(f"WARNING: Failed to download {rel_path} after {max_retries} retries.")

        pbar.set_postfix(
            downloaded=downloaded_count,
            skipped=skipped_count,
            failed=failed_count,
        )

    print("\n" + "=" * 60)
    print("Download Summary:")
    print(f"  - Total items   : {total_items}")
    print(f"  - Skipped       : {skipped_count}")
    print(f"  - Downloaded    : {downloaded_count}")
    print(f"  - Failed        : {failed_count}")
    print("=" * 60)

    return 0 if failed_count == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for downloading dataset.

    Args:
        argv: Optional sequence of CLI arguments.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    args = parse_args(argv)
    return download_dataset(
        folder_url=args.folder_url,
        output_dir=args.output_dir,
        exclude=args.exclude,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    raise SystemExit(main())
