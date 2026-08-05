"""Download TSV-listed ZIPs, extract videos, and upload each batch to S3."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import httpx
from tqdm import tqdm


def read_links(tsv_path: Path) -> list[tuple[str, str]]:
    """Read and validate archive names and URLs from a two-column TSV."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    with tsv_path.open(encoding="utf-8", newline="") as source:
        for line_number, row in enumerate(csv.reader(source, delimiter="\t"), 1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 2:
                raise ValueError(f"{tsv_path}:{line_number}: expected 2 columns")
            archive_name, url = (value.strip() for value in row)
            if Path(archive_name).name != archive_name or not archive_name.endswith(".zip"):
                raise ValueError(f"{tsv_path}:{line_number}: invalid ZIP name")
            if archive_name in seen:
                raise ValueError(f"{tsv_path}:{line_number}: duplicate ZIP name")
            if not url.startswith(("https://", "http://")):
                raise ValueError(f"{tsv_path}:{line_number}: invalid HTTP(S) URL")
            seen.add(archive_name)
            rows.append((archive_name, url))
    if not rows:
        raise ValueError(f"No dataset links found in {tsv_path}")
    return rows


def download_zip(url: str, destination: Path, max_retries: int) -> None:
    """Stream one URL to disk, retrying without leaving a completed partial file."""
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0)) or None
                with (
                    partial.open("wb") as sink,
                    tqdm(
                        total=total,
                        desc=f"Download {destination.name}",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress,
                ):
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        sink.write(chunk)
                        progress.update(len(chunk))
            partial.replace(destination)
            return
        except (httpx.HTTPError, OSError):
            partial.unlink(missing_ok=True)
            if attempt == max_retries:
                raise
            time.sleep(min(2**attempt, 30))


def _safe_mp4_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    names: set[str] = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if member.is_dir() or path.suffix.lower() != ".mp4":
            continue
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe ZIP member: {member.filename}")
        filename = path.name
        if filename in names:
            raise ValueError(f"Duplicate video filename in ZIP: {filename}")
        names.add(filename)
        members.append(member)
    if not members:
        raise ValueError("ZIP contains no .mp4 files")
    return members


def extract_batch(archive_path: Path, output_dir: Path, batch_name: str) -> Path:
    """Extract MP4 files into an atomic ``<batch>/videos`` directory."""
    batch_dir = output_dir / batch_name
    existing_videos = batch_dir / "videos"
    if batch_dir.exists():
        if existing_videos.is_dir() and any(existing_videos.glob("*.mp4")):
            archive_path.unlink(missing_ok=True)
            return batch_dir
        raise FileExistsError(f"Refusing to overwrite incomplete directory: {batch_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{batch_name}-", dir=output_dir))
    try:
        videos_dir = temp_dir / "videos"
        videos_dir.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for member in _safe_mp4_members(archive):
                with archive.open(member) as source, (videos_dir / PurePosixPath(member.filename).name).open("wb") as sink:
                    shutil.copyfileobj(source, sink)
        temp_dir.replace(batch_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    archive_path.unlink()
    return batch_dir


def make_s3_client():
    """Create an S3 client using boto3's standard credential chain."""
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError('boto3 is required; install with: aic/bin/python -m pip install -e ".[s3]"') from error
    return boto3.client("s3", endpoint_url=os.getenv("HCMAI_S3_ENDPOINT_URL"))


def upload_batch(client, batch_dir: Path, bucket: str, prefix: str) -> int:
    """Upload a batch directory while preserving ``<batch>/videos/<file>`` keys."""
    video_paths = sorted((batch_dir / "videos").glob("*.mp4"))
    total_bytes = sum(path.stat().st_size for path in video_paths)
    key_root = "/".join(part for part in (prefix.strip("/"), batch_dir.name) if part)
    with tqdm(
        total=total_bytes,
        desc=f"Upload {batch_dir.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        for video_path in video_paths:
            client.upload_file(
                str(video_path),
                bucket,
                f"{key_root}/videos/{video_path.name}",
                Callback=progress.update,
            )
    return len(video_paths)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, default=Path("data/data_link.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--bucket", default=os.getenv("HCMAI_S3_BUCKET"))
    parser.add_argument("--prefix", default=os.getenv("HCMAI_S3_PREFIX", ""))
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Validate and list batches only")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    rows = read_links(args.links)
    if args.dry_run:
        for archive_name, url in rows:
            print(f"{Path(archive_name).stem}: {url}")
        return 0
    if not args.bucket:
        raise ValueError("Set HCMAI_S3_BUCKET or pass --bucket")

    client = make_s3_client()
    for archive_name, url in rows:
        batch_name = Path(archive_name).stem
        batch_dir = args.output_dir / batch_name
        if not ((batch_dir / "videos").is_dir() and any((batch_dir / "videos").glob("*.mp4"))):
            archive_path = args.output_dir / archive_name
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if not archive_path.exists():
                print(f"[{batch_name}] Downloading {url}")
                download_zip(url, archive_path, args.max_retries)
            print(f"[{batch_name}] Extracting to {batch_dir / 'videos'}")
            extract_batch(archive_path, args.output_dir, batch_name)
        count = upload_batch(client, batch_dir, args.bucket, args.prefix)
        print(f"[{batch_name}] Uploaded {count} video(s) to s3://{args.bucket}")
        shutil.rmtree(batch_dir / "videos")
        print(f"[{batch_name}] Removed local directory {batch_dir / 'videos'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, httpx.HTTPError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
