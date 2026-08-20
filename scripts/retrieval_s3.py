"""Transfer fast-track retrieval inputs and validated index bundles through S3.

This module intentionally owns only the S3 boundary for the offline retrieval
workflow. It does not build embeddings, mutate canonical identity, or expose a
partially uploaded bundle through the ``latest.json`` pointer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

LOGGER = logging.getLogger("hcmai.retrieval_s3")


@dataclass(frozen=True)
class S3TransferStats:
    """Counts from one resumable S3 prefix transfer."""

    prefix: str
    files: int
    downloaded: int
    skipped: int
    bytes_transferred: int


@dataclass(frozen=True)
class BundleFile:
    """Content identity for one file in a versioned retrieval bundle."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class RetrievalBundlePublication:
    """Published S3 location for one validated retrieval-index bundle."""

    bucket: str
    bundle_id: str
    version_prefix: str
    latest_key: str
    file_count: int
    total_bytes: int


def _prefix(value: str) -> str:
    """Normalize one bucket-relative S3 prefix and reject traversal."""

    normalized = value.strip().strip("/")
    if not normalized or normalized.startswith("s3://"):
        raise ValueError("S3 prefix must be a non-empty bucket-relative key")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("S3 prefix must not contain path traversal")
    return normalized + "/"


def _relative_key(key: str, prefix: str) -> Path:
    """Map one listed S3 key below a prefix to a safe local relative path."""

    if not key.startswith(prefix):
        raise ValueError(f"S3 key is outside requested prefix: {key}")
    relative = Path(key[len(prefix) :])
    if (
        not relative.parts
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"S3 key has an unsafe relative path: {key}")
    return relative


def _list_objects(client: Any, bucket: str, prefix: str) -> list[tuple[str, int]]:
    """List non-empty objects below one prefix in deterministic key order."""

    normalized = _prefix(prefix)
    objects: list[tuple[str, int]] = []
    pages = client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket,
        Prefix=normalized,
    )
    for page in pages:
        for item in page.get("Contents", ()):
            key = str(item.get("Key", ""))
            size = int(item.get("Size", 0))
            if key and size > 0 and key != normalized:
                _relative_key(key, normalized)
                objects.append((key, size))
    return sorted(objects)


def _download_one(
    client: Any,
    bucket: str,
    key: str,
    size: int,
    prefix: str,
    destination: Path,
) -> tuple[str, int]:
    """Download one object atomically, or resume it when its size matches."""

    relative = _relative_key(key, prefix)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == size:
        return "skipped", 0

    partial = target.with_name(f".{target.name}.part")
    try:
        partial.unlink(missing_ok=True)
        client.download_file(bucket, key, str(partial))
        if not partial.is_file() or partial.stat().st_size != size:
            raise OSError(
                f"Downloaded size mismatch for s3://{bucket}/{key}: "
                f"expected {size}, got {partial.stat().st_size if partial.exists() else 0}"
            )
        os.replace(partial, target)
        return "downloaded", size
    finally:
        partial.unlink(missing_ok=True)


def download_prefix(
    client: Any,
    bucket: str,
    prefix: str,
    destination: Path | str,
    *,
    workers: int = 16,
    dry_run: bool = False,
) -> S3TransferStats:
    """Resume a parallel download of one S3 prefix into a local directory.

    Existing files with the same byte size are retained. Every replacement is
    downloaded to a sibling temporary file and atomically renamed, so an
    interrupted transfer cannot leave a truncated image or Parquet file at its
    final path. ``dry_run`` lists and reports the prefix without writing files.
    """

    if workers <= 0:
        raise ValueError("S3 download workers must be positive")
    normalized = _prefix(prefix)
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    objects = _list_objects(client, bucket, normalized)
    if not objects:
        raise FileNotFoundError(f"No objects found at s3://{bucket}/{normalized}")
    if dry_run:
        stats = S3TransferStats(
            prefix=normalized.rstrip("/"),
            files=len(objects),
            downloaded=0,
            skipped=0,
            bytes_transferred=0,
        )
        LOGGER.info(
            "S3 dry-run prefix=%s files=%d remote_bytes=%d",
            stats.prefix,
            stats.files,
            sum(size for _, size in objects),
        )
        return stats
    downloaded = skipped = transferred = 0
    errors: list[tuple[str, Exception]] = []

    def submit_one(item: tuple[str, int]) -> tuple[str, int]:
        return _download_one(
            client,
            bucket,
            item[0],
            item[1],
            normalized,
            destination_path,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[tuple[str, int]], tuple[str, int]] = {
            pool.submit(submit_one, item): item for item in objects
        }
        with tqdm(
            total=len(futures),
            desc=f"Downloading {normalized.rstrip('/')}",
            unit="file",
            dynamic_ncols=True,
        ) as progress:
            for future in as_completed(futures):
                key, size = futures[future]
                try:
                    status, transferred_bytes = future.result()
                except Exception as error:  # noqa: BLE001 - aggregate all transfer failures
                    errors.append((key, error))
                else:
                    if status == "downloaded":
                        downloaded += 1
                        transferred += transferred_bytes
                    else:
                        skipped += 1
                progress.update(1)
                progress.set_postfix(
                    downloaded=downloaded,
                    skipped=skipped,
                    failed=len(errors),
                )

    if errors:
        details = "; ".join(f"{key}: {error}" for key, error in errors[:3])
        suffix = f" (and {len(errors) - 3} more)" if len(errors) > 3 else ""
        raise RuntimeError(
            f"S3 download failed for {len(errors)} file(s): {details}{suffix}"
        )
    stats = S3TransferStats(
        prefix=normalized.rstrip("/"),
        files=len(objects),
        downloaded=downloaded,
        skipped=skipped,
        bytes_transferred=transferred,
    )
    LOGGER.info(
        "S3 download complete prefix=%s files=%d downloaded=%d skipped=%d bytes=%d",
        stats.prefix,
        stats.files,
        stats.downloaded,
        stats.skipped,
        stats.bytes_transferred,
    )
    return stats


def _sha256(path: Path) -> str:
    """Hash one local bundle file in bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_inventory(index_root: Path) -> tuple[list[BundleFile], BundleFile]:
    """Inventory only validated index directories and the build report."""

    required = ("visual", "context", "asr_segments")
    files: list[BundleFile] = []
    for name in required:
        directory = index_root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing retrieval index directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                files.append(
                    BundleFile(
                        path=path.relative_to(index_root).as_posix(),
                        size=path.stat().st_size,
                        sha256=_sha256(path),
                    )
                )
    report_path = index_root / "build_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing validated build report: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Invalid build report: {report_path}") from error
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise ValueError("Refusing to upload indexes without a passed build report")
    report_file = BundleFile(
        path="build_report.json",
        size=report_path.stat().st_size,
        sha256=_sha256(report_path),
    )
    if not files:
        raise ValueError("No retrieval index files found")
    return sorted(files, key=lambda item: item.path), report_file


def _put_verified(client: Any, bucket: str, key: str, body: bytes) -> None:
    """Write one small JSON pointer and verify its remote byte count."""

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    response = client.head_object(Bucket=bucket, Key=key)
    if int(response["ContentLength"]) != len(body):
        raise OSError(f"Uploaded size mismatch for s3://{bucket}/{key}")


def publish_retrieval_bundle(
    client: Any,
    bucket: str,
    index_root: Path | str,
    output_prefix: str,
    *,
    workers: int = 8,
) -> RetrievalBundlePublication:
    """Upload a passed bundle under an immutable version and advance latest.

    Index files are uploaded first, followed by ``_SUCCESS.json`` and finally
    ``latest.json``. Readers that follow ``latest.json`` therefore never see a
    pointer to a partially uploaded bundle.
    """

    if workers <= 0:
        raise ValueError("S3 upload workers must be positive")
    root = Path(index_root).expanduser().resolve()
    normalized = _prefix(output_prefix).rstrip("/")
    files, report_file = _bundle_inventory(root)
    inventory = [asdict(item) for item in [*files, report_file]]
    bundle_id = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    version_prefix = f"{normalized}/versions/{bundle_id}"

    def upload(item: BundleFile) -> None:
        local = root / item.path
        key = f"{version_prefix}/{item.path}"
        client.upload_file(str(local), bucket, key)
        response = client.head_object(Bucket=bucket, Key=key)
        if int(response["ContentLength"]) != item.size:
            raise OSError(f"Uploaded size mismatch for s3://{bucket}/{key}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(upload, item) for item in files]
        with tqdm(
            total=len(futures),
            desc="Uploading retrieval indexes",
            unit="file",
            dynamic_ncols=True,
        ) as progress:
            for future in as_completed(futures):
                future.result()
                progress.update(1)

    # Keep the report inside the immutable version before writing completion.
    # The report is the contract consumed by pull/serving validators; a
    # completion marker must never advertise a version that lacks it.
    upload(report_file)

    completion = {
        "schema_version": "retrieval-index-bundle-v1",
        "status": "passed",
        "bundle_id": bundle_id,
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": sum(item["size"] for item in inventory),
    }
    completion_bytes = (json.dumps(completion, indent=2, sort_keys=True) + "\n").encode()
    completion_key = f"{version_prefix}/_SUCCESS.json"
    _put_verified(client, bucket, completion_key, completion_bytes)

    report_key = f"{version_prefix}/{report_file.path}"
    latest = {
        "schema_version": "retrieval-index-latest-v1",
        "status": "passed",
        "bucket": bucket,
        "bundle_id": bundle_id,
        "version_prefix": version_prefix,
        "completion_key": completion_key,
        "report_key": report_key,
        "indexes": {
            name: f"{version_prefix}/{name}"
            for name in ("visual", "context", "asr_segments")
        },
    }
    latest_bytes = (json.dumps(latest, indent=2, sort_keys=True) + "\n").encode()
    latest_key = f"{normalized}/latest.json"
    _put_verified(client, bucket, latest_key, latest_bytes)

    publication = RetrievalBundlePublication(
        bucket=bucket,
        bundle_id=bundle_id,
        version_prefix=version_prefix,
        latest_key=latest_key,
        file_count=len(inventory),
        total_bytes=sum(item["size"] for item in inventory),
    )
    LOGGER.info(
        "S3 retrieval bundle published bucket=%s version=%s files=%d bytes=%d",
        publication.bucket,
        publication.version_prefix,
        publication.file_count,
        publication.total_bytes,
    )
    return publication
