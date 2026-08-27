"""Tiện ích kết nối và thao tác với Amazon S3.

Cung cấp các hàm hỗ trợ việc upload, download, và quản lý các tài nguyên trên S3 bucket.

Các tính năng chính:
1. Upload/Download: Tải video từ S3 về máy cục bộ và đẩy các artifacts (frames, text) lên S3.
2. Quản lý URI: Xử lý các đường dẫn `s3://` và tự động ánh xạ với file cache ở máy cục bộ.
3. Tối ưu truyền tải: Hỗ trợ truyền tải đa luồng (multi-part) cho các file dung lượng lớn."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from tqdm.auto import tqdm

from hcmai.common.utils.io import read_yaml_section

VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
LOGGER = logging.getLogger(__name__)


class S3VideoConfig(Protocol):
    """Structural transport settings shared by all data preparation stages."""

    bucket: str
    videos_prefix: str
    region: str | None
    endpoint_url: str | None
    staging_root: Path | None
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_attempts: int


@dataclass(frozen=True, slots=True)
class S3VideoObject:
    """Stable identity and local-checkpoint metadata for one S3 video."""

    key: str
    size: int
    etag: str
    last_modified_ns: int

    @property
    def video_id(self) -> str:
        return Path(self.key).stem

    @property
    def source_version(self) -> str:
        payload = f"{self.key}\0{self.size}\0{self.etag}\0{self.last_modified_ns}"
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class S3TransferStats:
    """Counts produced by one resumable S3 prefix download."""

    prefix: str
    files: int
    downloaded: int
    skipped: int
    bytes_transferred: int


@dataclass(frozen=True, slots=True)
class BundleFile:
    """Content identity for one file in a retrieval-index bundle."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RetrievalBundlePublication:
    """Immutable S3 location of one validated retrieval-index bundle."""

    bucket: str
    bundle_id: str
    version_prefix: str
    latest_key: str
    file_count: int
    total_bytes: int


def load_s3_config(path: str | Path) -> S3VideoConfig:
    """Load the S3 transport mapping from shared preparation configuration.

    The ``storage`` section contains transport settings under ``s3``.
    Credentials remain in boto3's standard credential chain, while explicit
    HCMAI environment variables may override deployment-specific fields.
    """

    # Import lazily because the legacy corpus-build package initializer imports
    # its pipeline, and that pipeline imports this transport module.
    from hcmai.data.corpus_build.config import S3PreparationConfig

    config_path = Path(path)
    values = read_yaml_section(config_path, "storage")
    storage = values.get("s3", values)
    if not isinstance(storage, dict):
        raise ValueError(f"S3 config requires an 's3' mapping: {config_path}")

    overrides = {
        "bucket": os.getenv("HCMAI_S3_BUCKET"),
        "region": os.getenv("HCMAI_S3_REGION"),
        "endpoint_url": os.getenv("HCMAI_S3_ENDPOINT_URL"),
    }
    normalized = dict(storage)
    normalized.update(
        {name: value for name, value in overrides.items() if value}
    )
    return S3PreparationConfig.model_validate(normalized)


def create_s3_client(config: Any) -> Any:
    """Create an S3 client using boto3's standard credential chain."""

    try:
        import boto3
        from botocore.config import Config
    except ImportError as error:
        raise RuntimeError(
            'boto3 is required; install with: python -m pip install -e ".[s3]"'
        ) from error

    def _val(key: str, default: Any = None) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    region = _val("region") or _val("region_name")
    endpoint_url = _val("endpoint_url") or os.getenv("HCMAI_S3_ENDPOINT_URL")

    arguments: dict[str, Any] = {}
    if region is not None and str(region).strip():
        arguments["region_name"] = str(region).strip()
    if endpoint_url is not None and str(endpoint_url).strip():
        arguments["endpoint_url"] = str(endpoint_url).strip()

    for cred_key in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
        cred_val = _val(cred_key)
        if cred_val is not None and str(cred_val).strip():
            arguments[cred_key] = str(cred_val).strip()

    custom_client_config = _val("client_config")
    if custom_client_config is not None:
        arguments["config"] = custom_client_config
    else:
        connect_timeout = _val("connect_timeout_seconds", 10.0)
        read_timeout = _val("read_timeout_seconds", 300.0)
        max_attempts = _val("max_attempts", 4)
        arguments["config"] = Config(
            connect_timeout=float(connect_timeout),
            read_timeout=float(read_timeout),
            retries={"max_attempts": int(max_attempts), "mode": "standard"},
        )

    return boto3.client("s3", **arguments)


def _last_modified_ns(value: object) -> int:
    if isinstance(value, datetime):
        return round(value.timestamp() * 1_000_000_000)
    raise ValueError("S3 video object is missing LastModified")


def list_video_objects(
    client: Any,
    config: S3VideoConfig,
    *,
    limit: int | None = None,
) -> list[S3VideoObject]:
    """List supported source videos below the configured prefix."""

    prefix = f"{config.videos_prefix}/"
    pages = client.get_paginator("list_objects_v2").paginate(
        Bucket=config.bucket,
        Prefix=prefix,
    )
    objects = sorted(
        (
            S3VideoObject(
                key=str(item["Key"]),
                size=int(item["Size"]),
                etag=str(item.get("ETag", "")).strip('"'),
                last_modified_ns=_last_modified_ns(item.get("LastModified")),
            )
            for page in pages
            for item in page.get("Contents", ())
            if int(item.get("Size", 0)) > 0
            and Path(str(item.get("Key", ""))).suffix.lower()
            in VIDEO_EXTENSIONS
        ),
        key=lambda item: item.key,
    )
    if limit is not None:
        objects = objects[:limit]
    video_ids = [item.video_id for item in objects]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("Video IDs must be unique across the S3 corpus")
    if not objects:
        raise FileNotFoundError(
            f"No supported videos found at s3://{config.bucket}/{prefix}"
        )
    return objects


@contextmanager
def staged_video(
    client: Any,
    config: S3VideoConfig,
    source: S3VideoObject,
) -> Iterator[Path]:
    """Download one video once and retain it for the full consumer scope."""

    staging_root = config.staging_root
    if staging_root is not None:
        staging_root = staging_root.expanduser().resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hcmai-s3-video-",
        dir=staging_root,
    ) as directory:
        path = Path(directory) / Path(source.key).name
        client.download_file(config.bucket, source.key, str(path))
        if not path.is_file() or path.stat().st_size != source.size:
            raise OSError(
                f"Downloaded size mismatch for s3://{config.bucket}/{source.key}"
            )
        os.utime(path, ns=(source.last_modified_ns, source.last_modified_ns))
        yield path


def _normalize_prefix(value: str) -> str:
    """Normalize a bucket-relative S3 prefix and reject unsafe components."""

    normalized = value.strip().strip("/")
    if not normalized or normalized.startswith("s3://"):
        raise ValueError("S3 prefix must be a non-empty bucket-relative key")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError("S3 prefix must not contain path traversal")
    return normalized + "/"


def _relative_object_path(key: str, prefix: str) -> Path:
    """Map one object key below a prefix to a safe local relative path."""

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


def _list_prefix_objects(
    client: Any,
    bucket: str,
    prefix: str,
) -> list[tuple[str, int]]:
    """List non-empty objects below a prefix in deterministic key order."""

    objects: list[tuple[str, int]] = []
    pages = client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket,
        Prefix=prefix,
    )
    for page in pages:
        for item in page.get("Contents", ()):
            key = str(item.get("Key", ""))
            size = int(item.get("Size", 0))
            if key and size > 0 and key != prefix:
                _relative_object_path(key, prefix)
                objects.append((key, size))
    return sorted(objects)


def _download_object(
    client: Any,
    bucket: str,
    key: str,
    size: int,
    prefix: str,
    destination: Path,
) -> tuple[str, int]:
    """Download one object atomically or skip an equal-size local file."""

    target = destination / _relative_object_path(key, prefix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == size:
        return "skipped", 0

    partial = target.with_name(f".{target.name}.part")
    try:
        partial.unlink(missing_ok=True)
        client.download_file(bucket, key, str(partial))
        actual_size = partial.stat().st_size if partial.is_file() else 0
        if actual_size != size:
            raise OSError(
                f"Downloaded size mismatch for s3://{bucket}/{key}: "
                f"expected {size}, got {actual_size}"
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
    workers: int = 8,
    dry_run: bool = False,
) -> S3TransferStats:
    """Download an S3 prefix resumably using bounded parallel file transfers.

    Existing files with the expected byte size are retained. New payloads are
    moved into place atomically, so interruption cannot expose truncated input
    artifacts to an embedding stage.
    """

    if workers <= 0:
        raise ValueError("S3 download workers must be positive")
    normalized = _normalize_prefix(prefix)
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    objects = _list_prefix_objects(client, bucket, normalized)
    if not objects:
        raise FileNotFoundError(f"No objects found at s3://{bucket}/{normalized}")
    if dry_run:
        return S3TransferStats(
            prefix=normalized.rstrip("/"),
            files=len(objects),
            downloaded=0,
            skipped=0,
            bytes_transferred=0,
        )

    downloaded = skipped = transferred = 0
    errors: list[tuple[str, Exception]] = []

    def transfer(item: tuple[str, int]) -> tuple[str, int]:
        return _download_object(
            client,
            bucket,
            item[0],
            item[1],
            normalized,
            destination_path,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[tuple[str, int]], tuple[str, int]] = {
            pool.submit(transfer, item): item for item in objects
        }
        with tqdm(
            total=len(futures),
            desc=f"Downloading {normalized.rstrip('/')}",
            unit="file",
            dynamic_ncols=True,
        ) as progress:
            for future in as_completed(futures):
                key, _ = futures[future]
                try:
                    status, transferred_bytes = future.result()
                except Exception as error:  # noqa: BLE001 - report all failed keys
                    errors.append((key, error))
                else:
                    if status == "downloaded":
                        downloaded += 1
                        transferred += transferred_bytes
                    else:
                        skipped += 1
                progress.update(1)

    if errors:
        details = "; ".join(f"{key}: {error}" for key, error in errors[:3])
        suffix = f" (and {len(errors) - 3} more)" if len(errors) > 3 else ""
        raise RuntimeError(
            f"S3 download failed for {len(errors)} file(s): {details}{suffix}"
        )
    return S3TransferStats(
        prefix=normalized.rstrip("/"),
        files=len(objects),
        downloaded=downloaded,
        skipped=skipped,
        bytes_transferred=transferred,
    )


def _sha256_file(path: Path) -> str:
    """Hash one artifact without loading it fully into host memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retrieval_bundle_inventory(
    index_root: Path,
) -> tuple[list[BundleFile], BundleFile]:
    """Inventory the three required indexes and their passed build report."""

    files: list[BundleFile] = []
    for name in ("visual", "context", "asr_segments"):
        directory = index_root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing retrieval index directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                files.append(
                    BundleFile(
                        path=path.relative_to(index_root).as_posix(),
                        size=path.stat().st_size,
                        sha256=_sha256_file(path),
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
        sha256=_sha256_file(report_path),
    )
    if not files:
        raise ValueError("No retrieval index files found")
    return sorted(files, key=lambda item: item.path), report_file


def _put_verified_json(
    client: Any,
    bucket: str,
    key: str,
    body: bytes,
) -> None:
    """Upload a small JSON contract and verify its remote byte count."""

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
    """Publish validated indexes immutably and advance ``latest.json`` last."""

    if workers <= 0:
        raise ValueError("S3 upload workers must be positive")
    root = Path(index_root).expanduser().resolve()
    normalized = _normalize_prefix(output_prefix).rstrip("/")
    files, report_file = _retrieval_bundle_inventory(root)
    inventory = [asdict(item) for item in [*files, report_file]]
    bundle_id = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    version_prefix = f"{normalized}/versions/{bundle_id}"

    def upload(item: BundleFile) -> None:
        key = f"{version_prefix}/{item.path}"
        client.upload_file(str(root / item.path), bucket, key)
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

    # The report is part of the immutable version and must exist before the
    # completion marker allows any consumer to accept the bundle.
    upload(report_file)
    total_bytes = sum(int(item["size"]) for item in inventory)
    completion = {
        "schema_version": "retrieval-index-bundle-v1",
        "status": "passed",
        "bundle_id": bundle_id,
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": total_bytes,
    }
    completion_bytes = (
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    ).encode()
    completion_key = f"{version_prefix}/_SUCCESS.json"
    _put_verified_json(client, bucket, completion_key, completion_bytes)

    latest = {
        "schema_version": "retrieval-index-latest-v1",
        "status": "passed",
        "bucket": bucket,
        "bundle_id": bundle_id,
        "version_prefix": version_prefix,
        "completion_key": completion_key,
        "report_key": f"{version_prefix}/{report_file.path}",
        "indexes": {
            name: f"{version_prefix}/{name}"
            for name in ("visual", "context", "asr_segments")
        },
    }
    latest_bytes = (json.dumps(latest, indent=2, sort_keys=True) + "\n").encode()
    latest_key = f"{normalized}/latest.json"
    _put_verified_json(client, bucket, latest_key, latest_bytes)

    publication = RetrievalBundlePublication(
        bucket=bucket,
        bundle_id=bundle_id,
        version_prefix=version_prefix,
        latest_key=latest_key,
        file_count=len(inventory),
        total_bytes=total_bytes,
    )
    LOGGER.info(
        "Published retrieval bundle bucket=%s version=%s files=%d bytes=%d",
        publication.bucket,
        publication.version_prefix,
        publication.file_count,
        publication.total_bytes,
    )
    return publication
