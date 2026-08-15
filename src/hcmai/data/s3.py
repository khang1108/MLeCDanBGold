"""Tiện ích kết nối và thao tác với Amazon S3.

Cung cấp các hàm hỗ trợ việc upload, download, và quản lý các tài nguyên trên S3 bucket.

Các tính năng chính:
1. Upload/Download: Tải video từ S3 về máy cục bộ và đẩy các artifacts (frames, text) lên S3.
2. Quản lý URI: Xử lý các đường dẫn `s3://` và tự động ánh xạ với file cache ở máy cục bộ.
3. Tối ưu truyền tải: Hỗ trợ truyền tải đa luồng (multi-part) cho các file dung lượng lớn."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


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
