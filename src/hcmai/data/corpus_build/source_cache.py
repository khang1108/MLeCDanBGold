"""Persistent, version-addressed cache for immutable S3 source videos."""

from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from hcmai.data.s3 import S3VideoObject

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CachePopulation:
    """Summary of one bounded cache population pass."""

    paths: tuple[Path, ...]
    downloaded_count: int
    reused_count: int
    total_bytes: int
    duration_seconds: float


class SourceVideoCache:
    """Download each immutable S3 object once and reuse it across resumes."""

    def __init__(self, client: Any, bucket: str, root: Path) -> None:
        self.client = client
        self.bucket = bucket
        self.root = root.expanduser().resolve()

    def path_for(self, source: S3VideoObject) -> Path:
        suffix = Path(source.key).suffix.lower()
        return self.root / source.source_version / f"{source.video_id}{suffix}"

    def is_valid(self, source: S3VideoObject) -> bool:
        path = self.path_for(source)
        if not path.is_file():
            return False
        stat = path.stat()
        return (
            stat.st_size == source.size
            and stat.st_mtime_ns == source.last_modified_ns
        )

    def ensure(self, source: S3VideoObject) -> tuple[Path, bool]:
        target = self.path_for(source)
        if self.is_valid(source):
            return target, False
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.partial")
        partial.unlink(missing_ok=True)
        try:
            self.client.download_file(self.bucket, source.key, str(partial))
            if not partial.is_file() or partial.stat().st_size != source.size:
                raise OSError(
                    f"Downloaded size mismatch for s3://{self.bucket}/{source.key}"
                )
            os.utime(
                partial,
                ns=(source.last_modified_ns, source.last_modified_ns),
            )
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        return target, True

    def populate(
        self,
        sources: list[S3VideoObject],
        *,
        max_workers: int,
        minimum_free_bytes: int,
    ) -> CachePopulation:
        started = perf_counter()
        self.root.mkdir(parents=True, exist_ok=True)
        missing_bytes = sum(
            source.size for source in sources if not self.is_valid(source)
        )
        free = shutil.disk_usage(self.root).free
        required = missing_bytes + minimum_free_bytes
        if missing_bytes > 0 and free < required:
            raise OSError(
                "Insufficient disk for source cache: "
                f"free={free} required={required} "
                f"downloads={missing_bytes} reserve={minimum_free_bytes}"
            )
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="hcmai-s3-cache",
        ) as executor:
            results = list(executor.map(self.ensure, sources))
        paths = tuple(path for path, _ in results)
        downloaded = sum(was_downloaded for _, was_downloaded in results)
        total_bytes = sum(source.size for source in sources)
        duration = perf_counter() - started
        logger.info(
            "Source cache ready: videos=%d downloaded=%d reused=%d bytes=%d seconds=%.1f",
            len(sources),
            downloaded,
            len(sources) - downloaded,
            total_bytes,
            duration,
        )
        return CachePopulation(
            paths=paths,
            downloaded_count=downloaded,
            reused_count=len(sources) - downloaded,
            total_bytes=total_bytes,
            duration_seconds=duration,
        )
