"""Đóng gói và xuất bản (Publish) Corpus.

Module này chịu trách nhiệm đẩy dữ liệu corpus lên hệ thống lưu trữ sau khi đã build xong.

Các tính năng chính:
1. Đóng gói Artifacts: Nén hoặc gom nhóm các file metadata và media vào cấu trúc chuẩn.
2. Cloud Upload: Đẩy dữ liệu lên kho lưu trữ đám mây (như Amazon S3) hoặc thư mục mạng tập trung.
3. Registry Update: Đánh dấu phiên bản corpus mới nhất để hệ thống online dễ dàng nhận diện và load."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
from hcmai.data.corpus_build.pipeline import PreparationPaths
from hcmai.data.preprocessing.s3 import ArtifactFile, S3Publication, _json_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_remote_size(client: Any, bucket: str, key: str, size: int) -> None:
    response = client.head_object(Bucket=bucket, Key=key)
    if int(response["ContentLength"]) != size:
        raise OSError(f"Uploaded size mismatch for s3://{bucket}/{key}")


def inventory_artifacts(artifacts_root: Path) -> list[ArtifactFile]:
    """Inventory all valid files within the complete artifacts tree."""
    files: list[Path] = []
    # Using sorted list for deterministic bundle ID
    for path in sorted(artifacts_root.rglob("*")):
        if path.is_file():
            files.append(path)
            
    if not files:
        raise ValueError(f"No artifacts found in {artifacts_root}")

    return [
        ArtifactFile(
            path=path.relative_to(artifacts_root).as_posix(),
            size=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path in files
    ]


def publish_run_artifacts(
    client: Any,
    paths: PreparationPaths,
    config: S3CorpusPreparationConfig,
    limit: int | None,
) -> S3Publication:
    """Upload complete artifacts to S3 and atomically advance latest.json."""
    storage = config.preprocessing.s3
    if storage is None:
        raise ValueError("Cannot publish run artifacts without S3 config")

    artifacts_root = paths.artifacts_root.resolve()
    files = inventory_artifacts(artifacts_root)
    inventory = [asdict(item) for item in files]
    
    # Bundle ID is a stable hash of the entire inventory contents
    bundle_id = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]

    # Smoke runs use an isolated prefix
    base_prefix = config.smoke_artifacts_prefix if limit is not None else config.full_artifacts_prefix
    version_prefix = f"{base_prefix}/versions/{bundle_id}"
    
    for item in files:
        key = f"{version_prefix}/{item.path}"
        client.upload_file(str(artifacts_root / item.path), storage.bucket, key)
        _verify_remote_size(client, storage.bucket, key, item.size)

    # Publish _SUCCESS.json
    completion = _json_bytes({
        "bundle_id": bundle_id,
        "files": inventory,
        "file_count": len(files),
        "total_bytes": sum(item.size for item in files),
    })
    completion_key = f"{version_prefix}/_SUCCESS.json"
    client.put_object(
        Bucket=storage.bucket,
        Key=completion_key,
        Body=completion,
        ContentType="application/json",
    )
    _verify_remote_size(client, storage.bucket, completion_key, len(completion))

    # Identify primary entrypoints
    def _find_path(suffix: str) -> str | None:
        for item in inventory:
            if item["path"].endswith(suffix):
                return f"{version_prefix}/{item['path']}"
        return None

    # Advance latest.json
    latest_key = f"{base_prefix}/latest.json"
    latest = _json_bytes({
        "bucket": storage.bucket,
        "bundle_id": bundle_id,
        "completion_key": completion_key,
        "version_prefix": version_prefix,
        "frames_key": _find_path("frame_store/frames.parquet"),
        "visual_index_key": _find_path("indexes/visual/dense.index"),
        "caption_index_key": _find_path("indexes/caption/dense.index"),
        "ocr_index_key": _find_path("indexes/ocr/dense.index"),
        "asr_index_key": _find_path("indexes/asr/dense.index"),
    })
    client.put_object(
        Bucket=storage.bucket,
        Key=latest_key,
        Body=latest,
        ContentType="application/json",
    )
    _verify_remote_size(client, storage.bucket, latest_key, len(latest))

    return S3Publication(
        bucket=storage.bucket,
        version_prefix=version_prefix,
        latest_key=latest_key,
        bundle_id=bundle_id,
        file_count=len(files),
        total_bytes=sum(item.size for item in files),
    )
