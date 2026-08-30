"""Đóng gói và xuất bản (Publish) Corpus.

Module này chịu trách nhiệm đẩy dữ liệu corpus lên hệ thống lưu trữ sau khi đã build xong.

Các tính năng chính:
1. Đóng gói Artifacts: Nén hoặc gom nhóm các file metadata và media vào cấu trúc chuẩn.
2. Cloud Upload: Đẩy dữ liệu lên kho lưu trữ đám mây (như Amazon S3) hoặc thư mục mạng tập trung.
3. Registry Update: Đánh dấu phiên bản corpus mới nhất để hệ thống online dễ dàng nhận diện và load."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from offline.ingestion.corpus_build.config import S3CorpusPreparationConfig

if TYPE_CHECKING:
    from offline.ingestion.corpus_build.pipeline import PreparationPaths


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """Content identity of one file in a completed corpus bundle."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class S3Publication:
    """Location and size of one immutable published artifact version."""

    bucket: str
    version_prefix: str
    latest_key: str
    bundle_id: str
    file_count: int
    total_bytes: int


def _json_bytes(value: dict[str, object]) -> bytes:
    """Serialize a publication marker deterministically."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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


def publish_group_artifacts(
    client: Any,
    paths: PreparationPaths,
    config: S3CorpusPreparationConfig,
    *,
    group_id: str,
    run_id: str,
    source_manifest: dict[str, Any],
) -> S3Publication:
    """Tải tất cả các artifact của MỘT group lên S3 (không cập nhật biến global latest).
    Các file được lưu vào prefix độc lập (theo group_id và run_id) kèm theo COMMITTED.json
    để đánh dấu group này đã hoàn tất và sẵn sàng cho bước Reduce.
    """

    storage = config.preprocessing.s3
    if storage is None:
        raise ValueError("group publication requires S3 storage")
    
    files = inventory_artifacts(paths.artifacts_root.resolve())
    inventory = [asdict(item) for item in files]
    
    base = config.full_artifacts_prefix.rstrip("/")
    version_prefix = f"{base}/groups/{group_id}/runs/{run_id}"
    
    for item in files:
        key = f"{version_prefix}/{item.path}"
        client.upload_file(str(paths.artifacts_root / item.path), storage.bucket, key)
        _verify_remote_size(client, storage.bucket, key, item.size)

    manifest = _json_bytes({
        "schema_version": "group-artifact-manifest-v1",
        "corpus_revision": config.corpus_revision,
        "group_id": group_id,
        "run_id": run_id,
        "source": source_manifest,
        "files": inventory,
        "file_count": len(files),
        "total_bytes": sum(item.size for item in files),
    })
    
    manifest_key = f"{version_prefix}/manifest.json"
    client.put_object(
        Bucket=storage.bucket,
        Key=manifest_key,
        Body=manifest,
        ContentType="application/json",
    )
    
    _verify_remote_size(client, storage.bucket, manifest_key, len(manifest))
    
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    committed = _json_bytes({
        "schema_version": "group-commit-v1",
        "group_id": group_id,
        "run_id": run_id,
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_sha256,
    })
    
    commit_key = f"{version_prefix}/COMMITTED.json"
    client.put_object(
        Bucket=storage.bucket,
        Key=commit_key,
        Body=committed,
        ContentType="application/json",
    )
    _verify_remote_size(client, storage.bucket, commit_key, len(committed))
    
    return S3Publication(
        bucket=storage.bucket,
        version_prefix=version_prefix,
        latest_key=commit_key,
        bundle_id=run_id,
        file_count=len(files),
        total_bytes=sum(item.size for item in files),
    )
