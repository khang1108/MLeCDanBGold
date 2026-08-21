"""Tích hợp S3 cho quá trình Preprocessing (Tiền xử lý).

Hỗ trợ tương tác với S3 để tải video nguồn và lưu kết quả FrameStore.

Các tính năng chính:
1. Tải Video (Download): Kéo file video gốc từ bucket S3 xuống ổ cứng cục bộ (cache) để xử lý offline.
2. Upload Artifacts: Đẩy các frames hình ảnh và metadata đã trích xuất lên lại S3 an toàn.
3. Publish FrameStore: Đồng bộ hoá thư mục FrameStore và metadata (dạng parquet) thành bản release bất biến."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hcmai.data.preprocessing.config import (
    PreprocessingConfig,
    S3PreprocessingConfig,
)
from hcmai.data.preprocessing.models import EfficientGEBDDetector, TransNetDetector
from hcmai.data.preprocessing.prepare import (
    _config_hash,
    _finalize_frame_store,
    _limit_config,
    _model_fingerprints,
    _prepare_video,
    _write_json,
)
from hcmai.data.preprocessing.selection import DinoEncoder
from hcmai.data.s3 import S3VideoObject as _S3VideoObject
from hcmai.data.s3 import (
    create_s3_client,
    list_video_objects,
    staged_video,
)
from hcmai.data.stores import FrameStore

S3VideoObject = _S3VideoObject


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """Content identity of one file in a completed FrameStore bundle."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frame_store_bundle(output_root: Path) -> list[ArtifactFile]:
    """Validate canonical metadata/images and inventory every public file."""

    output_root = output_root.expanduser().resolve()
    frames_path = output_root / "frames.parquet"
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing FrameStore manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    store = FrameStore(frames_path)
    if int(manifest.get("frame_count", -1)) != len(store):
        raise ValueError("FrameStore manifest frame_count does not match metadata")
    video_ids: set[str] = set()
    public_paths = {frames_path, manifest_path}
    source_manifest = output_root / "source-manifest.json"
    if source_manifest.is_file():
        public_paths.add(source_manifest)
    root = output_root.resolve()
    for frame in store.iter_frames():
        video_ids.add(frame.video_id)
        image = (output_root / frame.image_path).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise FileNotFoundError(
                f"Missing or invalid canonical frame image: {frame.image_path}"
            )
        public_paths.add(image)
    if int(manifest.get("video_count", -1)) != len(video_ids):
        raise ValueError("FrameStore manifest video_count does not match metadata")
    files = sorted(public_paths)
    return [
        ArtifactFile(
            path=path.relative_to(output_root).as_posix(),
            size=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path in files
    ]


def _json_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _verify_remote_size(client: Any, bucket: str, key: str, size: int) -> None:
    response = client.head_object(Bucket=bucket, Key=key)
    if int(response["ContentLength"]) != size:
        raise OSError(f"Uploaded size mismatch for s3://{bucket}/{key}")


def publish_frame_store(
    client: Any,
    config: S3PreprocessingConfig,
    output_root: Path,
) -> S3Publication:
    """Publish an immutable bundle, then atomically advance ``latest.json``."""

    output_root = output_root.expanduser().resolve()
    files = validate_frame_store_bundle(output_root)
    inventory = [asdict(item) for item in files]
    bundle_id = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    version_prefix = f"{config.artifacts_prefix}/versions/{bundle_id}"
    for item in files:
        key = f"{version_prefix}/{item.path}"
        client.upload_file(str(output_root / item.path), config.bucket, key)
        _verify_remote_size(client, config.bucket, key, item.size)

    completion = _json_bytes({
        "bundle_id": bundle_id,
        "files": inventory,
        "file_count": len(files),
        "total_bytes": sum(item.size for item in files),
    })
    completion_key = f"{version_prefix}/_SUCCESS.json"
    client.put_object(
        Bucket=config.bucket,
        Key=completion_key,
        Body=completion,
        ContentType="application/json",
    )
    _verify_remote_size(client, config.bucket, completion_key, len(completion))

    latest_key = f"{config.artifacts_prefix}/latest.json"
    latest = _json_bytes({
        "bucket": config.bucket,
        "bundle_id": bundle_id,
        "completion_key": completion_key,
        "frames_key": f"{version_prefix}/frames.parquet",
        "version_prefix": version_prefix,
    })
    client.put_object(
        Bucket=config.bucket,
        Key=latest_key,
        Body=latest,
        ContentType="application/json",
    )
    _verify_remote_size(client, config.bucket, latest_key, len(latest))
    return S3Publication(
        bucket=config.bucket,
        version_prefix=version_prefix,
        latest_key=latest_key,
        bundle_id=bundle_id,
        file_count=len(files),
        total_bytes=sum(item.size for item in files),
    )


def prepare_frame_store_from_s3(
    config: PreprocessingConfig,
    *,
    shot_detector: Any | None = None,
    event_detector: Any | None = None,
    encoder: Any | None = None,
    client: Any | None = None,
    resume: bool = True,
    limit: int | None = None,
) -> Path:
    """Stage S3 videos one at a time, preprocess, and publish artifacts."""

    if config.s3 is None or config.videos_root is not None:
        raise ValueError("S3 preprocessing requires only the s3 source")
    config = _limit_config(config, limit)
    if limit is not None:
        config = config.model_copy(update={
            "s3": config.s3.model_copy(update={
                "artifacts_prefix": config.s3.artifacts_prefix_for_run(limit),
            }),
        })
    config.output_root.mkdir(parents=True, exist_ok=True)
    s3_client = client if client is not None else create_s3_client(config.s3)
    sources = list_video_objects(s3_client, config.s3, limit=limit)
    model_fingerprints = _model_fingerprints(config)
    config_hash = _config_hash(config, model_fingerprints)
    checkpoint_resume = resume and config.dino_revision is not None
    shot_detector = shot_detector or TransNetDetector(config)
    event_detector = event_detector or EfficientGEBDDetector(config)
    encoder = encoder or DinoEncoder(config)
    tables = []
    for source in sources:
        with staged_video(s3_client, config.s3, source) as path:
            tables.append(_prepare_video(
                path,
                config,
                shot_detector,
                event_detector,
                encoder,
                checkpoint_resume,
                config_hash,
                source.source_version,
            ))

    _write_json(
        {
            "bucket": config.s3.bucket,
            "videos_prefix": config.s3.videos_prefix,
            "objects": [asdict(source) for source in sources],
        },
        config.output_root / "source-manifest.json",
    )
    output = _finalize_frame_store(
        config,
        tables,
        config_hash=config_hash,
        model_fingerprints=model_fingerprints,
        limited_run=limit is not None,
        resume_enabled=checkpoint_resume,
        source={
            "type": "s3",
            "bucket": config.s3.bucket,
            "videos_prefix": config.s3.videos_prefix,
            "object_count": len(sources),
            "inventory": "source-manifest.json",
        },
    )
    publish_frame_store(s3_client, config.s3, config.output_root)
    return output
