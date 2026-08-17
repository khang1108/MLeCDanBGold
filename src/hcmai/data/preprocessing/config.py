"""Cấu hình cho Video Preprocessing.

Chứa các thông số cấu hình và ngưỡng quyết định (thresholds) cho quá trình tiền xử lý video.

Các thiết lập chính:
1. Kích thước (Resolutions): Kích thước ảnh dùng để chạy TransNet, GEBD và lưu trữ.
2. Ngưỡng lọc (Thresholds): Điểm số tối thiểu của thuật toán Shot/Event để giữ lại một frame ứng viên.
3. Tần suất lấy mẫu: Khoảng cách thời gian tối đa (max gap) để đảm bảo không bỏ sót ngữ cảnh."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


def _s3_prefix(value: str) -> str:
    """Normalize one relative S3 prefix without accepting URI/path traversal."""

    normalized = value.strip().strip("/")
    if not normalized or value.strip().startswith("s3://"):
        raise ValueError("S3 prefixes must be non-empty bucket-relative keys")
    if "\\" in normalized or any(
        part in {"", ".", ".."} for part in normalized.split("/")
    ):
        raise ValueError("S3 prefixes must not contain path traversal")
    return normalized


class S3PreprocessingConfig(BaseModel):
    """Offline S3 transport for raw videos and versioned frame artifacts."""

    bucket: str = Field(min_length=3)
    videos_prefix: str = "videos"
    artifacts_prefix: str = "artifacts"
    smoke_artifacts_prefix: str = "artifacts/smoke"
    region: str | None = None
    endpoint_url: str | None = None
    staging_root: Path | None = None
    cache_root: Path | None = None
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    read_timeout_seconds: float = Field(default=300.0, gt=0)
    max_attempts: int = Field(default=4, ge=1, le=10)

    @field_validator("bucket")
    @classmethod
    def normalize_bucket(cls, value: str) -> str:
        bucket = value.strip()
        if len(bucket) < 3 or bucket.startswith("s3://") or "/" in bucket:
            raise ValueError("bucket must be a plain S3 bucket name")
        return bucket

    @field_validator(
        "videos_prefix", "artifacts_prefix", "smoke_artifacts_prefix"
    )
    @classmethod
    def normalize_prefix(cls, value: str) -> str:
        return _s3_prefix(value)

    def artifacts_prefix_for_run(self, limit: int | None) -> str:
        """Keep smoke-test publication pointers outside the full corpus."""

        if limit is None:
            return self.artifacts_prefix
        return f"{self.smoke_artifacts_prefix}/limit-{limit}"


class PreprocessingConfig(BaseModel):
    """Essential settings for the full frame preprocessing pipeline."""

    videos_root: Path | None = None
    s3: S3PreprocessingConfig | None = None
    output_root: Path
    transnet_repo: Path = Path("/models/TransNetV2")
    transnet_weights: Path = Path("/models/TransNetV2/inference/transnetv2-weights")
    efficientgebd_repo: Path = Path("/models/EfficientGEBD")
    efficientgebd_config: Path = Path("/models/EfficientGEBD/model_config.yaml")
    efficientgebd_checkpoint: Path = Path("/models/EfficientGEBD/model_best.pth")
    device: str = "cuda"
    dino_model: str = "facebook/dinov2-small"
    dino_revision: str | None = None
    dino_dtype: str = "float16"
    dino_batch_size: int = Field(default=64, gt=0)
    max_video_workers: int = Field(default=3, ge=1)
    efficientgebd_sample_fps: float = Field(default=10.0, gt=0)
    efficientgebd_resolution: int = Field(default=224, gt=0)
    efficientgebd_sequence_length: int = Field(default=100, gt=0)
    efficientgebd_overlap: int = Field(default=20, ge=0)
    motion_threshold: float = Field(default=0.012, ge=0)
    shot_threshold: float = Field(default=0.5, ge=0, le=1)
    event_threshold: float = Field(default=0.5, ge=0, le=1)
    minimum_gap_ms: int = Field(default=500, gt=0)
    maximum_gap_ms: int = Field(default=2_000, gt=0)
    dedup_similarity: float = Field(default=0.985, ge=-1, le=1)
    image_quality: int = Field(default=92, ge=1, le=100)

    @model_validator(mode="after")
    def validate_gaps(self) -> PreprocessingConfig:
        """Keep the dynamic gap range ordered."""

        if self.minimum_gap_ms > self.maximum_gap_ms:
            raise ValueError("minimum_gap_ms must not exceed maximum_gap_ms")
        if self.efficientgebd_overlap >= self.efficientgebd_sequence_length:
            raise ValueError(
                "efficientgebd_overlap must be shorter than sequence length"
            )
        if (self.videos_root is None) == (self.s3 is None):
            raise ValueError("configure exactly one of videos_root or s3")
        return self

    @property
    def work_root(self) -> Path:
        """Return the private checkpoint directory beside FrameStore."""

        return self.output_root.parent / f".{self.output_root.name}_preprocessing_work"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreprocessingConfig:
        """Load the preprocessing section with optional GPU overrides."""

        with Path(path).open(encoding="utf-8") as handle:
            values: dict[str, Any] = yaml.safe_load(handle) or {}
        config = dict(values.get("preprocessing", values))
        overrides = {
            "device": os.getenv("HCMAI_PREPROCESSING_DEVICE"),
            "dino_dtype": os.getenv("HCMAI_PREPROCESSING_DINO_DTYPE"),
        }
        config.update({key: value for key, value in overrides.items() if value})
        return cls.model_validate(config)
