"""Configuration for adaptive frame preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PreprocessingConfig(BaseSettings):
    """Keep every runtime choice in one serializable configuration."""

    model_config = SettingsConfigDict(
        env_prefix="HCMAI_PREPROCESSING_",
        env_nested_delimiter="__",
    )

    videos_root: Path
    output_root: Path
    analysis_width: int = Field(default=320, gt=0)
    analysis_height: int = Field(default=180, gt=0)
    video_extensions: tuple[str, ...] = (
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",
        ".m4v",
    )
    motion_threshold: float = Field(default=0.012, ge=0)
    minimum_gap_ms: int = Field(default=500, gt=0)
    maximum_gap_ms: int = Field(default=2_000, gt=0)
    burst_radius_ms: int = Field(default=500, ge=0)
    burst_step_ms: int = Field(default=200, gt=0)
    transnet_enabled: bool = True
    transnet_repo: Path | None = None
    transnet_weights: Path | None = None
    shot_threshold: float = Field(default=0.5, ge=0, le=1)
    efficientgebd_enabled: bool = False
    efficientgebd_repo: Path | None = None
    efficientgebd_config: Path | None = None
    efficientgebd_checkpoint: Path | None = None
    efficientgebd_device: str = "cuda"
    efficientgebd_sample_fps: float = Field(default=10.0, gt=0)
    efficientgebd_overlap_frames: int = Field(default=20, ge=0)
    event_threshold: float = Field(default=0.5, ge=0, le=1)
    dino_enabled: bool = False
    dino_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    dino_device: str = "cuda"
    dino_dtype: str = "float16"
    dino_batch_size: int = Field(default=16, gt=0)
    dedup_window_ms: int = Field(default=1_000, ge=0)
    dedup_similarity: float = Field(default=0.985, ge=-1, le=1)
    dedup_motion_threshold: float = Field(default=0.008, ge=0)
    image_quality: int = Field(default=92, ge=1, le=100)
    resume: bool = True
    limit: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_gaps(self) -> PreprocessingConfig:
        """Keep the dynamic gap range ordered."""

        if self.minimum_gap_ms > self.maximum_gap_ms:
            raise ValueError("minimum_gap_ms must not exceed maximum_gap_ms")
        return self

    @property
    def work_root(self) -> Path:
        """Return the private checkpoint directory beside FrameStore."""

        return self.output_root.parent / ".preprocessing_work"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreprocessingConfig:
        """Load either a root config or its ``preprocessing`` section."""

        with Path(path).open(encoding="utf-8") as handle:
            values: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls.model_validate(values.get("preprocessing", values))
