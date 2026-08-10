"""Configuration for adaptive frame preprocessing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class PreprocessingConfig(BaseModel):
    """Essential settings for the full frame preprocessing pipeline."""

    videos_root: Path
    output_root: Path
    transnet_repo: Path
    transnet_weights: Path
    efficientgebd_repo: Path
    efficientgebd_config: Path
    efficientgebd_checkpoint: Path
    device: str = "cuda"
    dino_model: str = "facebook/dinov2-small"
    dino_dtype: str = "float16"
    dino_batch_size: int = Field(default=16, gt=0)
    efficientgebd_sample_fps: float = Field(default=10.0, gt=0)
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
        return self

    @property
    def work_root(self) -> Path:
        """Return the private checkpoint directory beside FrameStore."""

        return self.output_root.parent / ".preprocessing_work"

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
