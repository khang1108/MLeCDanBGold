"""Configuration for one reproducible OCR enrichment run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OCRConfig:
    """Settings identifying one reproducible OCR enrichment."""

    enabled: bool = True
    backend: str = "florence2"
    checkpoint: str | None = "florence-community/Florence-2-base-ft"
    revision: str | None = "0b03b6f15a4a211370fb204aee4e7dd48887ea37"
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = 32
    image_size: int | None = 768
    enrichment_version: str = "florence2_ocr_v1"
    dataset_version: str = "unknown"
    min_region_confidence: float = 0.0
    min_context_quality: float = 0.5
    artifact_version: str = "ocr-v1"

    def __post_init__(self) -> None:
        if self.batch_size < 1 or (self.image_size is not None and self.image_size < 1):
            raise ValueError("batch_size and image_size must be positive")
        if not 0.0 <= self.min_region_confidence <= 1.0:
            raise ValueError("min_region_confidence must be in [0, 1]")
        if not 0.0 <= self.min_context_quality <= 1.0:
            raise ValueError("min_context_quality must be in [0, 1]")

    @property
    def model_name(self) -> str:
        """Return the canonical backend identity."""
        return self.checkpoint or self.backend
