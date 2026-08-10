"""Configuration owned by OCR enrichment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OCRConfig:
    """Settings identifying one reproducible OCR enrichment."""

    enabled: bool = True
    backend: str = "florence2"
    checkpoint: str | None = "florence-community/Florence-2-base-ft"
    revision: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1
    image_size: int | None = 768
    enrichment_version: str = "florence2_ocr_v1"
    dataset_version: str = "unknown"

    def __post_init__(self) -> None:
        if self.batch_size < 1 or (self.image_size is not None and self.image_size < 1):
            raise ValueError("batch_size and image_size must be positive")

    @property
    def model_name(self) -> str:
        """Return the canonical backend identity."""
        return self.checkpoint or self.backend
