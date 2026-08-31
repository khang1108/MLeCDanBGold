"""Deterministic frame-context artifact contract."""

from __future__ import annotations

from pydantic import Field

from offline.contracts import ContractModel, NonEmptyString


class FrameContext(ContractModel):
    """Deterministic Caption/OCR/Object context derived for one canonical frame."""

    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    caption_text: str | None = None
    ocr_text: str | None = None
    object_summary: str | None = None
    context_text: str | None = None
    caption_available: bool = False
    ocr_quality: float = Field(default=0.0, ge=0, le=1)
    object_count: int = Field(default=0, ge=0)
    context_version: NonEmptyString
    caption_version: NonEmptyString
    ocr_version: NonEmptyString
    object_version: NonEmptyString
    frame_store_id: NonEmptyString | None = None


__all__ = ["FrameContext"]
