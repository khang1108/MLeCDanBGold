"""Public HTTP contracts for the inspectable keyframe catalog."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CatalogTranscriptSegment(BaseModel):
    """Timeline transcript projection included in a catalog response."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    segment_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    segment_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)


class FrameCatalogEntry(BaseModel):
    """One keyframe and its lightweight, inspectable evidence projection."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    video_id: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    frame_idx: int = Field(ge=0)
    caption: str | None = None
    ocr: str | None = None
    objects: dict[str, int] | None = None
    title: str | None = None
    asr_segments: list[CatalogTranscriptSegment] = Field(default_factory=list)
    video_url: str | None = None


__all__ = ["CatalogTranscriptSegment", "FrameCatalogEntry"]
