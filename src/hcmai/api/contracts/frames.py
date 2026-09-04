"""Public HTTP contracts for the inspectable keyframe catalog."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .search import SearchResultMetadata


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


class FrameInspectionResponse(BaseModel):
    """Canonical keyframe evidence resolved for a manually opened video moment.

    ``requested_timestamp_ms`` remains the browser seek target. ``timestamp_ms``
    and every identity field belong to the selected canonical keyframe and must
    remain internally consistent with ``frame_id``.
    """

    model_config = ConfigDict(extra="forbid")

    requested_timestamp_ms: int = Field(ge=0)
    frame_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    fps: float | None = Field(default=None, gt=0)
    metadata: SearchResultMetadata


__all__ = [
    "CatalogTranscriptSegment",
    "FrameCatalogEntry",
    "FrameInspectionResponse",
]
