"""HTTP contracts for direct literal search over loaded frame evidence."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


_NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
_OptionalScope = Annotated[
    str | None,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class FilterRequest(BaseModel):
    """Request one page of literal text matches with optional frame scope."""

    model_config = ConfigDict(extra="forbid")

    query: _NonBlankString
    folder_id: _OptionalScope = None
    video_id: _OptionalScope = None
    frames_per_pages: int = Field(default=12, ge=1, le=48)
    page_id: int = Field(default=1, ge=1)


class FilterResult(BaseModel):
    """One canonical frame with complete display metadata and matched text."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    fps: float | None = None
    folder_id: str
    title: str | None = None
    caption: str | None = None
    ocr: str | None = None
    objects: dict[str, int] = Field(default_factory=dict)
    asr: str | None = None
    matches: dict[str, str] = Field(default_factory=dict)


class FilterResponse(BaseModel):
    """Paginated literal matches and evidence sources loaded at startup."""

    model_config = ConfigDict(extra="forbid")

    page_id: int
    frames_per_pages: int
    total_pages: int
    total_results: int
    available_sources: list[str]
    results: list[FilterResult] = Field(default_factory=list)


__all__ = ["FilterRequest", "FilterResponse", "FilterResult"]
