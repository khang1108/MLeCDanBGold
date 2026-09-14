"""Thin HTTP contracts for the KIS search endpoint.

This module owns the public request/response boundary for KIS. It does not own
query splitting, temporal alignment, or evidence retrieval logic.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .latency import SearchLatency


class SearchResultMetadata(BaseModel):
    """Representative-frame metadata duplicated for frontend simplicity."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    caption: str | None = None
    ocr: str | None = None
    objects: list[str] = Field(default_factory=list)
    asr: str | None = None


class SearchResult(BaseModel):
    """One ranked KIS result with its retained aligned path evidence."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    score: float
    frame_ids: list[str]
    timestamps_ms: list[int]
    metadata: SearchResultMetadata
    fps: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_alignment_arrays(self) -> Self:
        """Keep aligned path arrays indexed by the same event position."""

        if len(self.frame_ids) != len(self.timestamps_ms):
            raise ValueError("alignment arrays must have equal lengths")

        return self


class ImageSearchResponse(BaseModel):
    """Visual nearest-neighbour results for one uploaded query image."""

    model_config = ConfigDict(extra="forbid")

    results: list[SearchResult] = Field(default_factory=list)
    latency: SearchLatency
