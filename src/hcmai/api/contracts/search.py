"""Thin HTTP contracts for the KIS search endpoint.

This module owns the public request/response boundary for KIS. It does not own
query splitting, temporal alignment, or evidence retrieval logic.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .latency import SearchLatency


_NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class SearchRequest(BaseModel):
    """Public KIS request with only the raw query and desired path count."""

    model_config = ConfigDict(extra="forbid")

    query: _NonBlankString
    top_k: int = Field(default=20, ge=1)


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
    thumbnail_urls: list[str]
    frame_url: str
    thumbnail_url: str
    metadata: SearchResultMetadata

    @model_validator(mode="after")
    def validate_alignment_arrays(self) -> Self:
        """Keep aligned path arrays indexed by the same event position."""

        if not (
            len(self.frame_ids)
            == len(self.timestamps_ms)
            == len(self.thumbnail_urls)
        ):
            raise ValueError("alignment arrays must have equal lengths")

        return self


class SearchResponse(BaseModel):
    """Complete KIS response with deterministic event alignment paths."""

    model_config = ConfigDict(extra="forbid")

    query: str
    events: list[str]
    results: list[SearchResult] = Field(default_factory=list)
    latency: SearchLatency

    @model_validator(mode="after")
    def validate_result_paths(self) -> Self:
        """Ensure every returned path aligns exactly once to each event."""

        expected_length = len(self.events)
        if any(len(result.frame_ids) != expected_length for result in self.results):
            raise ValueError("each result must contain one aligned frame per event")

        return self
