"""Thin HTTP contracts for the KIS search endpoint.

This module owns the public request/response boundary for KIS. It does not own
query splitting, temporal alignment, or evidence retrieval logic.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .latency import SearchLatency

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class SearchRequest(BaseModel):
    """Public KIS request with only the raw query and desired path count."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: NonEmptyString = Field(max_length=1_000)
    top_k: int = Field(default=20, ge=1)


class SearchResultMetadata(BaseModel):
    """Representative-frame metadata duplicated for frontend simplicity."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = None
    caption: str | None = None
    ocr: str | None = None
    objects: list[str] = Field(default_factory=list)
    asr: str | None = None


class SearchResult(BaseModel):
    """One ranked KIS result with its retained aligned path evidence."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    score: float
    frame_ids: list[NonEmptyString] = Field(min_length=1)
    timestamps_ms: list[int] = Field(min_length=1)
    thumbnail_urls: list[NonEmptyString] = Field(min_length=1)
    frame_url: NonEmptyString
    thumbnail_url: NonEmptyString
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

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: NonEmptyString
    events: list[NonEmptyString] = Field(min_length=1)
    results: list[SearchResult] = Field(default_factory=list)
    latency: SearchLatency

    @model_validator(mode="after")
    def validate_result_paths(self) -> Self:
        """Ensure every returned path aligns exactly once to each event."""

        expected_length = len(self.events)
        if any(len(result.frame_ids) != expected_length for result in self.results):
            raise ValueError("each result must contain one aligned frame per event")

        return self
