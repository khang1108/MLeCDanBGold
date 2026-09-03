"""Thin HTTP contracts for the KIS search endpoint.

This module owns the public request/response boundary for KIS. It does not own
query splitting, temporal alignment, or evidence retrieval logic.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT

from .latency import SearchLatency

_NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class SearchRequest(BaseModel):
    """Public KIS request with optional selected retrieval events."""

    model_config = ConfigDict(extra="forbid")

    query: _NonBlankString
    retrieval_events: list[_NonBlankString] | None = Field(
        default=None,
        min_length=1,
        max_length=DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    )
    use_dense: bool = True
    use_bm25: bool = True
    top_k: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        """Require at least one full-corpus temporal evidence source."""

        if not self.use_dense and not self.use_bm25:
            raise ValueError("at least one of use_dense or use_bm25 must be true")
        return self


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


class SearchResponse(BaseModel):
    """Complete KIS response with deterministic event alignment paths."""

    model_config = ConfigDict(extra="forbid")

    query: str
    events: list[str]
    dense_events: list[str] | None = None
    bm25_caption_events: list[str] | None = None
    use_dense: bool = True
    use_bm25: bool = False
    results: list[SearchResult] = Field(default_factory=list)
    latency: SearchLatency

    @model_validator(mode="after")
    def validate_result_paths(self) -> Self:
        """Ensure every returned path aligns exactly once to each event."""

        expected_length = len(self.events)
        if self.use_dense and self.dense_events is None:
            raise ValueError("dense_events are required when Dense is enabled")
        if not self.use_dense and self.dense_events is not None:
            raise ValueError("dense_events must be absent when Dense is disabled")
        if self.dense_events is not None and len(self.dense_events) != expected_length:
            raise ValueError("dense_events must match the original event count")
        if self.use_bm25 and self.bm25_caption_events is None:
            raise ValueError("bm25_caption_events are required when BM25 is enabled")
        if not self.use_bm25 and self.bm25_caption_events is not None:
            raise ValueError("bm25_caption_events must be absent when BM25 is disabled")
        if (
            self.bm25_caption_events is not None
            and len(self.bm25_caption_events) != expected_length
        ):
            raise ValueError("bm25_caption_events must match the original event count")
        if any(len(result.frame_ids) != expected_length for result in self.results):
            raise ValueError("each result must contain one aligned frame per event")

        return self


class ImageSearchResponse(BaseModel):
    """Visual nearest-neighbour results for one uploaded query image."""

    model_config = ConfigDict(extra="forbid")

    results: list[SearchResult] = Field(default_factory=list)
    latency: SearchLatency
