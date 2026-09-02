"""Thin HTTP contracts for the TRAKE alignment endpoint.

This module owns only the public ordered-event request/response payloads. It
does not own event parsing, temporal search, or corpus materialization.
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


class TRAKERequest(BaseModel):
    """Public TRAKE request with explicit ordered events."""

    model_config = ConfigDict(extra="forbid")

    events: list[_NonBlankString] = Field(
        min_length=1,
        max_length=DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    )
    retrieval_events: list[_NonBlankString] | None = Field(
        default=None,
        min_length=1,
        max_length=DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    )
    use_dense: bool = True
    use_bm25: bool = True
    top_k: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def validate_retrieval_mode(self) -> Self:
        """Require an enabled source and aligned selected event positions."""

        if not self.use_dense and not self.use_bm25:
            raise ValueError("at least one of use_dense or use_bm25 must be true")
        if self.retrieval_events is not None and len(self.retrieval_events) != len(self.events):
            raise ValueError("retrieval_events must match the original event count")
        return self


class TRAKEPath(BaseModel):
    """One ranked ordered frame path for a single video."""

    model_config = ConfigDict(extra="forbid")

    video_id: str
    score: float
    frame_ids: list[str]
    frame_idxs: list[int]
    timestamps_ms: list[int]

    @model_validator(mode="after")
    def validate_alignment_arrays(self) -> Self:
        """Keep the ordered path arrays aligned by event index."""

        if not (len(self.frame_ids) == len(self.frame_idxs) == len(self.timestamps_ms)):
            raise ValueError("alignment arrays must have equal lengths")

        return self


class TRAKEResponse(BaseModel):
    """Complete TRAKE response with independent ranked paths."""

    model_config = ConfigDict(extra="forbid")

    events: list[str]
    dense_events: list[str] | None = None
    bm25_caption_events: list[str] | None = None
    use_dense: bool = True
    use_bm25: bool = False
    paths: list[TRAKEPath] = Field(default_factory=list)
    latency: SearchLatency

    @model_validator(mode="after")
    def validate_path_lengths(self) -> Self:
        """Ensure every path contains one aligned frame for each event."""

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
        if any(len(path.frame_ids) != expected_length for path in self.paths):
            raise ValueError("each path must contain one frame per event")

        return self
