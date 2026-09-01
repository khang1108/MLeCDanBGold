"""Thin HTTP contracts for the TRAKE alignment endpoint.

This module owns only the public ordered-event request/response payloads. It
does not own event parsing, temporal search, or corpus materialization.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .latency import SearchLatency


_NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class TRAKERequest(BaseModel):
    """Public TRAKE request with explicit ordered events."""

    model_config = ConfigDict(extra="forbid")

    events: list[_NonBlankString] = Field(min_length=1)
    top_k: int = Field(default=20, ge=1)


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

        if not (
            len(self.frame_ids)
            == len(self.frame_idxs)
            == len(self.timestamps_ms)
        ):
            raise ValueError("alignment arrays must have equal lengths")

        return self


class TRAKEResponse(BaseModel):
    """Complete TRAKE response with independent ranked paths."""

    model_config = ConfigDict(extra="forbid")

    events: list[str]
    paths: list[TRAKEPath] = Field(default_factory=list)
    latency: SearchLatency

    @model_validator(mode="after")
    def validate_path_lengths(self) -> Self:
        """Ensure every path contains one aligned frame for each event."""

        expected_length = len(self.events)
        if any(len(path.frame_ids) != expected_length for path in self.paths):
            raise ValueError("each path must contain one frame per event")

        return self
