"""Contracts for public AVS text search and candidate exposure.

This module defines public request, result, latency, and response shapes
for the dedicated AVS search API.
"""

from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .search import SearchResultMetadata

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AvsSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: NonBlank
    page_size: int = Field(default=80, ge=1)


class AvsSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: NonBlank
    frame_id: NonBlank
    video_id: NonBlank
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    fps: float | None = Field(default=None, gt=0)
    retrieval_rank: int = Field(ge=1)
    retrieval_score: float | None = None
    metadata: SearchResultMetadata


class AvsSearchLatency(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retrieval_ms: float = Field(ge=0)
    coverage_ms: float = Field(ge=0)
    materialization_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class AvsSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[AvsSearchResult]
    latency: AvsSearchLatency
    candidate_pool_size: int = Field(ge=0)
    deduplicated_candidate_count: int = Field(ge=0)
    unique_videos: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
