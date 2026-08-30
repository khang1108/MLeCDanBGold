from __future__ import annotations

from typing import Literal, Self
from pydantic import Field, field_validator, model_validator

from hcmai.common.schemas.base import ContractModel, NonEmptyString
from hcmai.common.schemas.enum import TaskType
from hcmai.common.schemas.retrieval import SearchScores
from hcmai.common.schemas.telemetry import PipelineTrace


class SearchFilters(ContractModel):
    """Optional restrictions applied to a search request."""

    video_ids: list[NonEmptyString] = Field(default_factory=list)
    start_time_ms: int | None = Field(default=None, ge=0)
    end_time_ms: int | None = Field(default=None, ge=0)
    min_score: float | None = None

    @field_validator("video_ids")
    @classmethod
    def deduplicate_video_ids(cls, video_ids: list[str]) -> list[str]:
        """Remove duplicate video identifiers without changing order."""

        return list(dict.fromkeys(video_ids))

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        """Ensure the end of a time range is not before its start."""

        if (
            self.start_time_ms is not None
            and self.end_time_ms is not None
            and self.end_time_ms < self.start_time_ms
        ):
            raise ValueError(
                "end_time_ms must be greater than or equal to start_time_ms"
            )

        return self


class SearchRequest(ContractModel):
    """Public request accepted by the frame search endpoint."""

    query: NonEmptyString = Field(max_length=1_000)
    query_type: Literal[TaskType.KIS] = TaskType.KIS
    events: list[NonEmptyString] | None = Field(
        default=None,
        min_length=1,
        max_length=10,
    )
    top_k: int = Field(default=20, ge=1, le=100)
    filters: SearchFilters | None = None
    search_id: NonEmptyString | None = None


class SearchLatency(ContractModel):
    """Backend latency of each search stage in milliseconds."""

    query_processing: int = Field(default=0, ge=0)
    query_encoding: int = Field(default=0, ge=0)

    candidate_retrieval: int = Field(default=0, ge=0)

    fusion: int = Field(default=0, ge=0)
    reranking: int = Field(default=0, ge=0)

    temporal_refinement: int = Field(default=0, ge=0)
    materialization: int = Field(default=0, ge=0)

    time_to_first_candidate: int = Field(default=0, ge=0)
    time_to_first_submission: int = Field(default=0, ge=0)

    total: int = Field(ge=0)


class SearchResult(ContractModel):
    """One ranked frame with exact internal and official identities."""

    rank: int = Field(ge=1)
    frame_id: NonEmptyString

    # ``frame_idx`` is the exact organizer-provided coordinate used for BTC
    # submission. Playback and temporal ranges must use ``timestamp_ms``.
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    fps: float = Field(default=25.0, gt=0)
    frame_ids: list[NonEmptyString] = Field(default_factory=list)
    
    timestamp_ms: int = Field(ge=0)
    thumbnail_url: NonEmptyString | None = None
    frame_url: NonEmptyString | None = None

    caption: NonEmptyString | None = None

    ocr_text: NonEmptyString | None = None
    asr_text: NonEmptyString | None = None

    scores: SearchScores

    @model_validator(mode="after")
    def populate_frame_ids(self) -> Self:
        """Ensure the selected internal ID is retained in scene evidence."""
        if not self.frame_ids:
            self.frame_ids = [self.frame_id]
        elif self.frame_id not in self.frame_ids:
            raise ValueError("frame_id must be included in frame_ids")
        return self


class SearchResponse(ContractModel):
    """Public response returned by the frame search endpoint."""

    request_id: NonEmptyString
    search_id: NonEmptyString | None = None

    query: NonEmptyString
    query_type: TaskType

    top_k: int = Field(ge=1, le=100)
    total_results: int = Field(ge=0)

    latency_ms: SearchLatency

    results: list[SearchResult] = Field(default_factory=list)

    warnings: list[NonEmptyString] = Field(default_factory=list)
    trace: PipelineTrace = Field(default_factory=PipelineTrace)

    @model_validator(mode="after")
    def validate_result_count(self) -> Self:
        """Validate result counts against the requested response size."""

        if self.total_results != len(self.results):
            raise ValueError("total_results must equal the number of results")

        if self.total_results > self.top_k:
            raise ValueError("total_results must not be greater than top_k")

        return self
