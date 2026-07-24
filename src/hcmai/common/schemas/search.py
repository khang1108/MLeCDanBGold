from __future__ import annotations

from typing import Self
from pydantic import Field, field_validator, model_validator

from hcmai.common.schemas import (
    ContractModel,
    NonEmptyString,
    SearchMode,
    SearchScores,
)
from hcmai.common.schemas.conversation import FrameFeedback


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
    top_k: int = Field(default=20, ge=1, le=100)
    search_mode: SearchMode = SearchMode.ACCURATE
    filters: SearchFilters | None = None
    session_id: NonEmptyString | None = None
    feedback: FrameFeedback | None = None

    @model_validator(mode="after")
    def validate_kisc_context(self) -> Self:
        """Require a session for stateful frame feedback."""
        if self.feedback is not None and self.session_id is None:
            raise ValueError("feedback requires session_id")
        return self


class SearchLatency(ContractModel):
    """Backend latency of each search stage in milliseconds."""

    query_processing: int = Field(default=0, ge=0)
    query_encoding: int = Field(default=0, ge=0)
    candidate_retrieval: int = Field(default=0, ge=0)
    fusion: int = Field(default=0, ge=0)
    reranking: int = Field(default=0, ge=0)
    temporal_refinement: int = Field(default=0, ge=0)
    materialization: int = Field(default=0, ge=0)
    total: int = Field(ge=0)


class SearchResult(ContractModel):
    """One ranked frame returned by the public search API."""

    rank: int = Field(ge=1)
    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    thumbnail_url: NonEmptyString | None = None
    frame_url: NonEmptyString | None = None
    caption: NonEmptyString | None = None
    ocr_text: NonEmptyString | None = None
    asr_text: NonEmptyString | None = None
    scores: SearchScores


class SearchResponse(ContractModel):
    """Public response returned by the frame search endpoint."""

    request_id: NonEmptyString
    query: NonEmptyString
    search_mode: SearchMode
    top_k: int = Field(ge=1, le=100)
    total_results: int = Field(ge=0)
    latency_ms: SearchLatency
    results: list[SearchResult] = Field(default_factory=list)
    warnings: list[NonEmptyString] = Field(default_factory=list)
    session_id: NonEmptyString | None = None
    turn_id: NonEmptyString | None = None
    assistant_turn_id: NonEmptyString | None = None
    ai_message: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_result_count(self) -> Self:
        """Validate result counts against the requested response size."""

        if self.total_results != len(self.results):
            raise ValueError("total_results must equal the number of results")

        if self.total_results > self.top_k:
            raise ValueError("total_results must not be greater than top_k")

        context = (self.turn_id, self.assistant_turn_id, self.ai_message)
        has_context = any(value is not None for value in context)
        missing_context = any(value is None for value in context)
        if self.session_id is None and has_context:
            raise ValueError("KISC turn metadata requires session_id")
        if self.session_id is not None and missing_context:
            raise ValueError("KISC responses require complete turn metadata")

        return self

