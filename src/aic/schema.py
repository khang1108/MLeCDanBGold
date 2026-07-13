"""An Unified Shared Data and API Contracts"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Self
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


# ==============================
# Shared Project Contracts
# ==============================


class ContractModel(BaseModel):
    """Base model for all shared project contracts."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class SearchMode(str, Enum):
    """Supported Search Profiles

    Args:
        FAST: We use this mode only when we are at the Finalist Round :33
        ACCURATE: Normal mode, set as Default
    """

    FAST = "fast"
    ACCURATE = "accuracte"


class ProcessingStatus(str, Enum):
    """Status of an offline processing operation."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RetrievalSource(str, Enum):
    """Evidence channels used to retrieve a frame."""

    VISUAL = "visual"
    CAPTION = "caption"
    OCR = "ocr"
    ASR = "asr"


class QueryLanguage(str, Enum):
    """Languages represented in the development query set."""

    VIETNAMESE = "vi"
    ENGLISH = "en"
    MIXED = "mixed"


class TaskType(str, Enum):
    """Competition task represented by an evaluation query."""

    TEXTUAL_KIS = "textual_kis"
    VIDEO_KIS = "video_kis"
    AD_HOC_SEARCH = "ad_hoc_search"
    VQA = "vqa"


class QueryDifficulty(str, Enum):
    """Human-assigned difficulty of an evaluation query."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ==============================
# Frame Contracts
# ==============================


class FrameRecord(ContractModel):
    """Canonical metadata for one searchable frame."""

    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    image_path: NonEmptyString
    thumbnail_path: NonEmptyString | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    shot_id: NonEmptyString | None = None
    is_anchor: bool = True


class FrameEnrichment(ContractModel):
    """Offline caption, OCR, ASR, and object metadata for a frame."""

    frame_id: NonEmptyString
    caption: NonEmptyString | None = None
    detailed_caption: NonEmptyString | None = None
    ocr_text: NonEmptyString | None = None
    asr_text: NonEmptyString | None = None
    enrichment_version: NonEmptyString | None = None
    objects: list[NonEmptyString] = Field(default_factory=list)
    model_name: NonEmptyString
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_message: NonEmptyString | None = None

    @field_validator("objects")
    @classmethod
    def deduplicate_objects(cls, objects: list[str]) -> list[str]:
        """Remove duplicate labels while preserving their original order."""

        return list(dict.fromkeys(objects))


# ==============================
# Search and Retrieval Contracts
# ==============================
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


class RetrievalCandidate(ContractModel):
    """Internal frame candidate shared by retrieval pipeline stages."""

    frame_id: NonEmptyString
    source_scores: dict[RetrievalSource, float] = Field(default_factory=dict)
    source_ranks: dict[RetrievalSource, int] = Field(default_factory=dict)
    fusion_score: float | None = None
    reranker_score: float | None = None
    final_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_ranks")
    @classmethod
    def validate_source_ranks(
        cls,
        source_ranks: dict[RetrievalSource, int],
    ) -> dict[RetrievalSource, int]:
        """Require retrieval ranks to be one-based positive integers."""

        if any(rank < 1 for rank in source_ranks.values()):
            raise ValueError("source ranks must be greater than or equal to 1")

        return source_ranks


class SearchScores(ContractModel):
    """Scores exposed for a returned frame."""

    visual: float | None = None
    caption: float | None = None
    ocr: float | None = None
    asr: float | None = None
    fusion: float | None = None
    reranker: float | None = None
    final: float


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

    @model_validator(mode="after")
    def validate_result_count(self) -> Self:
        """Validate result counts against the requested response size."""

        if self.total_results != len(self.results):
            raise ValueError("total_results must equal the number of results")

        if self.total_results > self.top_k:
            raise ValueError("total_results must not be greater than top_k")

        return self


# ==============================
# Evaluation Contracts
# ==============================
class EvaluationQuery(ContractModel):
    """One labelled query used by the offline evaluation harness."""

    schema_version: NonEmptyString = "1.0"
    query_id: NonEmptyString
    query: NonEmptyString = Field(max_length=1_000)
    language: QueryLanguage
    task_type: TaskType
    difficulty: QueryDifficulty
    gold_frame_ids: list[NonEmptyString] = Field(min_length=1)
    gold_video_ids: list[NonEmptyString] = Field(default_factory=list)
    temporal_tolerance_ms: int = Field(default=0, ge=0)
    tags: list[NonEmptyString] = Field(default_factory=list)
    notes: NonEmptyString | None = None

    @field_validator("gold_frame_ids", "gold_video_ids", "tags")
    @classmethod
    def deduplicate_string_lists(cls, values: list[str]) -> list[str]:
        """Remove duplicate strings while retaining deterministic order."""

        return list(dict.fromkeys(values))
