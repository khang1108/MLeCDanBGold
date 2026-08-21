"""Competition and inference-provider visual question-answering contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString
from .enum import ExecutionProfile, QueryLanguage, TaskType
from .search import SearchFilters
from .telemetry import PipelineTrace


class VQABaselineProfile(str, Enum):
    """Stable names for executable VQA baseline profiles."""

    SINGLE_FRAME = "vqa_single_frame"
    VRAG = "vqa_vrag"
    LOCALIZER = "vqa_localizer"
    HIERARCHICAL = "vqa_hierarchical"


class VQAInferenceRequest(ContractModel):
    """Ask an inference provider one question about one canonical frame."""

    frame_id: NonEmptyString
    video_id: NonEmptyString
    question: NonEmptyString = Field(max_length=1_000)


class VQAInferenceEvidenceItem(ContractModel):
    """One bounded, canonical text-evidence item supplied to a provider."""

    source: NonEmptyString
    value: NonEmptyString = Field(max_length=4_000)
    frame_id: NonEmptyString
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0, le=1)
    provenance: NonEmptyString

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end_ms < self.start_ms:
            raise ValueError("evidence end_ms must not precede start_ms")
        return self


class VQAInferenceEvidence(ContractModel):
    """Optional evidence supplied to a VQA inference provider."""

    items: list[VQAInferenceEvidenceItem] = Field(
        default_factory=list,
        max_length=64,
    )
    caption: NonEmptyString | None = None
    ocr_text: NonEmptyString | None = None
    asr_text: NonEmptyString | None = None
    objects: list[NonEmptyString] = Field(default_factory=list)


class VQAInferenceResponse(ContractModel):
    """One bounded provider answer grounded in supplied canonical frames."""

    request_id: NonEmptyString
    video_id: NonEmptyString
    frame_ids: list[NonEmptyString] = Field(min_length=1, max_length=32)
    selected_frame_id: NonEmptyString
    question: NonEmptyString
    answer: NonEmptyString = Field(max_length=100)
    answerable: bool = True
    grounded: bool
    confidence: float = Field(default=0.5, ge=0, le=1)
    model_name: NonEmptyString | None = None
    latency_ms: int = Field(ge=0)
    evidence: VQAInferenceEvidence = Field(default_factory=VQAInferenceEvidence)
    warnings: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selected_frame(self) -> Self:
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame_ids must be unique")
        if self.selected_frame_id not in self.frame_ids:
            raise ValueError("selected_frame_id must be one of frame_ids")
        return self


class VQARequest(ContractModel):
    """Competition VQA query containing retrieval and answer intent."""

    query_type: Literal[TaskType.VQA] = TaskType.VQA
    event_description: NonEmptyString = Field(max_length=1_000)
    question: NonEmptyString = Field(max_length=1_000)
    top_k: int = Field(default=20, ge=1, le=100)
    filters: SearchFilters | None = None
    language_hint: QueryLanguage | None = None
    execution_profile: ExecutionProfile | None = None
    baseline_profile: VQABaselineProfile = VQABaselineProfile.LOCALIZER
    search_id: NonEmptyString | None = None


class VQASubmission(ContractModel):
    """One ranked official VQA row with grounding and ranking provenance."""

    rank: int = Field(ge=1, le=100)

    video_id: NonEmptyString
    frame_idx: NonEmptyString

    fps: float = Field(default=25.0, gt=0)
    frame_ids: list[NonEmptyString] = Field(default_factory=list)
    
    answer: NonEmptyString = Field(max_length=100)
    normalized_answer: NonEmptyString | None = Field(default=None, max_length=100)
    
    retrieval_score: float
    grounding_score: float
    answer_score: float
    joint_score: float
    
    timestamp_ms: int | None = Field(default=None, ge=0)
    
    temporal_window: tuple[int, int] | None = None
    
    evidence_consistency_score: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    
    warnings: list[NonEmptyString] = Field(default_factory=list)
    caption: NonEmptyString | None = None
    evidence_summary: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_temporal_window(self) -> Self:
        if (
            self.temporal_window is not None
            and self.temporal_window[1] < self.temporal_window[0]
        ):
            raise ValueError("temporal_window end must not precede its start")
        frame_id = getattr(self, "frame_id", None)
        if not self.frame_ids and frame_id:
            self.frame_ids = [frame_id]
        return self


class VQARetrievalEvidence(ContractModel):
    """Canonical retrieval fallback exposed when answering is unavailable."""

    rank: int = Field(ge=1, le=100)
    video_id: NonEmptyString
    frame_id: NonEmptyString
    frame_idx: NonEmptyString
    
    fps: float = Field(default=25.0, gt=0)
    frame_ids: list[NonEmptyString] = Field(default_factory=list)
    timestamp_ms: int = Field(ge=0)
    retrieval_score: float

    @model_validator(mode="after")
    def populate_frame_ids(self) -> Self:
        if not self.frame_ids and self.frame_id:
            self.frame_ids = [self.frame_id]
        return self


class VQAResponse(ContractModel):
    """Ranked competition VQA submissions for one request."""

    request_id: NonEmptyString
    search_id: NonEmptyString | None = None
    query_type: Literal[TaskType.VQA] = TaskType.VQA
    event_description: NonEmptyString
    question: NonEmptyString
    top_k: int = Field(ge=1, le=100)
    total_results: int = Field(ge=0, le=100)
    submissions: list[VQASubmission] = Field(default_factory=list)
    evidence_candidates: list[VQARetrievalEvidence] = Field(default_factory=list)
    warnings: list[NonEmptyString] = Field(default_factory=list)
    latency_ms: int = Field(default=0, ge=0)
    trace: PipelineTrace = Field(default_factory=PipelineTrace)

    @model_validator(mode="after")
    def validate_submissions(self) -> Self:
        if self.total_results != len(self.submissions):
            raise ValueError("total_results must equal the number of submissions")
        if self.total_results > self.top_k:
            raise ValueError("total_results must not be greater than top_k")
        expected_ranks = list(range(1, self.total_results + 1))
        if [row.rank for row in self.submissions] != expected_ranks:
            raise ValueError("submission ranks must be consecutive and one-based")
        if len(self.evidence_candidates) > self.top_k:
            raise ValueError("evidence candidate count must not be greater than top_k")
        expected_evidence_ranks = list(range(1, len(self.evidence_candidates) + 1))
        if [row.rank for row in self.evidence_candidates] != expected_evidence_ranks:
            raise ValueError("evidence candidate ranks must be consecutive and one-based")
        return self
