"""Competition and inference-provider visual question-answering contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString
from .enum import ExecutionProfile, QueryLanguage, TaskType
from .search import SearchFilters


class VQAInferenceRequest(ContractModel):
    """Ask an inference provider one question about one canonical frame."""

    frame_id: NonEmptyString
    question: NonEmptyString = Field(max_length=1_000)


class VQAInferenceEvidence(ContractModel):
    """Optional evidence supplied to a one-frame VQA inference provider."""

    caption: NonEmptyString | None = None
    ocr_text: NonEmptyString | None = None
    asr_text: NonEmptyString | None = None
    objects: list[NonEmptyString] = Field(default_factory=list)


class VQAInferenceResponse(ContractModel):
    """One bounded provider answer grounded in the requested frame."""

    request_id: NonEmptyString
    frame_id: NonEmptyString
    question: NonEmptyString
    answer: NonEmptyString = Field(max_length=100)
    grounded: bool
    model_name: NonEmptyString | None = None
    latency_ms: int = Field(ge=0)
    evidence: VQAInferenceEvidence = Field(default_factory=VQAInferenceEvidence)
    warnings: list[NonEmptyString] = Field(default_factory=list)


class VQARequest(ContractModel):
    """Competition VQA query containing retrieval and answer intent."""

    query_type: Literal[TaskType.VQA] = TaskType.VQA
    event_description: NonEmptyString = Field(max_length=1_000)
    question: NonEmptyString = Field(max_length=1_000)
    top_k: int = Field(default=20, ge=1, le=100)
    filters: SearchFilters | None = None
    language_hint: QueryLanguage | None = None
    execution_profile: ExecutionProfile | None = None


class VQASubmission(ContractModel):
    """One ranked official VQA row with grounding and ranking provenance."""

    rank: int = Field(ge=1, le=100)
    video_id: NonEmptyString
    frame_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    answer: NonEmptyString = Field(max_length=100)
    retrieval_score: float
    grounding_score: float
    answer_score: float
    joint_score: float
    warnings: list[NonEmptyString] = Field(default_factory=list)
    evidence_summary: NonEmptyString | None = None


class VQAResponse(ContractModel):
    """Ranked competition VQA submissions for one request."""

    request_id: NonEmptyString
    query_type: Literal[TaskType.VQA] = TaskType.VQA
    event_description: NonEmptyString
    question: NonEmptyString
    top_k: int = Field(ge=1, le=100)
    total_results: int = Field(ge=0, le=100)
    submissions: list[VQASubmission] = Field(default_factory=list)
    warnings: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_submissions(self) -> Self:
        if self.total_results != len(self.submissions):
            raise ValueError("total_results must equal the number of submissions")
        if self.total_results > self.top_k:
            raise ValueError("total_results must not be greater than top_k")
        expected_ranks = list(range(1, self.total_results + 1))
        if [row.rank for row in self.submissions] != expected_ranks:
            raise ValueError("submission ranks must be consecutive and one-based")
        return self
