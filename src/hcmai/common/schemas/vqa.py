"""Frame-grounded visual question-answering API contracts."""

from __future__ import annotations

from pydantic import Field

from .base import ContractModel, NonEmptyString


class VQARequest(ContractModel):
    """Ask one question about one canonical frame."""

    frame_id: NonEmptyString
    question: NonEmptyString = Field(max_length=1_000)


class VQAEvidence(ContractModel):
    """Optional evidence made available to a VQA provider."""

    caption: NonEmptyString | None = None
    ocr_text: NonEmptyString | None = None
    asr_text: NonEmptyString | None = None
    objects: list[NonEmptyString] = Field(default_factory=list)


class VQAResponse(ContractModel):
    """One bounded answer grounded in the requested frame."""

    request_id: NonEmptyString
    frame_id: NonEmptyString
    question: NonEmptyString
    answer: NonEmptyString = Field(max_length=100)
    grounded: bool
    model_name: NonEmptyString | None = None
    latency_ms: int = Field(ge=0)
    evidence: VQAEvidence = Field(default_factory=VQAEvidence)
    warnings: list[NonEmptyString] = Field(default_factory=list)
