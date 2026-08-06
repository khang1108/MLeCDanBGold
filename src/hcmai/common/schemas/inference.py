"""Contracts shared by the local search backend and remote model service."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .base import ContractModel, NonEmptyString
from .conversation import ConversationState, ConversationTurn, FrameFeedback


class TextEmbeddingRequest(ContractModel):
    """Ordered text batch routed to one configured retrieval encoder."""

    source: Literal["visual", "caption"] = "visual"
    texts: list[NonEmptyString] = Field(min_length=1, max_length=64)


class TextEmbeddingResponse(ContractModel):
    """Normalized vectors plus index-compatibility provenance."""

    model: NonEmptyString
    revision: str | None = None
    dimension: int = Field(gt=0)
    normalized: bool
    embeddings: list[list[float]]
    latency_ms: float = Field(ge=0)

    @field_validator("embeddings")
    @classmethod
    def validate_shape(cls, values: list[list[float]]) -> list[list[float]]:
        if not values or not values[0]:
            raise ValueError("embeddings must be a non-empty matrix")
        if any(len(row) != len(values[0]) for row in values):
            raise ValueError("embedding rows must have equal dimensions")
        return values


class CaptionItem(ContractModel):
    """One caller-owned image and its generated caption."""

    item_id: NonEmptyString
    caption: NonEmptyString


class CaptionResponse(ContractModel):
    """Ordered captions returned by the hosted vision-language model."""

    model: NonEmptyString
    revision: NonEmptyString
    items: list[CaptionItem]
    latency_ms: float = Field(ge=0)


class RerankItem(ContractModel):
    """One caller-owned item and its model relevance score."""

    item_id: NonEmptyString
    score: float


class RerankResponse(ContractModel):
    """Ordered multimodal scores returned by the remote reranker."""

    model: NonEmptyString
    revision: str | None = None
    items: list[RerankItem]
    latency_ms: float = Field(ge=0)


class ConversationInferenceRequest(ContractModel):
    """Complete bounded context required for one resolver call."""

    instruction: NonEmptyString
    history: list[ConversationTurn] = Field(default_factory=list)
    current_message: NonEmptyString
    feedback: FrameFeedback | None = None
    previous_state: ConversationState | None = None
    response_schema: dict = Field(default_factory=dict)


class ModelStatus(ContractModel):
    """Readiness and provenance for one hosted model."""

    enabled: bool = True
    loaded: bool
    checkpoint: str | None = None
    revision: str | None = None


class InferenceCapabilities(ContractModel):
    """Feature-level readiness advertised by one inference deployment."""

    embedding: bool = False
    reranking: bool = False
    multi_image_vqa: bool = False
    structured_parsing: bool = False


class InferenceReadiness(ContractModel):
    """Readiness snapshot for all configured inference capabilities."""

    ready: bool
    models: dict[str, ModelStatus]
    capabilities: InferenceCapabilities = Field(
        default_factory=InferenceCapabilities
    )
