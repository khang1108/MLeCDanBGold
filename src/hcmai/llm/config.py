"""Typed configuration for the GPU-only inference service."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from hcmai.common.config import EncoderConfig
from hcmai.common.utils.io import read_yaml


class HostedCaptionConfig(BaseModel):
    """Caption model settings owned by the hosted inference service."""

    model_checkpoint: str = "florence-community/Florence-2-base-ft"
    revision: str | None = "0b03b6f15a4a211370fb204aee4e7dd48887ea37"
    prompt: str = "<CAPTION>"
    decoding: dict = Field(
        default_factory=lambda: {
            "max_new_tokens": 64,
            "num_beams": 3,
            "do_sample": False,
        }
    )
    device: str = "cuda"
    dtype: str = "bfloat16"


class ServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8100, ge=1, le=65535)


class HostedRerankerConfig(BaseModel):
    checkpoint: str = "Qwen/Qwen3-VL-Reranker-2B"
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = Field(default=1, ge=1, le=16)
    max_length: int = Field(default=1024, ge=1)
    max_pixels: int = Field(default=262144, ge=4096)


class HostedConversationConfig(BaseModel):
    checkpoint: str | None = None
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = Field(default=512, ge=32, le=2048)


class QuerySuggestionGpuConfig(BaseModel):
    """Model and private endpoint used by the owned GPU service."""

    checkpoint: str | None = None
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    endpoint_path: str = "/v1/query-suggestions"
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)


class QuerySuggestionApiConfig(BaseModel):
    """OpenAI-compatible third-party provider settings."""

    base_url: str = "https://api.example.com/v1"
    api_key_env: str = "HCMAI_QUERY_SUGGESTION_API_KEY"
    model: str = "configure-when-active"
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)


class QuerySuggestionGenerationConfig(BaseModel):
    """Shared decoding controls applied by either provider."""

    max_new_tokens: int = Field(default=512, ge=128, le=2048)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)


class QuerySuggestionConfig(BaseModel):
    """Exactly one configured provider for the operator suggestion endpoint."""

    enabled: bool = False
    active_provider: Literal["gpu_inference", "openai_compatible"] = "gpu_inference"
    default_count: int = Field(default=8, ge=5, le=10)
    generation: QuerySuggestionGenerationConfig = Field(
        default_factory=QuerySuggestionGenerationConfig
    )
    gpu_inference: QuerySuggestionGpuConfig = Field(
        default_factory=QuerySuggestionGpuConfig
    )
    openai_compatible: QuerySuggestionApiConfig = Field(
        default_factory=QuerySuggestionApiConfig
    )

    @model_validator(mode="after")
    def validate_active_provider(self) -> Self:
        if not self.enabled:
            return self
        if (
            self.active_provider == "gpu_inference"
            and self.gpu_inference.checkpoint is None
        ):
            raise ValueError("active GPU query-suggestion provider needs a checkpoint")
        if (
            self.active_provider == "openai_compatible"
            and self.openai_compatible.model == "configure-when-active"
        ):
            raise ValueError(
                "active OpenAI-compatible provider needs a configured model"
            )
        return self


class LLMServiceConfig(BaseModel):
    server: ServiceConfig = Field(default_factory=ServiceConfig)
    caption_generation: HostedCaptionConfig = Field(
        default_factory=HostedCaptionConfig
    )
    visual_embedding: EncoderConfig = Field(default_factory=EncoderConfig)
    caption_embedding: EncoderConfig = Field(default_factory=EncoderConfig)
    reranker: HostedRerankerConfig = Field(default_factory=HostedRerankerConfig)
    conversation: HostedConversationConfig = Field(
        default_factory=HostedConversationConfig
    )
    query_suggestions: QuerySuggestionConfig = Field(
        default_factory=QuerySuggestionConfig
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMServiceConfig:
        data = read_yaml(path)
        for field in ("visual_embedding", "caption_embedding"):
            if field in data:
                data[field] = EncoderConfig.from_dict(data[field])
        return cls.model_validate(data)
