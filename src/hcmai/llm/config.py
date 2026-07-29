"""Typed configuration for the GPU-only inference service."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from hcmai.common.config import EncoderConfig
from hcmai.common.utils.io import read_yaml


class HostedCaptionConfig(BaseModel):
    """Caption model settings owned by the hosted inference service."""

    model_checkpoint: str = "microsoft/Florence-2-base-ft"
    revision: str | None = None
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMServiceConfig:
        data = read_yaml(path)
        for field in ("visual_embedding", "caption_embedding"):
            if field in data:
                data[field] = EncoderConfig.from_dict(data[field])
        return cls.model_validate(data)
