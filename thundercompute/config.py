"""Typed configuration for the GPU-only inference service."""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field

from hcmai.common.config import EncoderConfig
from hcmai.common.utils.io import read_yaml


class HostedCaptionConfig(BaseModel):
    """Caption model settings owned by the hosted inference service."""

    model_checkpoint: str = "Qwen/Qwen3-VL-8B-Instruct"
    revision: str | None = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    prompt: str = "qwen vl"
    decoding: dict = Field(
        default_factory=lambda: {
            "max_new_tokens": 160,
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


class LLMServiceConfig(BaseModel):
    """Hosted inference settings plus pinned dense encoder configurations."""

    server: ServiceConfig = Field(default_factory=ServiceConfig)
    caption_generation: HostedCaptionConfig = Field(
        default_factory=HostedCaptionConfig
    )
    visual_embedding: EncoderConfig = Field(default_factory=EncoderConfig)
    caption_embedding: EncoderConfig = Field(default_factory=EncoderConfig)
    evidence_embedding: EncoderConfig | None = None
    reranker: HostedRerankerConfig = Field(default_factory=HostedRerankerConfig)

    @property
    def resolved_evidence_embedding(self) -> EncoderConfig:
        """Return the generic evidence encoder with caption compatibility fallback."""

        return self.evidence_embedding or self.caption_embedding

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMServiceConfig:
        """Load model settings while accepting legacy encoder mapping syntax."""

        data = read_yaml(path)
        for field in (
            "visual_embedding",
            "caption_embedding",
            "evidence_embedding",
        ):
            if field in data:
                data[field] = EncoderConfig.from_dict(data[field])
        return cls.model_validate(data)
