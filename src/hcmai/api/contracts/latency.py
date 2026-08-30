"""Shared HTTP latency contracts for KIS and TRAKE.

This module owns the public API latency breakdown exposed by the FastAPI
boundary. It does not own internal tracing or fine-grained observability
stages that remain private to orchestration code.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SearchLatency(BaseModel):
    """Public latency breakdown for one temporal search request."""

    model_config = ConfigDict(extra="forbid")

    query_ms: float = Field(default=0, ge=0)
    retrieval_ms: float = Field(default=0, ge=0)
    alignment_ms: float = Field(default=0, ge=0)
    materialization_ms: float = Field(default=0, ge=0)
    total_ms: float = Field(default=0, ge=0)
