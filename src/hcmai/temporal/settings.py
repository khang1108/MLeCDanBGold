"""Configuration for ordered temporal alignment.

This module owns the tunable budgets and scoring knobs used by TRAKE's
ordered-path alignment. It does not own request validation or HTTP response
formatting.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TRAKESettings(BaseSettings):
    """TRAKE alignment knobs read from ``.env`` with the ``HCMAI_TRAKE_`` prefix."""

    model_config = SettingsConfigDict(
        env_prefix="HCMAI_TRAKE_",
        env_file=".env",
        extra="ignore",
    )

    top_k: int = Field(
        default=500,
        ge=1,
        description="Frames kept per event when shortlisting.",
    )
    max_videos: int = Field(
        default=200,
        ge=1,
        description="Videos kept for rescoring.",
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        description="RRF constant when voting for videos.",
    )
    lambda_gap: float = Field(
        default=1e-5,
        ge=0.0,
        description="Alignment time-gap penalty per ms.",
    )
    event_power: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Similarity exponent; below 1.0 penalizes weak events.",
    )
    chunk_size: int = Field(
        default=65_536,
        ge=1,
        description="Vectors reconstructed per rescoring chunk.",
    )
    cluster_delta: float = Field(
        default=0.0,
        ge=0.0,
        description="Cluster radius; above 0.0 events must land in different clusters.",
    )
