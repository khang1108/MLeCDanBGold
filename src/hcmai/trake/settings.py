"""Env-tunable knobs for the TRAKE pipeline."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TRAKESettings(BaseSettings):
    """TRAKE tuning knobs, read from ``.env`` under the ``HCMAI_TRAKE_`` prefix."""

    model_config = SettingsConfigDict(
        env_prefix="HCMAI_TRAKE_",
        env_file=".env",
        extra="ignore",
    )

    top_k: int = Field(default=500, ge=1, description="Frames kept per event when shortlisting.")
    max_videos: int = Field(default=200, ge=1, description="Videos kept for rescoring.")
    rrf_k: int = Field(default=60, gt=0, description="RRF constant when voting for videos.")
    lambda_gap: float = Field(default=1e-5, ge=0.0, description="Alignment time-gap penalty per ms.")
    event_power: float = Field(default=1.0, gt=0.0, le=1.0, description="Similarity exponent; below 1.0 penalizes weak events.")
    chunk_size: int = Field(default=65_536, ge=1, description="Vectors reconstructed per rescoring chunk.")
