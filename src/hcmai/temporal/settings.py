"""Env-tunable knobs for the shared KIS/VQA temporal core."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from hcmai.temporal.alignment.coverage import CoverageWindowAligner
from hcmai.temporal.engine import TemporalEvidenceEngine
from hcmai.temporal.retrieval import SparseEvidenceProvider
from hcmai.temporal.scoring import SceneScorer
from hcmai.temporal.state import ProgressiveStateStore


class TemporalSettings(BaseSettings):
    """Temporal core knobs, read from ``.env`` under the ``HCMAI_TEMPORAL_`` prefix."""

    model_config = SettingsConfigDict(
        env_prefix="HCMAI_TEMPORAL_",
        env_file=".env",
        extra="ignore",
    )

    top_m: int = Field(default=10, ge=1, description="Evidence points kept per unit and video.")
    global_quota: int = Field(default=100, ge=1, description="Points kept from the corpus-wide search.")
    local_quota: int = Field(default=100, ge=1, description="Points kept from the per-video rescan.")

    max_span_ms: int = Field(default=30_000, ge=0, description="Widest span one scene may cover.")
    max_per_video: int = Field(default=5, ge=1, description="Scenes kept per video before ranking.")
    merge_overlap: float = Field(default=0.8, gt=0.0, le=1.0, description="Overlap above which two windows are the same moment.")

    semantic_weight: float = Field(default=0.4, ge=0.0, description="Scene score weight for evidence strength.")
    coverage_weight: float = Field(default=0.3, ge=0.0, description="Scene score weight for hint coverage.")
    temporal_weight: float = Field(default=0.15, ge=0.0, description="Scene score weight for compactness.")
    relation_weight: float = Field(default=0.15, ge=0.0, description="Scene score weight for parsed relations.")
    min_score_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Pull of the weakest unit on the semantic mean.")
    compact_half_life_ms: int = Field(default=5_000, ge=1, description="Gap at which the compactness score halves.")
    discriminative_hint_weights: bool = Field(default=False, description="Weight rare hints above common ones.")

    def engine(
        self,
        provider: SparseEvidenceProvider,
        states: ProgressiveStateStore,
    ) -> TemporalEvidenceEngine:
        """Build the engine with every knob resolved; the four score weights must sum to 1."""
        return TemporalEvidenceEngine(
            provider,
            states,
            aligner=CoverageWindowAligner(
                self.max_span_ms, self.max_per_video, self.merge_overlap
            ),
            scorer=SceneScorer(
                self.semantic_weight,
                self.coverage_weight,
                self.temporal_weight,
                self.relation_weight,
                self.min_score_weight,
                self.compact_half_life_ms,
                self.discriminative_hint_weights,
            ),
            top_m=self.top_m,
            global_quota=self.global_quota,
            local_quota=self.local_quota,
        )
