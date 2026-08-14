from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from hcmai.common.schemas.vqa import VQABaselineProfile


class VQAProfileConfig(BaseModel):
    """Hard budgets for one reproducible competition VQA baseline."""

    candidate_videos: int = Field(default=5, ge=1, le=100)
    candidates_per_branch: int = Field(default=100, ge=1, le=1_000)
    window_ms: int = Field(default=15_000, ge=1_000, le=120_000)
    max_windows: int = Field(default=12, ge=1, le=100)
    max_frames_per_window: int = Field(default=4, ge=1, le=32)
    max_evidence_items: int = Field(default=24, ge=1, le=256)
    max_vlm_calls: int = Field(default=8, ge=0, le=100)
    localizer_enabled: bool = True
    temporal_core_enabled: bool = False
    hierarchical_refinement: bool = False
    temporal_fallback_ms: int = Field(default=15_000, ge=0, le=120_000)


def _default_vqa_profiles() -> dict[VQABaselineProfile, VQAProfileConfig]:
    return {
        VQABaselineProfile.SINGLE_FRAME: VQAProfileConfig(
            candidate_videos=1,
            window_ms=8_000,
            max_windows=1,
            max_frames_per_window=1,
            max_vlm_calls=1,
            localizer_enabled=False,
            temporal_fallback_ms=0,
        ),
        VQABaselineProfile.VRAG: VQAProfileConfig(
            candidate_videos=10,
            window_ms=15_000,
            max_windows=20,
            max_frames_per_window=4,
            max_vlm_calls=10,
            localizer_enabled=False,
        ),
        VQABaselineProfile.LOCALIZER: VQAProfileConfig(),
        VQABaselineProfile.HIERARCHICAL: VQAProfileConfig(
            candidate_videos=8,
            candidates_per_branch=150,
            window_ms=30_000,
            max_windows=16,
            max_frames_per_window=8,
            max_vlm_calls=12,
            hierarchical_refinement=True,
        ),
    }


class VQAConfig(BaseModel):
    """Executable VQA profiles selected without hidden inference budgets."""

    default_profile: VQABaselineProfile = VQABaselineProfile.LOCALIZER
    profiles: dict[VQABaselineProfile, VQAProfileConfig] = Field(
        default_factory=_default_vqa_profiles
    )

    @model_validator(mode="after")
    def validate_profiles(self) -> VQAConfig:
        if set(self.profiles) != set(VQABaselineProfile):
            raise ValueError(
                "vqa profiles must configure every baseline profile"
            )
        if self.default_profile not in self.profiles:
            raise ValueError("default VQA profile must be configured")
        return self
