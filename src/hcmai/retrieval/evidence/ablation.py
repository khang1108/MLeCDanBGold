"""Ablation matrix and configuration definitions for P0 temporal evidence evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from hcmai.common.config import (
    AdaptiveTemporalFusionConfig,
    AlignmentConfig,
    HybridTemporalConfig,
)


@dataclass(frozen=True)
class AblationRunConfig:
    """Explicit configuration definition for one stage in the P0 ablation matrix."""

    name: str
    fusion_mode: Literal["legacy", "adaptive_p0"]
    robust_calibration: bool
    confidence_gating: bool
    event_routing: bool
    interval_projection: bool
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)

    def to_hybrid_config(self) -> HybridTemporalConfig:
        """Convert ablation settings into a concrete HybridTemporalConfig."""

        return HybridTemporalConfig(
            fusion_mode=self.fusion_mode,
            adaptive=AdaptiveTemporalFusionConfig(
                robust_calibration=self.robust_calibration,
                confidence_gating=self.confidence_gating,
                event_routing=self.event_routing,
            ),
        )


ABLATION_RUNS: dict[str, AblationRunConfig] = {
    "A0_legacy_v9": AblationRunConfig(
        name="A0_legacy_v9",
        fusion_mode="legacy",
        robust_calibration=False,
        confidence_gating=False,
        event_routing=False,
        interval_projection=False,
    ),
    "A1_components_fixed": AblationRunConfig(
        name="A1_components_fixed",
        fusion_mode="adaptive_p0",
        robust_calibration=False,
        confidence_gating=False,
        event_routing=False,
        interval_projection=False,
    ),
    "A2_robust_calibration": AblationRunConfig(
        name="A2_robust_calibration",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=False,
        event_routing=False,
        interval_projection=False,
    ),
    "A3_confidence_gating": AblationRunConfig(
        name="A3_confidence_gating",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=False,
        interval_projection=False,
    ),
    "A4_asr_interval": AblationRunConfig(
        name="A4_asr_interval",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=False,
        interval_projection=True,
    ),
    "A5_adaptive_p0": AblationRunConfig(
        name="A5_adaptive_p0",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
    ),
}

# Alias short names A0-A5
_SHORT_KEY_MAP = {
    f"A{i}": name for i, name in enumerate(ABLATION_RUNS.keys())
}


def resolve_ablation_run(key: str) -> AblationRunConfig:
    """Resolve an ablation run by full name or short alias (e.g. 'A0' -> 'A0_legacy_v9')."""

    if key in ABLATION_RUNS:
        return ABLATION_RUNS[key]
    upper_key = key.upper()
    if upper_key in _SHORT_KEY_MAP:
        return ABLATION_RUNS[_SHORT_KEY_MAP[upper_key]]
    valid = ", ".join(list(ABLATION_RUNS.keys()) + list(_SHORT_KEY_MAP.keys()))
    raise KeyError(f"Unknown ablation run '{key}'. Valid options: {valid}")
