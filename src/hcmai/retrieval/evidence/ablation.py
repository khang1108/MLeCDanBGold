"""Ablation matrix and configuration definitions for P0 temporal evidence evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from hcmai.common.config import (
    AdaptiveTemporalFusionConfig,
    AlignmentConfig,
    HybridTemporalConfig,
)

if TYPE_CHECKING:
    from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer


@dataclass(frozen=True)
class AblationRunConfig:
    """Explicit configuration definition for one stage in the P0 ablation matrix."""

    name: str
    fusion_mode: Literal["legacy", "adaptive_p0"]
    robust_calibration: bool
    confidence_gating: bool
    event_routing: bool
    interval_projection: bool
    use_dense: bool = True
    use_bm25: bool = True
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)

    def to_hybrid_config(self) -> HybridTemporalConfig:
        """Convert ablation settings into a concrete HybridTemporalConfig."""

        return HybridTemporalConfig(
            fusion_mode=self.fusion_mode,
            adaptive=AdaptiveTemporalFusionConfig(
                robust_calibration=self.robust_calibration,
                confidence_gating=self.confidence_gating,
                event_routing=self.event_routing,
                asr_interval_projection=self.interval_projection,
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
        use_dense=True,
        use_bm25=True,
    ),
    "A1_components_fixed": AblationRunConfig(
        name="A1_components_fixed",
        fusion_mode="adaptive_p0",
        robust_calibration=False,
        confidence_gating=False,
        event_routing=False,
        interval_projection=False,
        use_dense=True,
        use_bm25=True,
    ),
    "A2_asr_interval": AblationRunConfig(
        name="A2_asr_interval",
        fusion_mode="adaptive_p0",
        robust_calibration=False,
        confidence_gating=False,
        event_routing=False,
        interval_projection=True,
        use_dense=True,
        use_bm25=True,
    ),
    "A3_robust_calibration": AblationRunConfig(
        name="A3_robust_calibration",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=False,
        event_routing=False,
        interval_projection=True,
        use_dense=True,
        use_bm25=True,
    ),
    "A4_confidence_gating": AblationRunConfig(
        name="A4_confidence_gating",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=False,
        interval_projection=True,
        use_dense=True,
        use_bm25=True,
    ),
    "A5_adaptive_p0": AblationRunConfig(
        name="A5_adaptive_p0",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        use_dense=True,
        use_bm25=True,
    ),
    "A6_dense_only": AblationRunConfig(
        name="A6_dense_only",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        use_dense=True,
        use_bm25=False,
    ),
}

# Alias short names A0-A6
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


def make_ablation_scorer(
    baseline_scorer: TemporalEvidenceScorer,
    run_key: str,
) -> tuple[TemporalEvidenceScorer, dict[str, Any]]:
    """Derive an isolated ablation scorer from a baseline scorer.

    Preserves runtime base weights and config while applying only the
    isolated parameters for the specified ablation run.
    """

    run_cfg = resolve_ablation_run(run_key)
    base_hybrid = baseline_scorer.config
    updated_adaptive = base_hybrid.adaptive.model_copy(
        update={
            "robust_calibration": run_cfg.robust_calibration,
            "confidence_gating": run_cfg.confidence_gating,
            "event_routing": run_cfg.event_routing,
            "asr_interval_projection": run_cfg.interval_projection,
        }
    )
    run_hybrid = base_hybrid.model_copy(
        update={
            "fusion_mode": run_cfg.fusion_mode,
            "adaptive": updated_adaptive,
        }
    )
    cloned_scorer = baseline_scorer.with_config(run_hybrid)
    run_kwargs: dict[str, Any] = {
        "use_dense": run_cfg.use_dense,
        "use_bm25": run_cfg.use_bm25,
    }
    return cloned_scorer, run_kwargs


__all__ = [
    "ABLATION_RUNS",
    "AblationRunConfig",
    "make_ablation_scorer",
    "resolve_ablation_run",
]
