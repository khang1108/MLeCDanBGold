"""Ablation matrix and configuration definitions for P0 temporal evidence evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from hcmai.common.config import AlignmentConfig, HybridTemporalConfig

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

    def apply_to(self, baseline: HybridTemporalConfig) -> HybridTemporalConfig:
        """Apply this run's switches without resetting baseline weights or boosts."""

        adaptive = baseline.adaptive.model_copy(
            update={
                "robust_calibration": self.robust_calibration,
                "confidence_gating": self.confidence_gating,
                "event_routing": self.event_routing,
                "asr_interval_projection": self.interval_projection,
            }
        )
        return baseline.model_copy(
            update={"fusion_mode": self.fusion_mode, "adaptive": adaptive}
        )


ABLATION_RUNS: dict[str, AblationRunConfig] = {
    "B0_legacy_v9": AblationRunConfig(
        name="B0_legacy_v9",
        fusion_mode="legacy",
        robust_calibration=False,
        confidence_gating=False,
        event_routing=False,
        interval_projection=False,
        use_dense=True,
        use_bm25=True,
    ),
    "B1_flat_components": AblationRunConfig(
        name="B1_flat_components",
        fusion_mode="adaptive_p0",
        robust_calibration=False,
        confidence_gating=False,
        event_routing=False,
        interval_projection=False,
        use_dense=True,
        use_bm25=True,
    ),
    "B2_asr_interval": AblationRunConfig(
        name="B2_asr_interval",
        fusion_mode="adaptive_p0",
        robust_calibration=False,
        confidence_gating=False,
        event_routing=False,
        interval_projection=True,
        use_dense=True,
        use_bm25=True,
    ),
    "B3_robust_calibration": AblationRunConfig(
        name="B3_robust_calibration",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=False,
        event_routing=False,
        interval_projection=True,
        use_dense=True,
        use_bm25=True,
    ),
    "B4_confidence_gating": AblationRunConfig(
        name="B4_confidence_gating",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=False,
        interval_projection=True,
        use_dense=True,
        use_bm25=True,
    ),
    "B5_adaptive_p0": AblationRunConfig(
        name="B5_adaptive_p0",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        use_dense=True,
        use_bm25=True,
    ),
    "B6_dense_only": AblationRunConfig(
        name="B6_dense_only",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        use_dense=True,
        use_bm25=False,
    ),
    "P2_paths2_sep30s": AblationRunConfig(
        name="P2_paths2_sep30s",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(paths_per_video=2, path_min_separation_ms=30000),
    ),
    "P3_paths3_sep30s": AblationRunConfig(
        name="P3_paths3_sep30s",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(paths_per_video=3, path_min_separation_ms=30000),
    ),
    "P5_paths5_sep30s": AblationRunConfig(
        name="P5_paths5_sep30s",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(paths_per_video=5, path_min_separation_ms=30000),
    ),
    "P3_paths3_sep10s": AblationRunConfig(
        name="P3_paths3_sep10s",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(paths_per_video=3, path_min_separation_ms=10000),
    ),
    "G0_lambda_0": AblationRunConfig(
        name="G0_lambda_0",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(lambda_gap=0.0),
    ),
    "G1_lambda_1e6": AblationRunConfig(
        name="G1_lambda_1e6",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(lambda_gap=1e-06),
    ),
    "G2_lambda_1e4": AblationRunConfig(
        name="G2_lambda_1e4",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(lambda_gap=0.0001),
    ),
    "G3_lambda_1e3": AblationRunConfig(
        name="G3_lambda_1e3",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(lambda_gap=0.001),
    ),
    "C1_paths3_sep10s_g1e6": AblationRunConfig(
        name="C1_paths3_sep10s_g1e6",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(
            lambda_gap=1e-06, paths_per_video=3, path_min_separation_ms=10000
        ),
    ),
    "C2_paths5_sep10s_g1e6": AblationRunConfig(
        name="C2_paths5_sep10s_g1e6",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(
            lambda_gap=1e-06, paths_per_video=5, path_min_separation_ms=10000
        ),
    ),
    "C3_paths5_sep5s_g1e6": AblationRunConfig(
        name="C3_paths5_sep5s_g1e6",
        fusion_mode="adaptive_p0",
        robust_calibration=True,
        confidence_gating=True,
        event_routing=True,
        interval_projection=True,
        alignment=AlignmentConfig(
            lambda_gap=1e-06, paths_per_video=5, path_min_separation_ms=5000
        ),
    ),
}

# A1 componentized-legacy is a regression test, not a runtime condition.
# B0-B6 are performance experiments whose flat-component baseline is allowed
# to differ from the legacy fusion equation.
_SHORT_KEY_MAP = {
    name.split("_", 1)[0]: name for name in ABLATION_RUNS if name.startswith("B")
}


def resolve_ablation_run(key: str) -> AblationRunConfig:
    """Resolve an ablation run by full name or short alias (for example, ``B0``)."""

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
    run_hybrid = run_cfg.apply_to(baseline_scorer.config)
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
