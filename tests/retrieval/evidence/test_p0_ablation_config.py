"""Regression test for the P0 ablation matrix configurations."""

from __future__ import annotations

import pytest

from hcmai.common.config import (
    AdaptiveTemporalFusionConfig,
    HybridTemporalConfig,
    RobustCalibrationConfig,
)
from hcmai.retrieval.evidence.ablation import (
    ABLATION_RUNS,
    resolve_ablation_run,
)


def test_ablation_matrix_all_runs_present() -> None:
    """Ensure all seven B-series performance experiments are resolvable."""

    expected_keys = [
        "B0_legacy_v9",
        "B1_flat_components",
        "B2_asr_interval",
        "B3_robust_calibration",
        "B4_confidence_gating",
        "B5_adaptive_p0",
        "B6_dense_only",
    ]
    assert list(ABLATION_RUNS.keys()) == expected_keys

    for i, full_name in enumerate(expected_keys):
        short_key = f"B{i}"
        run = resolve_ablation_run(short_key)
        assert run.name == full_name
        assert resolve_ablation_run(full_name).name == full_name


def test_runs_do_not_own_or_replace_loaded_alignment_config() -> None:
    """Emission experiments must leave the service's loaded DP config untouched."""

    assert all(not hasattr(run, "alignment") for run in ABLATION_RUNS.values())


def test_single_feature_delta_between_ablation_stages() -> None:
    """Verify sequential B-stage settings after the explicit fusion baseline."""

    runs = list(ABLATION_RUNS.values())

    # B0 -> B1 intentionally changes the fusion equation. It is not presented
    # as an exact componentized-legacy ablation.
    a0, a1 = runs[0], runs[1]
    assert a0.fusion_mode == "legacy"
    assert a1.fusion_mode == "adaptive_p0"
    assert a0.robust_calibration == a1.robust_calibration is False
    assert a0.confidence_gating == a1.confidence_gating is False
    assert a0.event_routing == a1.event_routing is False
    assert a0.interval_projection == a1.interval_projection is False
    assert a0.use_bm25 == a1.use_bm25 is True

    # A1 -> A2: only interval_projection changes from False to True
    a1, a2 = runs[1], runs[2]
    assert a1.interval_projection is False
    assert a2.interval_projection is True
    assert a1.fusion_mode == a2.fusion_mode == "adaptive_p0"
    assert a1.robust_calibration == a2.robust_calibration is False
    assert a1.confidence_gating == a2.confidence_gating is False
    assert a1.event_routing == a2.event_routing is False
    assert a1.use_bm25 == a2.use_bm25 is True

    # A2 -> A3: only robust_calibration changes from False to True
    a2, a3 = runs[2], runs[3]
    assert a2.robust_calibration is False
    assert a3.robust_calibration is True
    assert a2.fusion_mode == a3.fusion_mode == "adaptive_p0"
    assert a2.confidence_gating == a3.confidence_gating is False
    assert a2.event_routing == a3.event_routing is False
    assert a2.interval_projection == a3.interval_projection is True
    assert a2.use_bm25 == a3.use_bm25 is True

    # A3 -> A4: only confidence_gating changes from False to True
    a3, a4 = runs[3], runs[4]
    assert a3.confidence_gating is False
    assert a4.confidence_gating is True
    assert a3.fusion_mode == a4.fusion_mode == "adaptive_p0"
    assert a3.robust_calibration == a4.robust_calibration is True
    assert a3.event_routing == a4.event_routing is False
    assert a3.interval_projection == a4.interval_projection is True
    assert a3.use_bm25 == a4.use_bm25 is True

    # A4 -> A5: only event_routing changes from False to True (Full P0)
    a4, a5 = runs[4], runs[5]
    assert a4.event_routing is False
    assert a5.event_routing is True
    assert a4.fusion_mode == a5.fusion_mode == "adaptive_p0"
    assert a4.robust_calibration == a5.robust_calibration is True
    assert a4.confidence_gating == a5.confidence_gating is True
    assert a4.interval_projection == a5.interval_projection is True
    assert a4.use_bm25 == a5.use_bm25 is True

    # A5 -> A6: only use_bm25 changes from True to False (Dense-only)
    a5, a6 = runs[5], runs[6]
    assert a5.use_bm25 is True
    assert a6.use_bm25 is False
    assert a5.fusion_mode == a6.fusion_mode == "adaptive_p0"
    assert a5.robust_calibration == a6.robust_calibration is True
    assert a5.confidence_gating == a6.confidence_gating is True
    assert a5.interval_projection == a6.interval_projection is True
    assert a5.event_routing == a6.event_routing is True


def test_apply_to_preserves_all_baseline_weights_and_boosts() -> None:
    """Applying a run changes only its switches and keeps tuned baseline values."""

    a5 = ABLATION_RUNS["B5_adaptive_p0"]
    baseline = HybridTemporalConfig(
        dense_weight=0.7,
        bm25_weight=0.3,
        adaptive=AdaptiveTemporalFusionConfig(
            calibration=RobustCalibrationConfig(q_low=0.1, q_high=0.8),
            base_component_weights={"visual_dense": 0.8, "bm25_ocr": 0.2},
            visual_boost=1.7,
            speech_boost=2.5,
            ocr_boost=4.0,
            robust_calibration=False,
            confidence_gating=False,
            event_routing=False,
            asr_interval_projection=False,
        ),
    )

    hybrid = a5.apply_to(baseline)

    assert hybrid.fusion_mode == "adaptive_p0"
    assert hybrid.adaptive.robust_calibration is True
    assert hybrid.adaptive.confidence_gating is True
    assert hybrid.adaptive.event_routing is True
    assert hybrid.dense_weight == 0.7
    assert hybrid.bm25_weight == 0.3
    assert (
        hybrid.adaptive.base_component_weights
        == baseline.adaptive.base_component_weights
    )
    assert hybrid.adaptive.calibration == baseline.adaptive.calibration
    assert hybrid.adaptive.visual_boost == 1.7
    assert hybrid.adaptive.speech_boost == 2.5
    assert hybrid.adaptive.ocr_boost == 4.0
    assert baseline.fusion_mode == "legacy"
    assert baseline.adaptive.event_routing is False

    a0 = ABLATION_RUNS["B0_legacy_v9"]
    hybrid_a0 = a0.apply_to(baseline)
    assert hybrid_a0.fusion_mode == "legacy"


def test_resolve_ablation_run_unknown_key() -> None:
    """Verify error on invalid run key."""

    with pytest.raises(KeyError, match="Unknown ablation run"):
        resolve_ablation_run("B99")
