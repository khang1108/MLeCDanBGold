"""Regression test for the P0 ablation matrix configurations."""

from __future__ import annotations

import pytest

from hcmai.retrieval.evidence.ablation import (
    ABLATION_RUNS,
    resolve_ablation_run,
)


def test_ablation_matrix_all_runs_present() -> None:
    """Ensure all 7 named stages A0-A6 are defined and resolvable by aliases."""

    expected_keys = [
        "A0_legacy_v9",
        "A1_components_fixed",
        "A2_asr_interval",
        "A3_robust_calibration",
        "A4_confidence_gating",
        "A5_adaptive_p0",
        "A6_dense_only",
    ]
    assert list(ABLATION_RUNS.keys()) == expected_keys

    for i, full_name in enumerate(expected_keys):
        short_key = f"A{i}"
        run = resolve_ablation_run(short_key)
        assert run.name == full_name
        assert resolve_ablation_run(full_name).name == full_name


def test_alignment_config_is_identical_across_all_stages() -> None:
    """Assert DP alignment settings are strictly identical across A0-A6."""

    base_alignment = ABLATION_RUNS["A0_legacy_v9"].alignment

    for name, run in ABLATION_RUNS.items():
        assert run.alignment == base_alignment, (
            f"Run {name} modified AlignmentConfig: {run.alignment} != {base_alignment}"
        )


def test_single_feature_delta_between_ablation_stages() -> None:
    """Verify that each sequential stage A0->A6 differs by exactly one intended feature."""

    runs = list(ABLATION_RUNS.values())

    # A0 -> A1: only fusion_mode changes from "legacy" to "adaptive_p0"
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


def test_to_hybrid_config_converts_properly() -> None:
    """Verify that to_hybrid_config correctly sets fusion_mode and adaptive sub-flags."""

    a5 = ABLATION_RUNS["A5_adaptive_p0"]
    hybrid = a5.to_hybrid_config()
    assert hybrid.fusion_mode == "adaptive_p0"
    assert hybrid.adaptive.robust_calibration is True
    assert hybrid.adaptive.confidence_gating is True
    assert hybrid.adaptive.event_routing is True

    a0 = ABLATION_RUNS["A0_legacy_v9"]
    hybrid_a0 = a0.to_hybrid_config()
    assert hybrid_a0.fusion_mode == "legacy"


def test_resolve_ablation_run_unknown_key() -> None:
    """Verify error on invalid run key."""

    with pytest.raises(KeyError, match="Unknown ablation run"):
        resolve_ablation_run("A99")
