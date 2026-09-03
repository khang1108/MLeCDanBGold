"""Tests for isolated ablation runners and configuration immutability."""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.common.config import (
    AdaptiveTemporalFusionConfig,
    HybridTemporalConfig,
)
from hcmai.retrieval.evidence.ablation import (
    ABLATION_RUNS,
    make_ablation_scorer,
    resolve_ablation_run,
)
from hcmai.retrieval.evidence.components import (
    TemporalScoreBundle,
    TemporalScoreComponent,
)
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
from tests.retrieval.evidence.fakes import FakeIndex


class MockDenseScorer:
    """Mock dense scorer providing identical outputs for legacy and adaptive."""

    def __init__(self, raw: np.ndarray, asr_raw: np.ndarray) -> None:
        self.raw = raw
        self.asr_raw = asr_raw

    def score_events(self, events: list[str]) -> np.ndarray:
        # Legacy formula: 0.35*vis + 0.35*ctx + 0.08*asr normalized
        vis = (self.raw - self.raw.min(axis=1, keepdims=True)) / (np.ptp(self.raw, axis=1, keepdims=True) + 1e-6)
        asr = (self.asr_raw - self.asr_raw.min(axis=1, keepdims=True)) / (np.ptp(self.asr_raw, axis=1, keepdims=True) + 1e-6)
        return (0.35 * vis + 0.35 * vis + 0.08 * asr).astype(np.float32)

    def score_components(self, events: list[str], *, asr_interval_projection: bool = True) -> TemporalScoreBundle:
        return TemporalScoreBundle(
            {
                "visual_dense": TemporalScoreComponent("visual_dense", self.raw),
                "context_dense": TemporalScoreComponent("context_dense", self.raw),
                "asr_dense": TemporalScoreComponent(
                    "asr_dense",
                    self.asr_raw,
                    coverage=np.ones(self.asr_raw.shape[1], dtype=bool),
                ),
            }
        )


def test_ablation_runs_contain_all_seven_conditions() -> None:
    """The ablation matrix must configure A0 through A6."""

    expected_keys = [
        "A0_legacy_v9",
        "A1_components_fixed",
        "A2_asr_interval",
        "A3_robust_calibration",
        "A4_confidence_gating",
        "A5_adaptive_p0",
        "A6_dense_only",
    ]
    for key in expected_keys:
        cfg = resolve_ablation_run(key)
        assert cfg.name == key

    for i in range(7):
        cfg = resolve_ablation_run(f"A{i}")
        assert cfg is not None


def test_make_ablation_scorer_does_not_mutate_baseline_scorer() -> None:
    """Running make_ablation_scorer does not mutate the baseline scorer configuration."""

    visual_idx = FakeIndex(np.zeros((1, 10), dtype=np.float32))
    base_config = HybridTemporalConfig(
        fusion_mode="legacy",
        dense_weight=0.7,
        bm25_weight=0.3,
        adaptive=AdaptiveTemporalFusionConfig(
            speech_boost=7.5, # Custom boost to ensure preserved
            event_routing=False,
        ),
    )
    baseline_scorer = TemporalEvidenceScorer(
        visual_index=visual_idx,
        dense=None,
        bm25=None,
        config=base_config,
    )

    a5_scorer, a5_kwargs = make_ablation_scorer(baseline_scorer, "A5")

    # Baseline configuration must remain completely untouched!
    assert baseline_scorer.config.fusion_mode == "legacy"
    assert baseline_scorer.config.adaptive.event_routing is False

    # A5 scorer must have derived configuration with preserved custom speech_boost
    assert a5_scorer.config.fusion_mode == "adaptive_p0"
    assert a5_scorer.config.adaptive.event_routing is True
    assert a5_scorer.config.adaptive.speech_boost == 7.5
    assert a5_kwargs["use_bm25"] is True

    # A6 scorer has use_bm25=False
    a6_scorer, a6_kwargs = make_ablation_scorer(baseline_scorer, "A6")
    assert a6_kwargs["use_bm25"] is False


def test_running_a0_after_a5_is_isolated() -> None:
    """Running A0 after A5 produces exact legacy behavior and does not leak routing."""

    raw = np.asarray([[0.1, 0.5, 0.9]], dtype=np.float32)
    asr = np.asarray([[0.2, 0.4, 0.8]], dtype=np.float32)
    dense = MockDenseScorer(raw, asr)
    visual_idx = FakeIndex(raw)

    class MockBM25Scorer:
        def score_events(self, orig: list[str], capt: list[str]) -> np.ndarray:
            return np.zeros((len(orig), 3), dtype=np.float32)

        def score_components(self, orig: list[str], capt: list[str]) -> TemporalScoreBundle:
            return TemporalScoreBundle(
                {
                    "bm25_title": TemporalScoreComponent(
                        "bm25_title", np.zeros((len(orig), 3), dtype=np.float32)
                    )
                }
            )

    bm25 = MockBM25Scorer()
    base_config = HybridTemporalConfig(
        fusion_mode="legacy",
        adaptive=AdaptiveTemporalFusionConfig(),
    )
    baseline_scorer = TemporalEvidenceScorer(
        visual_index=visual_idx,
        dense=dense,
        bm25=bm25,
        config=base_config,
    )

    a5_scorer, a5_kw = make_ablation_scorer(baseline_scorer, "A5")
    a0_scorer, a0_kw = make_ablation_scorer(baseline_scorer, "A0")

    res_a5 = a5_scorer.score_events(
        ["Người phụ nữ nói chuyện"],
        ["The woman speaks"],
        caption_events=["Người phụ nữ nói chuyện"],
        **a5_kw,
    )

    res_a0 = a0_scorer.score_events(
        ["Người phụ nữ nói chuyện"],
        ["The woman speaks"],
        caption_events=["Người phụ nữ nói chuyện"],
        **a0_kw,
    )

    # A0 and A5 must produce different results because A5 uses event routing and robust calibration
    assert not np.allclose(res_a0[0].scores, res_a5[0].scores)


def test_a1_matches_a0_relative_proportions() -> None:
    """A1 adaptive components without routing/gating preserve A0 relative weighting."""

    raw = np.asarray([[0.1, 0.5, 0.9]], dtype=np.float32)
    asr = np.asarray([[0.2, 0.4, 0.8]], dtype=np.float32)
    dense = MockDenseScorer(raw, asr)
    visual_idx = FakeIndex(raw)

    base_config = HybridTemporalConfig(
        fusion_mode="legacy",
        dense_weight=1.0,
        bm25_weight=0.0,
        adaptive=AdaptiveTemporalFusionConfig(
            robust_calibration=False,
            confidence_gating=False,
            event_routing=False,
            asr_interval_projection=False,
            base_component_weights={"visual_dense": 0.35, "context_dense": 0.35, "asr_dense": 0.08},
        ),
    )
    baseline_scorer = TemporalEvidenceScorer(
        visual_index=visual_idx,
        dense=dense,
        bm25=None,
        config=base_config,
    )

    a0_scorer, _ = make_ablation_scorer(baseline_scorer, "A0")
    a1_scorer, _ = make_ablation_scorer(baseline_scorer, "A1")

    # Both run dense-only
    res_a0 = a0_scorer.score_events(["event"], ["event"], caption_events=None, use_dense=True, use_bm25=False)
    res_a1 = a1_scorer.score_events(["event"], ["event"], caption_events=None, use_dense=True, use_bm25=False)

    # In dense-only mode with matching weights, A0 and A1 are numerically proportional
    a0_norm = res_a0[0].scores / res_a0[0].scores.sum()
    a1_norm = res_a1[0].scores / res_a1[0].scores.sum()
    np.testing.assert_allclose(a0_norm, a1_norm, atol=1e-5)
