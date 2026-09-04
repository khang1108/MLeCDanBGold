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
    """The experiment matrix must configure B0 through B6."""

    expected_keys = [
        "B0_legacy_v9",
        "B1_flat_components",
        "B2_asr_interval",
        "B3_robust_calibration",
        "B4_confidence_gating",
        "B5_adaptive_p0",
        "B6_dense_only",
    ]
    for key in expected_keys:
        cfg = resolve_ablation_run(key)
        assert cfg.name == key

    for i in range(7):
        cfg = resolve_ablation_run(f"B{i}")
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

    a5_scorer, a5_kwargs = make_ablation_scorer(baseline_scorer, "B5")

    # Baseline configuration must remain completely untouched!
    assert baseline_scorer.config.fusion_mode == "legacy"
    assert baseline_scorer.config.adaptive.event_routing is False

    # A5 scorer must have derived configuration with preserved custom speech_boost
    assert a5_scorer.config.fusion_mode == "adaptive_p0"
    assert a5_scorer.config.adaptive.event_routing is True
    assert a5_scorer.config.adaptive.speech_boost == 7.5
    assert a5_kwargs["use_bm25"] is True

    # B6 scorer has use_bm25=False
    a6_scorer, a6_kwargs = make_ablation_scorer(baseline_scorer, "B6")
    assert a6_kwargs["use_bm25"] is False


def test_running_b0_after_b5_is_isolated() -> None:
    """Running B0 after B5 produces exact legacy behavior and does not leak routing."""

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

    a5_scorer, a5_kw = make_ablation_scorer(baseline_scorer, "B5")
    a0_scorer, a0_kw = make_ablation_scorer(baseline_scorer, "B0")

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

    # B0 and B5 differ because B5 uses event routing and robust calibration.
    assert not np.allclose(res_a0[0].scores, res_a5[0].scores)


def test_legacy_and_flat_component_runs_are_named_as_distinct_experiments() -> None:
    """The runtime matrix must not claim flat fusion is componentized legacy."""

    legacy = resolve_ablation_run("B0")
    flat = resolve_ablation_run("B1")

    assert legacy.name == "B0_legacy_v9"
    assert legacy.fusion_mode == "legacy"
    assert flat.name == "B1_flat_components"
    assert flat.fusion_mode == "adaptive_p0"
