"""Tests for selecting isolated B-series temporal diagnostic configurations."""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.common.config import AdaptiveTemporalFusionConfig, HybridTemporalConfig
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
from scripts.debug_temporal_evidence import _configure_debug_run, parse_args
from tests.retrieval.evidence.fakes import FakeIndex


def _baseline_scorer() -> TemporalEvidenceScorer:
    """Build a scorer whose non-default values reveal accidental config resets."""

    config = HybridTemporalConfig(
        fusion_mode="legacy",
        dense_weight=0.7,
        bm25_weight=0.3,
        adaptive=AdaptiveTemporalFusionConfig(
            speech_boost=2.25,
            base_component_weights={"visual_dense": 0.9, "bm25_asr": 0.1},
        ),
    )
    return TemporalEvidenceScorer(
        visual_index=FakeIndex(np.zeros((1, 3), dtype=np.float32)),
        dense=None,
        bm25=None,
        config=config,
    )


def test_parse_args_accepts_b_series_run() -> None:
    """The diagnostic CLI exposes an explicit run selector."""

    args = parse_args(["--query-file", "query.yaml", "--run", "B3"])

    assert args.run == "B3"


def test_configure_debug_run_isolated_and_preserves_baseline() -> None:
    """B3 selection preserves tuned values and does not mutate the baseline."""

    baseline = _baseline_scorer()
    scorer, use_dense, use_bm25 = _configure_debug_run(
        baseline,
        "B3",
        use_dense=True,
        use_bm25=True,
    )

    assert scorer is not baseline
    assert scorer.config.adaptive.robust_calibration is True
    assert scorer.config.adaptive.confidence_gating is False
    assert scorer.config.adaptive.event_routing is False
    assert scorer.config.adaptive.speech_boost == 2.25
    assert scorer.config.adaptive.base_component_weights == {
        "visual_dense": 0.9,
        "bm25_asr": 0.1,
    }
    assert use_dense is True
    assert use_bm25 is True
    assert baseline.config.fusion_mode == "legacy"
    assert baseline.config.adaptive.event_routing is True


def test_configure_debug_run_respects_dense_only_stage() -> None:
    """B6 disables BM25 even when the CLI source switch is enabled."""

    _, use_dense, use_bm25 = _configure_debug_run(
        _baseline_scorer(),
        "B6",
        use_dense=True,
        use_bm25=True,
    )

    assert use_dense is True
    assert use_bm25 is False


def test_configure_debug_run_rejects_legacy_stage() -> None:
    """B0 cannot claim component diagnostics equivalent to legacy scoring."""

    with pytest.raises(ValueError, match="choose B1-B6"):
        _configure_debug_run(
            _baseline_scorer(),
            "B0",
            use_dense=True,
            use_bm25=True,
        )
