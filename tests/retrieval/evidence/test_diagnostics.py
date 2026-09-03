"""Tests for component evidence diagnostics and telemetry."""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.common.config import HybridTemporalConfig, RobustCalibrationConfig
from hcmai.retrieval.evidence.calibration import CalibratedComponent
from hcmai.retrieval.evidence.components import (
    TemporalScoreBundle,
    TemporalScoreComponent,
)
from hcmai.retrieval.evidence.diagnostics import (
    ComponentEventDebug,
    TemporalEvidenceDebugResult,
    build_evidence_diagnostics,
)
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer


def test_build_evidence_diagnostics_coverage_and_positions() -> None:
    """Check coverage ratio calculation and deterministic top positions."""

    raw_scores = np.asarray([[0.1, 0.9, 0.4]], dtype=np.float32)
    calibrated_scores = np.asarray([[0.0, 1.0, 0.375]], dtype=np.float32)
    coverage = np.asarray([False, True, False])

    bundle = TemporalScoreBundle(
        {
            "asr_dense": TemporalScoreComponent(
                "asr_dense",
                raw_scores,
                coverage=coverage,
            )
        }
    )
    calibrated = {
        "asr_dense": CalibratedComponent(
            scores=calibrated_scores,
            reliability=np.asarray([0.8], dtype=np.float32),
        )
    }
    fused_scores = calibrated_scores.copy()

    result = build_evidence_diagnostics(
        bundle=bundle,
        calibrated=calibrated,
        fused_scores=fused_scores,
        top_positions=2,
    )

    assert isinstance(result, TemporalEvidenceDebugResult)
    np.testing.assert_allclose(result.fused_scores, fused_scores)
    assert len(result.rows) == 1

    row = result.rows[0]
    assert isinstance(row, ComponentEventDebug)
    assert row.component == "asr_dense"
    assert row.event_index == 0
    assert pytest.approx(row.raw_max, abs=1e-5) == 0.9
    assert pytest.approx(row.raw_median, abs=1e-5) == 0.4
    assert pytest.approx(row.calibrated_max, abs=1e-5) == 1.0
    assert pytest.approx(row.reliability, abs=1e-5) == 0.8
    assert pytest.approx(row.coverage_ratio, abs=1e-5) == 1.0 / 3.0
    # Highest calibrated score is index 1 (1.0), second highest is index 2 (0.375)
    assert row.top_positions == (1, 2)


class MockVisualIndex:
    frame_ids = np.asarray(["f1", "f2", "f3"])
    video_ids = np.asarray(["v1", "v1", "v2"])
    frame_idx = np.asarray([10, 20, 30])
    timestamps = np.asarray([1000, 2000, 3000])

    def video_positions(self, video_id: str) -> np.ndarray:
        return np.flatnonzero(self.video_ids == video_id)


class MockComponentScorer:
    def __init__(self, name: str, scores: np.ndarray, coverage: np.ndarray | None = None) -> None:
        self.name = name
        self.scores = scores
        self.coverage = coverage

    def score_components(self, *events: object) -> TemporalScoreBundle:
        return TemporalScoreBundle(
            {
                self.name: TemporalScoreComponent(
                    self.name,
                    self.scores.copy(),
                    coverage=self.coverage,
                )
            }
        )


def test_debug_score_events_integration() -> None:
    """Verify debug_score_events on TemporalEvidenceScorer."""

    scorer = TemporalEvidenceScorer(
        visual_index=MockVisualIndex(),
        dense=MockComponentScorer(
            "visual_dense",
            np.asarray([[0.2, 0.8, 0.4]], dtype=np.float32),
        ),
        bm25=MockComponentScorer(
            "bm25_ocr",
            np.asarray([[1.0, 5.0, 3.0]], dtype=np.float32),
        ),
        config=HybridTemporalConfig(fusion_mode="adaptive_p0"),
    )

    debug_result = scorer.debug_score_events(
        original_events=["dòng chữ"],
        retrieval_events=["text"],
        caption_events=["caption"],
        use_dense=True,
        use_bm25=True,
        top_positions=3,
    )

    assert isinstance(debug_result, TemporalEvidenceDebugResult)
    assert debug_result.fused_scores.shape == (1, 3)
    assert len(debug_result.rows) == 2  # visual_dense, bm25_ocr

    row_names = {r.component for r in debug_result.rows}
    assert row_names == {"visual_dense", "bm25_ocr"}


def test_debug_score_events_validates_inputs() -> None:
    """Verify validation errors are raised on mismatched inputs."""

    scorer = TemporalEvidenceScorer(
        visual_index=MockVisualIndex(),
        dense=MockComponentScorer("visual_dense", np.zeros((1, 3), dtype=np.float32)),
        bm25=None,
        config=HybridTemporalConfig(),
    )

    with pytest.raises(ValueError, match="at least one"):
        scorer.debug_score_events(["e1"], ["e1"], use_dense=False, use_bm25=False)

    with pytest.raises(ValueError, match="counts must match"):
        scorer.debug_score_events(["e1", "e2"], ["e1"], use_dense=True, use_bm25=False)
