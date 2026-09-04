"""Tests for selectable Dense/BM25 temporal evidence fusion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from typing import Any

from hcmai.common.config import AdaptiveTemporalFusionConfig, HybridTemporalConfig
from hcmai.retrieval.evidence.components import (
    TemporalScoreBundle,
    TemporalScoreComponent,
)
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer


class MatrixScorer:
    """Return one scripted canonical score matrix."""

    def __init__(self, scores: np.ndarray) -> None:
        self.scores = scores

    def score_events(self, *events: object) -> np.ndarray:
        return self.scores.copy()


class FakeIndex:
    frame_ids = np.asarray(["f1", "f2", "f3"])
    video_ids = np.asarray(["v1", "v1", "v2"])
    frame_idx = np.asarray([1, 2, 3])
    timestamps = np.asarray([100, 200, 300])

    def video_positions(self, video_id: str) -> np.ndarray:
        return np.flatnonzero(self.video_ids == video_id)


def _matrix(result: Sequence[Any]) -> np.ndarray:
    return np.concatenate([item.scores for item in result], axis=1)


@pytest.mark.parametrize(
    ("use_dense", "use_bm25", "expected"),
    [
        (True, False, [[0.2, 0.4, 0.8]]),
        (False, True, [[0.0, 0.5, 1.0]]),
        (True, True, [[0.1, 0.45, 0.9]]),
],)
def test_three_fusion_modes(use_dense: bool, use_bm25: bool, expected: list[list[float]]) -> None:
    """Route Dense-only, BM25-only, and neutral hybrid modes exactly."""

    scorer = TemporalEvidenceScorer(
        visual_index=FakeIndex(),
        dense=MatrixScorer(np.asarray([[0.2, 0.4, 0.8]], dtype=np.float32)),
        bm25=MatrixScorer(np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32)),
        config=HybridTemporalConfig(fusion_mode="legacy"),
    )

    result = scorer.score_events(
        ("vi",),
        ("retrieval",),
        caption_events=("caption",),
        use_dense=use_dense,
        use_bm25=use_bm25,
    )

    np.testing.assert_allclose(_matrix(result), expected)


def test_both_off_and_event_mismatch_are_rejected() -> None:
    scorer = TemporalEvidenceScorer(
        visual_index=FakeIndex(),
        dense=MatrixScorer(np.zeros((1, 3), dtype=np.float32)),
        bm25=MatrixScorer(np.zeros((1, 3), dtype=np.float32)),
        config=HybridTemporalConfig(fusion_mode="legacy"),
    )

    with pytest.raises(ValueError, match="at least one"):
        scorer.score_events(
            ("vi",), ("en",), caption_events=("en",), use_dense=False, use_bm25=False
        )
    with pytest.raises(ValueError, match="event counts"):
        scorer.score_events(
            ("vi", "vi2"), ("en",), caption_events=("en",), use_dense=True, use_bm25=False
        )


def test_missing_bm25_degrades_instead_of_failing() -> None:
    """A missing BM25 artifact must fall back to Dense, not raise."""

    scorer = TemporalEvidenceScorer(
        visual_index=FakeIndex(),
        dense=MatrixScorer(np.asarray([[0.2, 0.4, 0.8]], dtype=np.float32)),
        bm25=None,
        config=HybridTemporalConfig(fusion_mode="legacy"),
    )

    result = scorer.score_events(
        ("vi",), ("retrieval",), caption_events=("caption",), use_dense=True, use_bm25=True
    )

    np.testing.assert_allclose(_matrix(result), [[0.2, 0.4, 0.8]])
    with pytest.raises(ValueError, match="at least one"):
        scorer.score_events(
            ("vi",), ("en",), caption_events=("en",), use_dense=False, use_bm25=True
        )


class ComponentScorer:
    """Return scripted component bundle."""

    def __init__(self, name: str, scores: np.ndarray) -> None:
        self.name = name
        self.scores = scores

    def score_components(self, *events: object) -> TemporalScoreBundle:
        return TemporalScoreBundle(
            {self.name: TemporalScoreComponent(self.name, self.scores.copy())}
        )


def test_adaptive_fusion_mode_runs_successfully() -> None:
    """Adaptive fusion mode splits scores into VideoEventScores correctly."""

    scorer = TemporalEvidenceScorer(
        visual_index=FakeIndex(),
        dense=ComponentScorer("visual_dense", np.asarray([[0.2, 0.4, 0.8]], dtype=np.float32)),
        bm25=ComponentScorer("bm25_caption", np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32)),
        config=HybridTemporalConfig(fusion_mode="adaptive_p0"),
    )

    result = scorer.score_events(
        ("vi",),
        ("retrieval",),
        caption_events=("caption",),
        use_dense=True,
        use_bm25=True,
    )

    assert len(result) == 2
    matrix = _matrix(result)
    assert matrix.shape == (1, 3)
    assert np.all(np.isfinite(matrix))

def test_config_reassignment_reaches_adaptive_fusion() -> None:
    """Swapping config after construction must re-route adaptive fusion, not use stale settings."""

    scorer = TemporalEvidenceScorer(
        visual_index=FakeIndex(),
        dense=ComponentScorer("visual_dense", np.asarray([[0.2, 0.4, 0.8]], dtype=np.float32)),
        bm25=ComponentScorer("bm25_caption", np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32)),
        config=HybridTemporalConfig(fusion_mode="adaptive_p0"),
    )
    events = (("vi",), ("retrieval",))
    routed = _matrix(scorer.score_events(*events, caption_events=("caption",), use_dense=True, use_bm25=True))

    scorer.config = HybridTemporalConfig(
        fusion_mode="adaptive_p0",
        adaptive=AdaptiveTemporalFusionConfig(base_component_weights={"visual_dense": 1.0}),
    )
    visual_only = _matrix(
        scorer.score_events(*events, caption_events=("caption",), use_dense=True, use_bm25=True)
    )

    assert not np.allclose(routed, visual_only)
