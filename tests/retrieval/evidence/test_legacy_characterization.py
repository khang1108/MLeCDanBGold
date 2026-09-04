"""Characterization tests for v9 temporal evidence scoring behavior."""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.common.config import (
    AdaptiveTemporalFusionConfig,
    BM25FieldWeights,
    DenseTemporalWeights,
    HybridTemporalConfig,
)
from hcmai.retrieval.evidence.dense import DenseTemporalScorer
from hcmai.retrieval.evidence.components import (
    TemporalScoreBundle,
    TemporalScoreComponent,
)
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
from hcmai.retrieval.evidence.normalization import minmax_rows
from tests.retrieval.evidence.fakes import FakeEncoder, FakeIndex


def reference_v9_asr(
    segment_scores: np.ndarray,       # [E, S]
    projected_positions: np.ndarray,  # [S], -1 allowed
    frame_count: int,
) -> np.ndarray:
    out = np.full((segment_scores.shape[0], frame_count), -np.inf, dtype=np.float32)
    valid = projected_positions >= 0
    for e in range(segment_scores.shape[0]):
        np.maximum.at(out[e], projected_positions[valid], segment_scores[e, valid])
        covered = np.isfinite(out[e])
        if not np.any(covered):
            out[e].fill(0.0)
        else:
            out[e, ~covered] = float(out[e, covered].min())
    return out


def test_minmax_rows_stretches_each_nonconstant_event_independently() -> None:
    raw = np.asarray([[0.20, 0.21, 0.22], [10.0, 20.0, 30.0]], dtype=np.float32)

    actual = minmax_rows(raw)

    np.testing.assert_allclose(
        actual,
        np.asarray([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_minmax_rows_turns_constant_event_into_zero() -> None:
    actual = minmax_rows(np.asarray([[0.3, 0.3, 0.3]], dtype=np.float32))
    np.testing.assert_array_equal(actual, np.zeros((1, 3), dtype=np.float32))


def test_reference_v9_asr_matches_point_and_floor_fill() -> None:
    segment_scores = np.asarray([[0.8, 0.4]], dtype=np.float32)
    projected = np.asarray([1, -1])  # Only frame 1 covered
    out = reference_v9_asr(segment_scores, projected, frame_count=3)
    np.testing.assert_allclose(out, np.asarray([[0.8, 0.8, 0.8]], dtype=np.float32))

    segment_scores_multi = np.asarray([[0.9, 0.5]], dtype=np.float32)
    projected_multi = np.asarray([0, 2])  # Frames 0 and 2 covered, frame 1 uncovered
    out_multi = reference_v9_asr(segment_scores_multi, projected_multi, frame_count=3)
    np.testing.assert_allclose(out_multi, np.asarray([[0.9, 0.5, 0.5]], dtype=np.float32))


def test_dense_legacy_normalizes_each_modality_then_averages() -> None:
    visual_raw = np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32)
    context_raw = np.asarray([[10.0, 10.0, 12.0]], dtype=np.float32)
    asr_raw = np.asarray([[0.70, 0.71, 0.72]], dtype=np.float32)

    visual = FakeIndex(visual_raw)
    context = FakeIndex(context_raw)
    asr = FakeIndex(asr_raw)
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    weights = DenseTemporalWeights(visual_weight=0.4, context_weight=0.3, asr_weight=0.3)
    scorer = DenseTemporalScorer(
        visual_index=visual,
        context_index=context,
        asr_index=asr,
        visual_encoder=encoder,
        text_encoder=encoder,
        weights=weights,
        chunk_size=8,
    )

    actual = scorer.score_events(["event"])

    expected = (
        0.4 * minmax_rows(visual_raw)
        + 0.3 * minmax_rows(context_raw)
        + 0.3 * minmax_rows(asr_raw)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class StaticDense:
    def __init__(self, scores: np.ndarray | None = None) -> None:
        self._scores = (
            scores
            if scores is not None
            else np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)
        )

    def score_events(self, events):
        del events
        return self._scores


class StaticBM25:
    def __init__(self, scores: np.ndarray | None = None) -> None:
        self._scores = (
            scores
            if scores is not None
            else np.asarray([[2.0, 3.0, 6.0]], dtype=np.float32)
        )

    def score_events(self, original, caption):
        del original, caption
        return self._scores


class ComponentizedBM25:
    """Expose deterministic field components and their legacy weighted sum."""

    def __init__(
        self,
        components: dict[str, np.ndarray],
        weights: BM25FieldWeights,
    ) -> None:
        self._components = components
        self._weights = weights

    def score_components(self, original, caption) -> TemporalScoreBundle:
        """Return raw lexical field matrices without changing their scale."""

        del original, caption
        return TemporalScoreBundle(
            {
                name: TemporalScoreComponent(name, scores)
                for name, scores in self._components.items()
            }
        )

    def score_events(self, original, caption) -> np.ndarray:
        """Reproduce the v9 field-weighted BM25 sum."""

        bundle = self.score_components(original, caption)
        return np.asarray(
            self._weights.title_weight
            * bundle.components["bm25_title"].raw_scores
            + self._weights.caption_weight
            * bundle.components["bm25_caption"].raw_scores
            + self._weights.ocr_weight
            * bundle.components["bm25_ocr"].raw_scores
            + self._weights.asr_weight
            * bundle.components["bm25_asr"].raw_scores,
            dtype=np.float32,
        )


def test_hybrid_legacy_minmaxes_bm25_then_uses_half_half_weights() -> None:
    visual = FakeIndex(np.zeros((1, 3), dtype=np.float32))
    scorer = TemporalEvidenceScorer(
        visual_index=visual,
        dense=StaticDense(),
        bm25=StaticBM25(),
        config=HybridTemporalConfig(fusion_mode="legacy"),
    )

    videos = scorer.score_events(
        ["vi"],
        ["en"],
        caption_events=["vi caption"],
        use_dense=True,
        use_bm25=True,
    )

    expected_bm25 = np.asarray([[0.0, 0.25, 1.0]], dtype=np.float32)
    expected = 0.5 * np.asarray([[0.0, 0.5, 1.0]]) + 0.5 * expected_bm25
    np.testing.assert_allclose(videos[0].scores, expected, rtol=1e-6, atol=1e-6)


def test_componentized_raw_scores_recombine_to_exact_legacy_hybrid() -> None:
    """Raw components reproduce v9 only when recombined with the v9 equation."""

    visual_raw = np.asarray([[0.2, 0.9, 0.4, 0.3]], dtype=np.float32)
    context_raw = np.asarray([[10.0, 12.0, 50.0, 20.0]], dtype=np.float32)
    asr_raw = np.asarray([[0.8, 0.1, 0.5, 0.4]], dtype=np.float32)
    dense_weights = DenseTemporalWeights(
        visual_weight=0.2,
        context_weight=0.3,
        asr_weight=0.5,
    )
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    dense = DenseTemporalScorer(
        visual_index=FakeIndex(visual_raw),
        context_index=FakeIndex(context_raw),
        asr_index=FakeIndex(asr_raw),
        visual_encoder=encoder,
        text_encoder=encoder,
        weights=dense_weights,
    )

    bm25_weights = BM25FieldWeights(
        title_weight=0.1,
        caption_weight=0.2,
        ocr_weight=0.3,
        asr_weight=0.4,
    )
    bm25 = ComponentizedBM25(
        {
            "bm25_title": np.asarray([[0.0, 9.0, 0.0, 0.0]], dtype=np.float32),
            "bm25_caption": np.asarray([[1.0, 0.0, 3.0, 0.0]], dtype=np.float32),
            "bm25_ocr": np.asarray([[0.0, 0.0, 4.0, 4.0]], dtype=np.float32),
            "bm25_asr": np.asarray([[2.0, 1.0, 0.0, 5.0]], dtype=np.float32),
        },
        bm25_weights,
    )
    config = HybridTemporalConfig(
        fusion_mode="legacy", dense_weight=0.4, bm25_weight=0.6
    )
    scorer = TemporalEvidenceScorer(
        visual_index=dense.visual_index,
        dense=dense,
        bm25=bm25,
        config=config,
    )

    dense_components = dense.score_components(
        ["event"],
        asr_interval_projection=False,
    ).components
    bm25_components = bm25.score_components(["sự kiện"], ["mô tả"]).components
    recombined_dense = (
        dense_weights.visual_weight
        * minmax_rows(dense_components["visual_dense"].raw_scores)
        + dense_weights.context_weight
        * minmax_rows(dense_components["context_dense"].raw_scores)
        + dense_weights.asr_weight
        * minmax_rows(dense_components["asr_dense"].raw_scores)
    )
    recombined_bm25 = minmax_rows(
        bm25_weights.title_weight * bm25_components["bm25_title"].raw_scores
        + bm25_weights.caption_weight * bm25_components["bm25_caption"].raw_scores
        + bm25_weights.ocr_weight * bm25_components["bm25_ocr"].raw_scores
        + bm25_weights.asr_weight * bm25_components["bm25_asr"].raw_scores
    )
    expected = (
        config.dense_weight * recombined_dense
        + config.bm25_weight * recombined_bm25
    )

    actual = scorer.score_events(
        ["sự kiện"],
        ["event"],
        caption_events=["mô tả"],
        use_dense=True,
        use_bm25=True,
    )[0].scores

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_partial_dense_startup_and_legacy_rejection() -> None:
    visual = FakeIndex(np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32))
    context = FakeIndex(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    asr = FakeIndex(np.asarray([[0.5, 0.6, 0.7]], dtype=np.float32))
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))

    # Visual only
    dense_visual_only = DenseTemporalScorer(
        visual_index=visual,
        context_index=None,
        asr_index=None,
        visual_encoder=encoder,
        text_encoder=None,
        weights=DenseTemporalWeights(),
    )
    # Visual + Context
    dense_visual_context = DenseTemporalScorer(
        visual_index=visual,
        context_index=context,
        asr_index=None,
        visual_encoder=encoder,
        text_encoder=encoder,
        weights=DenseTemporalWeights(),
    )
    # Visual + ASR
    dense_visual_asr = DenseTemporalScorer(
        visual_index=visual,
        context_index=None,
        asr_index=asr,
        visual_encoder=encoder,
        text_encoder=encoder,
        weights=DenseTemporalWeights(),
    )

    # Incomplete dense rejected in legacy mode
    with pytest.raises(RuntimeError, match="legacy Dense temporal fusion requires Visual, Context, and ASR"):
        dense_visual_only.score_events(["event"])

    with pytest.raises(RuntimeError, match="legacy Dense temporal fusion requires Visual, Context, and ASR"):
        dense_visual_context.score_events(["event"])

    with pytest.raises(RuntimeError, match="legacy Dense temporal fusion requires Visual, Context, and ASR"):
        dense_visual_asr.score_events(["event"])

    # Adaptive mode accepts partial dense
    adaptive_scorer = TemporalEvidenceScorer(
        visual_index=visual,
        dense=dense_visual_only,
        bm25=None,
        config=HybridTemporalConfig(fusion_mode="adaptive_p0"),
        visual_dense_ready=True,
        context_dense_ready=False,
        asr_dense_ready=False,
    )
    scores = adaptive_scorer.score_events(["event"], ["event"], caption_events=["event"], use_dense=True, use_bm25=False)
    assert len(scores) == 1
    assert scores[0].scores.shape == (1, 3)
