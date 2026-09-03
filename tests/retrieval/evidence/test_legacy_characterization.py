"""Characterization tests for v9 temporal evidence scoring behavior."""

from __future__ import annotations

import numpy as np

from hcmai.common.config import DenseTemporalWeights, HybridTemporalConfig
from hcmai.retrieval.evidence.dense import DenseTemporalScorer
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
from hcmai.retrieval.evidence.normalization import minmax_rows
from tests.retrieval.evidence.fakes import FakeEncoder, FakeIndex


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


def test_dense_legacy_normalizes_each_modality_then_averages() -> None:
    visual = FakeIndex(np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32))
    context = FakeIndex(np.asarray([[10.0, 10.0, 12.0]], dtype=np.float32))
    asr = FakeIndex(np.asarray([[0.70, 0.71, 0.72]], dtype=np.float32))
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    scorer = DenseTemporalScorer(
        visual_index=visual,
        context_index=context,
        asr_index=asr,
        visual_encoder=encoder,
        text_encoder=encoder,
        weights=DenseTemporalWeights(),
        chunk_size=8,
    )

    actual = scorer.score_events(["event"])

    expected = np.asarray([[0.0, 1.0 / 3.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class StaticDense:
    def score_events(self, events):
        del events
        return np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)


class StaticBM25:
    def score_events(self, original, caption):
        del original, caption
        return np.asarray([[2.0, 3.0, 6.0]], dtype=np.float32)


def test_hybrid_legacy_minmaxes_bm25_then_uses_half_half_weights() -> None:
    visual = FakeIndex(np.zeros((1, 3), dtype=np.float32))
    scorer = TemporalEvidenceScorer(
        visual_index=visual,
        dense=StaticDense(),
        bm25=StaticBM25(),
        config=HybridTemporalConfig(),
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
