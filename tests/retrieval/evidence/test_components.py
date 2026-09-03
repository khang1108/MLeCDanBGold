"""Tests for first-class temporal evidence components and bundles."""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.common.config import DenseTemporalWeights
from hcmai.retrieval.evidence.components import TemporalScoreBundle, TemporalScoreComponent
from hcmai.retrieval.evidence.dense import DenseTemporalScorer
from tests.retrieval.evidence.fakes import FakeEncoder, FakeIndex


def test_component_requires_finite_two_dimensional_scores() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        TemporalScoreComponent("visual_dense", np.asarray([1.0], dtype=np.float32))

    with pytest.raises(ValueError, match="finite"):
        TemporalScoreComponent(
            "visual_dense",
            np.asarray([[0.0, np.inf]], dtype=np.float32),
        )


def test_bundle_requires_same_event_and_frame_shape() -> None:
    with pytest.raises(ValueError, match="same score shape"):
        TemporalScoreBundle(
            {
                "visual_dense": TemporalScoreComponent(
                    "visual_dense", np.zeros((2, 3), dtype=np.float32)
                ),
                "context_dense": TemporalScoreComponent(
                    "context_dense", np.zeros((2, 4), dtype=np.float32)
                ),
            }
        )


def test_dense_score_components_preserves_raw_modality_scores() -> None:
    visual = FakeIndex(np.asarray([[0.20, 0.21, 0.22]], dtype=np.float32))
    context = FakeIndex(np.asarray([[2.0, 4.0, 8.0]], dtype=np.float32))
    asr = FakeIndex(np.asarray([[-0.1, 0.0, 0.1]], dtype=np.float32))
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    scorer = DenseTemporalScorer(
        visual_index=visual,
        context_index=context,
        asr_index=asr,
        visual_encoder=encoder,
        text_encoder=encoder,
        weights=DenseTemporalWeights(),
    )

    bundle = scorer.score_components(["event"])

    assert set(bundle.components) == {"visual_dense", "context_dense", "asr_dense"}
    np.testing.assert_array_equal(
        bundle.components["visual_dense"].raw_scores,
        visual.scores,
    )
    np.testing.assert_array_equal(
        bundle.components["context_dense"].raw_scores,
        context.scores,
    )
    np.testing.assert_array_equal(
        bundle.components["asr_dense"].raw_scores,
        asr.scores,
    )


def test_bm25_score_components_keeps_fields_separate(tmp_path) -> None:
    from hcmai.common.config import BM25FieldWeights
    from hcmai.retrieval.evidence.bm25 import BM25TemporalScorer
    from tests.retrieval.evidence.test_bm25 import _artifact

    artifact, canonical = _artifact(tmp_path)
    scorer = BM25TemporalScorer.load(artifact, canonical, BM25FieldWeights())
    bundle = scorer.score_components(["HTV tạp dề trắng"], ["tạp dề trắng"])

    assert set(bundle.components) == {
        "bm25_title",
        "bm25_caption",
        "bm25_ocr",
        "bm25_asr",
    }
    assert bundle.shape == (1, 3)
    # caption matches f1
    assert bundle.components["bm25_caption"].raw_scores[0, 0] > 0
    # ocr matches f2 for HTV
    assert bundle.components["bm25_ocr"].raw_scores[0, 1] > 0


def test_dense_scorer_supports_visual_only() -> None:
    visual = FakeIndex(np.asarray([[0.1, 0.2]], dtype=np.float32))
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    scorer = DenseTemporalScorer(
        visual_index=visual,
        context_index=None,
        asr_index=None,
        visual_encoder=encoder,
        text_encoder=None,
        weights=DenseTemporalWeights(),
    )
    bundle = scorer.score_components(["event"])
    assert set(bundle.components) == {"visual_dense"}
    assert len(encoder.calls) == 1


def test_dense_scorer_supports_visual_and_context() -> None:
    visual = FakeIndex(np.asarray([[0.1, 0.2]], dtype=np.float32))
    context = FakeIndex(np.asarray([[0.3, 0.4]], dtype=np.float32))
    v_encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    t_encoder = FakeEncoder(np.asarray([[0.0, 1.0]], dtype=np.float32))
    scorer = DenseTemporalScorer(
        visual_index=visual,
        context_index=context,
        asr_index=None,
        visual_encoder=v_encoder,
        text_encoder=t_encoder,
        weights=DenseTemporalWeights(),
    )
    bundle = scorer.score_components(["event"])
    assert set(bundle.components) == {"visual_dense", "context_dense"}
    assert len(v_encoder.calls) == 1
    assert len(t_encoder.calls) == 1


def test_dense_scorer_supports_visual_and_asr() -> None:
    visual = FakeIndex(np.asarray([[0.1, 0.2]], dtype=np.float32))
    asr = FakeIndex(np.asarray([[0.5, 0.6]], dtype=np.float32))
    v_encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    t_encoder = FakeEncoder(np.asarray([[0.0, 1.0]], dtype=np.float32))
    scorer = DenseTemporalScorer(
        visual_index=visual,
        context_index=None,
        asr_index=asr,
        visual_encoder=v_encoder,
        text_encoder=t_encoder,
        weights=DenseTemporalWeights(),
    )
    bundle = scorer.score_components(["event"])
    assert set(bundle.components) == {"visual_dense", "asr_dense"}
    assert len(v_encoder.calls) == 1
    assert len(t_encoder.calls) == 1


def test_dense_scorer_text_encoder_called_once_for_both_context_and_asr() -> None:
    visual = FakeIndex(np.asarray([[0.1, 0.2]], dtype=np.float32))
    context = FakeIndex(np.asarray([[0.3, 0.4]], dtype=np.float32))
    asr = FakeIndex(np.asarray([[0.5, 0.6]], dtype=np.float32))
    v_encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    t_encoder = FakeEncoder(np.asarray([[0.0, 1.0]], dtype=np.float32))
    scorer = DenseTemporalScorer(
        visual_index=visual,
        context_index=context,
        asr_index=asr,
        visual_encoder=v_encoder,
        text_encoder=t_encoder,
        weights=DenseTemporalWeights(),
    )
    bundle = scorer.score_components(["event"])
    assert set(bundle.components) == {"visual_dense", "context_dense", "asr_dense"}
    assert len(v_encoder.calls) == 1
    assert len(t_encoder.calls) == 1


