"""Tests for adaptive event-driven multimodal temporal evidence fusion."""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.common.config import AdaptiveTemporalFusionConfig
from hcmai.retrieval.evidence.components import (
    TemporalScoreBundle,
    TemporalScoreComponent,
)
from hcmai.retrieval.evidence.fusion import (
    EventModalityRouter,
    TemporalFusionScorer,
)


def test_speech_event_boosts_asr_components() -> None:
    """Speech cue in original or retrieval text increases ASR component weights."""

    router = EventModalityRouter(AdaptiveTemporalFusionConfig())
    weights = router.multipliers(
        "Cô gái nói chuyện với người đối diện về món ăn.",
        "The woman talks with a person seated opposite her about the dish.",
    )
    assert weights["asr_dense"] > weights["visual_dense"]
    assert weights["bm25_asr"] > weights["bm25_caption"]


def test_visible_text_event_boosts_ocr() -> None:
    """Visible text or screen cue increases OCR component weights."""

    router = EventModalityRouter(AdaptiveTemporalFusionConfig())
    weights = router.multipliers(
        'Màn hình hiển thị dòng chữ "TP.HCM".',
        'The screen displays the text "TP.HCM".',
    )
    assert weights["bm25_ocr"] > weights["bm25_asr"]


def test_visual_action_event_boosts_visual_components() -> None:
    """Visual actions (e.g. wearing, holding) boost visual and context weights."""

    router = EventModalityRouter(AdaptiveTemporalFusionConfig())
    weights = router.multipliers(
        "Người đàn ông mặc áo đỏ đang chạy xe.",
        "The man wearing a red shirt is riding a bike.",
    )
    assert weights["visual_dense"] > AdaptiveTemporalFusionConfig().base_component_weights["visual_dense"]
    assert weights["context_dense"] > AdaptiveTemporalFusionConfig().base_component_weights["context_dense"]


def test_fusion_renormalizes_when_asr_has_no_frame_coverage() -> None:
    """When ASR has no coverage on a frame, remaining components renormalize to 1.0."""

    bundle = TemporalScoreBundle(
        {
            "visual_dense": TemporalScoreComponent(
                "visual_dense", np.asarray([[0.2, 0.4, 0.6]], dtype=np.float32)
            ),
            "asr_dense": TemporalScoreComponent(
                "asr_dense",
                np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
                coverage=np.asarray([False, True, False]),
            ),
        }
    )
    config = AdaptiveTemporalFusionConfig(
        confidence_gating=False,
        event_routing=False,
        base_component_weights={"visual_dense": 0.5, "asr_dense": 0.5},
    )
    scorer = TemporalFusionScorer(config)

    actual = scorer.fuse(
        original_events=["người phụ nữ nói"],
        retrieval_events=["the woman speaks"],
        bundle=bundle,
    )
    visual_only = scorer.fuse(
        original_events=["người phụ nữ nói"],
        retrieval_events=["the woman speaks"],
        bundle=TemporalScoreBundle({"visual_dense": bundle.components["visual_dense"]}),
    )

    assert actual.shape == (1, 3)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[0, [0, 2]], visual_only[0, [0, 2]], atol=1e-6)


def test_constant_noise_component_contributes_zero_weight_under_confidence_gating() -> None:
    """A constant-score component has zero reliability and contributes nothing."""

    bundle = TemporalScoreBundle(
        {
            "visual_dense": TemporalScoreComponent(
                "visual_dense", np.asarray([[0.1, 0.5, 0.9]], dtype=np.float32)
            ),
            "bm25_ocr": TemporalScoreComponent(
                "bm25_ocr", np.asarray([[0.3, 0.3, 0.3]], dtype=np.float32)
            ),
        }
    )
    config = AdaptiveTemporalFusionConfig(
        confidence_gating=True,
        event_routing=False,
        base_component_weights={"visual_dense": 0.5, "bm25_ocr": 0.5},
    )
    scorer = TemporalFusionScorer(config)

    fused = scorer.fuse(
        original_events=["biển hiệu cửa hàng"],
        retrieval_events=["store sign"],
        bundle=bundle,
    )
    visual_only = scorer.fuse(
        original_events=["biển hiệu cửa hàng"],
        retrieval_events=["store sign"],
        bundle=TemporalScoreBundle({"visual_dense": bundle.components["visual_dense"]}),
    )

    np.testing.assert_allclose(fused, visual_only, atol=1e-6)


def test_event_count_mismatch_raises_value_error() -> None:
    """Mismatched event counts raise ValueError."""

    scorer = TemporalFusionScorer(AdaptiveTemporalFusionConfig())
    bundle = TemporalScoreBundle(
        {
            "visual_dense": TemporalScoreComponent(
                "visual_dense", np.asarray([[0.1, 0.2]], dtype=np.float32)
            )
        }
    )

    with pytest.raises(ValueError, match="counts must match"):
        scorer.fuse(
            original_events=["e1", "e2"],
            retrieval_events=["e1"],
            bundle=bundle,
        )

    with pytest.raises(ValueError, match="event count must match"):
        scorer.fuse(
            original_events=["e1", "e2"],
            retrieval_events=["e1", "e2"],
            bundle=bundle,
        )


def test_stateless_scorer_and_cloned_scorers_reflect_config_changes() -> None:
    """Cloned scorers and stateless config changes produce different fused matrices."""
    from hcmai.common.config import HybridTemporalConfig
    from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
    from tests.retrieval.evidence.fakes import FakeEncoder, FakeIndex

    visual = FakeIndex(np.asarray([[0.2, 0.4, 0.8]], dtype=np.float32))
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))

    class MockDense:
        def score_components(self, events, *, asr_interval_projection=True):
            del events, asr_interval_projection
            return TemporalScoreBundle(
                {
                    "visual_dense": TemporalScoreComponent(
                        "visual_dense", np.asarray([[0.1, 0.5, 0.9]], dtype=np.float32)
                    ),
                    "asr_dense": TemporalScoreComponent(
                        "asr_dense",
                        np.asarray([[0.9, 0.5, 0.1]], dtype=np.float32),
                        coverage=np.asarray([True, True, True]),
                    ),
                }
            )

    base_config = HybridTemporalConfig(
        fusion_mode="adaptive_p0",
        adaptive=AdaptiveTemporalFusionConfig(event_routing=False, confidence_gating=False),
    )
    scorer1 = TemporalEvidenceScorer(
        visual_index=visual,
        dense=MockDense(),
        bm25=None,
        config=base_config,
    )

    # Scorer 2 enables event routing where speech cue heavily boosts asr_dense
    routing_config = base_config.model_copy(
        update={
            "adaptive": base_config.adaptive.model_copy(update={"event_routing": True})
        }
    )
    scorer2 = scorer1.with_config(routing_config)

    res1 = scorer1.score_events(
        ["Người phụ nữ nói chuyện"],
        ["The woman speaks"],
        caption_events=None,
        use_dense=True,
        use_bm25=False,
    )
    res2 = scorer2.score_events(
        ["Người phụ nữ nói chuyện"],
        ["The woman speaks"],
        caption_events=None,
        use_dense=True,
        use_bm25=False,
    )

    # Scorer 1 (equal/base weights) and Scorer 2 (speech-boosted ASR) must differ!
    assert not np.allclose(res1[0].scores, res2[0].scores)
    # Under speech boost, frame 0 (high ASR score 0.9) must have higher score in res2 than in res1
    assert res2[0].scores[0, 0] > res1[0].scores[0, 0]
