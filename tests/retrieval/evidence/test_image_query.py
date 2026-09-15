"""Tests for ImageQueryTemporalScorer."""

import numpy as np
from PIL import Image
import pytest

from hcmai.kis.models import KISImageRef
from hcmai.retrieval.evidence.image_query import ImageQueryTemporalScorer


class FakeVisualIndex:
    def __init__(self) -> None:
        self.frame_ids = np.asarray(["f1", "f2", "f3"])
        self._vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)

    def score_subset(self, query_vectors, positions, chunk_size):
        del chunk_size
        return np.asarray(query_vectors, dtype=np.float32) @ self._vectors[positions].T


class FakeAssetStore:
    def open(self, asset_id: str):
        image = Image.new("RGB", (1, 1))
        image.info["asset_id"] = asset_id
        return image


class FakeImageEncoder:
    def encode_images(self, images, stats=None):
        del stats
        vectors = {
            "sha256:a": [1.0, 0.0],
            "sha256:b": [0.0, 1.0],
            "sha256:c": [0.6, 0.8],
        }
        return np.asarray([vectors[image.info["asset_id"]] for image in images], dtype=np.float32)


def test_image_query_scorer_max_pools_multiple_exemplars_per_event() -> None:
    visual_index = FakeVisualIndex()
    scorer = ImageQueryTemporalScorer(
        visual_index=visual_index,
        image_encoder=FakeImageEncoder(),
        asset_store=FakeAssetStore(),
        chunk_size=128,
    )
    component = scorer.score_events([
        (
            KISImageRef(asset_id="sha256:a", content_type="image/png"),
            KISImageRef(asset_id="sha256:b", content_type="image/png"),
        ),
        (),
    ])

    scores_a = np.asarray([[1.0, 0.0]], dtype=np.float32) @ visual_index._vectors.T
    scores_b = np.asarray([[0.0, 1.0]], dtype=np.float32) @ visual_index._vectors.T
    assert component.name == "visual_image"
    assert component.raw_scores.shape == (2, 3)
    np.testing.assert_allclose(component.raw_scores[0], np.maximum(scores_a[0], scores_b[0]))
    np.testing.assert_allclose(component.raw_scores[1], np.zeros(3, dtype=np.float32))


def test_image_query_scorer_empty_events() -> None:
    visual_index = FakeVisualIndex()
    scorer = ImageQueryTemporalScorer(
        visual_index=visual_index,
        image_encoder=FakeImageEncoder(),
        asset_store=FakeAssetStore(),
        chunk_size=128,
    )
    component = scorer.score_events([(), ()])
    assert component.name == "visual_image"
    assert component.raw_scores.shape == (2, 3)
    np.testing.assert_allclose(component.raw_scores, np.zeros((2, 3), dtype=np.float32))


def test_baseline_config_fusion_with_image_evidence() -> None:
    from hcmai.common.config import AppConfig
    from hcmai.retrieval.evidence.components import TemporalScoreBundle, TemporalScoreComponent
    from hcmai.retrieval.evidence.fusion import EventEvidenceProfile, TemporalFusionScorer

    config = AppConfig.from_yaml("configs/baseline.yaml")
    base_weights = config.search.hybrid_temporal.adaptive.base_component_weights
    assert base_weights.get("visual_image") == 0.35

    scorer = TemporalFusionScorer(config.search.hybrid_temporal.adaptive)
    raw = np.asarray([[0.1, 0.9, 0.4]], dtype=np.float32)
    bundle = TemporalScoreBundle({
        "visual_image": TemporalScoreComponent(name="visual_image", raw_scores=raw),
    })
    profiles = [
        EventEvidenceProfile(
            original_text=None,
            retrieval_text=None,
            has_text=False,
            has_images=True,
        )
    ]
    fused = scorer.fuse_profiles(profiles=profiles, bundle=bundle)
    assert fused.shape == (1, 3)
    assert fused[0, 1] > fused[0, 2] > fused[0, 0]
    assert np.all(fused >= 0.0)

