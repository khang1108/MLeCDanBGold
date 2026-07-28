"""Smoke tests for the dense encoder without model downloads."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.retriever.encoder import DenseEncoder, EncodingStats


class FakeBatch(dict):
    def to(self, _device: str) -> FakeBatch:
        return self


class FakeProcessor:
    def __call__(self, *, return_tensors: str, **inputs) -> FakeBatch:
        assert return_tensors == "pt"
        self.inputs = inputs
        samples = next(iter(inputs.values()))
        return FakeBatch(sample_count=len(samples))


class FakeModel:
    config = SimpleNamespace(
        projection_dim=2, text_config=SimpleNamespace(max_position_embeddings=64)
    )

    def get_image_features(self, sample_count: int) -> torch.Tensor:
        return torch.tensor([[3.0, 4.0]] * sample_count)

    def get_text_features(self, sample_count: int) -> torch.Tensor:
        return torch.tensor([[0.0, 2.0]] * sample_count)


def _encoder() -> DenseEncoder:
    encoder = DenseEncoder(EncoderConfig(batch_size=2))
    encoder.processor = FakeProcessor()
    encoder.model = FakeModel()
    return encoder


def test_config_alias_and_defaults() -> None:
    config = EncoderConfig.from_dict(
        {"name": "custom/model", "device": "cuda", "batch_size": 64}
    )
    assert config.model_name == "custom/model"
    assert config.device == "cuda"
    assert config.batch_size == 64
    assert config.image_size == 224


def test_constructor_does_not_load_model() -> None:
    encoder = DenseEncoder(EncoderConfig())
    assert encoder.model is None
    assert encoder.processor is None
    assert encoder.embedding_dim == 0


def test_image_batches_are_normalized_and_recorded() -> None:
    images = [Image.new("RGB", (2, 2)) for _ in range(3)]
    stats = EncodingStats()
    vectors = _encoder().encode_images(images, stats)

    assert vectors.shape == (3, 2)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors, [[0.6, 0.8]] * 3)
    assert stats.num_encoded == 3
    assert stats.embedding_dim == 2
    assert len(stats.batch_times_ms) == 2


def test_text_batches_are_normalized() -> None:
    encoder = _encoder()
    vectors = encoder.encode_text(["one", "two", "three"])
    np.testing.assert_allclose(vectors, [[0.0, 1.0]] * 3)
    assert isinstance(encoder.processor, FakeProcessor)
    assert encoder.processor.inputs["padding"] == "max_length"
    assert encoder.processor.inputs["max_length"] == 64
    assert encoder.processor.inputs["truncation"] is True


def test_empty_input_does_not_load_model() -> None:
    encoder = DenseEncoder(EncoderConfig())
    vectors = encoder.encode_text([])
    assert vectors.shape == (0, 0)
    assert encoder.model is None


def test_stats_report() -> None:
    stats = EncodingStats(
        num_encoded=100, num_failed=5, total_time_ms=1000.0,
        embedding_dim=768, batch_times_ms=[10, 20, 15],
    )
    assert stats.throughput_samples_per_sec == 100.0
    assert "failed=5" in stats.report()
