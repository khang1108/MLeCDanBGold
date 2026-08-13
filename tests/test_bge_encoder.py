"""Smoke tests for normalized BGE-M3 caption embeddings."""

import numpy as np

from hcmai.common.config import EncoderConfig
from hcmai.retrieval.embedding.adapters.bge import BGEAdapter
from hcmai.retrieval.embedding.models.stats import EncodingStats


class FakeSentenceTransformer:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["batch_size"] == 2
        return np.asarray([[3.0, 4.0, 0.0]] * len(texts), dtype=np.float32)


def test_bge_encoder_returns_finite_unit_vectors_and_stats():
    config = EncoderConfig(
        backend="bge_m3",
        model_name="BAAI/bge-m3",
        batch_size=2,
    )
    stats = EncodingStats()
    encoder = BGEAdapter(config, model=FakeSentenceTransformer())

    vectors = encoder.encode_text(["xin chào", "hello"], stats)

    assert vectors.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert stats.num_encoded == 2
    assert stats.embedding_dim == 3
