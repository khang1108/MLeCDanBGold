"""Tests for RemoteEmbeddingAdapter consuming TextEmbeddingBatch."""

import unittest
from unittest.mock import Mock

import numpy as np

from hcmai.common.config import EncoderConfig
from hcmai.inference.clients.embeddings import TextEmbeddingBatch
from hcmai.retrieval.embedding.adapters.remote import RemoteEmbeddingAdapter
from hcmai.retrieval.embedding.models.stats import EncodingStats


class RemoteEmbeddingAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EncoderConfig(
            model_name="BAAI/bge-m3",
            batch_size=2,
            dtype="float32",
        )

    def test_encode_text_success(self) -> None:
        client = Mock()
        client.embed_text.return_value = TextEmbeddingBatch(
            model="BAAI/bge-m3",
            vectors=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
        )
        adapter = RemoteEmbeddingAdapter(client, self.config, embedding_dim=3)
        texts = ["hello", "world"]

        vectors = adapter.encode_text(texts)

        client.embed_text.assert_called_once_with(["hello", "world"])
        self.assertIsInstance(vectors, np.ndarray)
        self.assertEqual(vectors.shape, (2, 3))
        np.testing.assert_allclose(
            vectors,
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )

    def test_encode_text_empty_input_returns_empty_array(self) -> None:
        client = Mock()
        adapter = RemoteEmbeddingAdapter(client, self.config, embedding_dim=3)

        vectors = adapter.encode_text([])

        self.assertEqual(vectors.shape, (0, 3))
        client.embed_text.assert_not_called()

    def test_encode_text_infers_dimension_when_zero(self) -> None:
        client = Mock()
        client.embed_text.return_value = TextEmbeddingBatch(
            model="BAAI/bge-m3",
            vectors=((0.6, 0.8),),
        )
        adapter = RemoteEmbeddingAdapter(client, self.config, embedding_dim=0)

        vectors = adapter.encode_text(["query"])

        self.assertEqual(adapter.embedding_dim, 2)
        self.assertEqual(vectors.shape, (1, 2))

    def test_encode_text_rejects_model_mismatch(self) -> None:
        client = Mock()
        client.embed_text.return_value = TextEmbeddingBatch(
            model="wrong-model",
            vectors=((1.0, 0.0),),
        )
        adapter = RemoteEmbeddingAdapter(client, self.config, embedding_dim=2)

        with self.assertRaisesRegex(ValueError, "remote embedding checkpoint mismatch"):
            adapter.encode_text(["query"])

    def test_encode_text_rejects_dimension_mismatch(self) -> None:
        client = Mock()
        client.embed_text.return_value = TextEmbeddingBatch(
            model="BAAI/bge-m3",
            vectors=((1.0, 0.0, 0.0),),
        )
        adapter = RemoteEmbeddingAdapter(client, self.config, embedding_dim=2)

        with self.assertRaisesRegex(ValueError, "remote embedding shape mismatch"):
            adapter.encode_text(["query"])

    def test_encode_text_rejects_count_mismatch(self) -> None:
        client = Mock()
        # Sent 2 texts, returned only 1 vector
        client.embed_text.return_value = TextEmbeddingBatch(
            model="BAAI/bge-m3",
            vectors=((1.0, 0.0),),
        )
        adapter = RemoteEmbeddingAdapter(client, self.config, embedding_dim=2)

        with self.assertRaisesRegex(ValueError, "remote embedding shape mismatch"):
            adapter.encode_text(["text 1", "text 2"])

    def test_encode_text_rejects_non_finite_values(self) -> None:
        client = Mock()
        client.embed_text.return_value = TextEmbeddingBatch(
            model="BAAI/bge-m3",
            vectors=((float("nan"), 1.0),),
        )
        adapter = RemoteEmbeddingAdapter(client, self.config, embedding_dim=2)

        with self.assertRaisesRegex(ValueError, "remote embedding contains non-finite values"):
            adapter.encode_text(["query"])

    def test_encode_text_rejects_non_normalized_vectors(self) -> None:
        client = Mock()
        # Norm is sqrt(2^2 + 2^2) = sqrt(8) != 1.0
        client.embed_text.return_value = TextEmbeddingBatch(
            model="BAAI/bge-m3",
            vectors=((2.0, 2.0),),
        )
        adapter = RemoteEmbeddingAdapter(client, self.config, embedding_dim=2)

        with self.assertRaisesRegex(ValueError, "remote embedding vectors are not L2-normalized"):
            adapter.encode_text(["query"])

    def test_encode_text_batches_large_input(self) -> None:
        client = Mock()
        # Batch size is 2, input is 3 texts -> 2 calls
        client.embed_text.side_effect = [
            TextEmbeddingBatch(
                model="BAAI/bge-m3",
                vectors=((1.0, 0.0), (0.0, 1.0)),
            ),
            TextEmbeddingBatch(
                model="BAAI/bge-m3",
                vectors=((0.6, 0.8),),
            ),
        ]
        adapter = RemoteEmbeddingAdapter(client, self.config, embedding_dim=2)

        vectors = adapter.encode_text(["a", "b", "c"])

        self.assertEqual(client.embed_text.call_count, 2)
        self.assertEqual(vectors.shape, (3, 2))

    def test_encode_text_records_stats(self) -> None:
        client = Mock()
        client.embed_text.return_value = TextEmbeddingBatch(
            model="BAAI/bge-m3",
            vectors=((1.0, 0.0),),
        )
        adapter = RemoteEmbeddingAdapter(client, self.config, embedding_dim=2)
        stats = EncodingStats()

        adapter.encode_text(["query"], stats=stats)

        self.assertEqual(stats.num_encoded, 1)
        self.assertGreater(stats.total_time_ms, 0)
        self.assertEqual(stats.embedding_dim, 2)


if __name__ == "__main__":
    unittest.main()
