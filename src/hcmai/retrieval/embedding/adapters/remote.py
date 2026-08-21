"""Remote text-embedding adapter."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import EmbeddingResponse, TextEmbeddingResponse
from hcmai.retrieval.embedding.models.stats import EncodingStats


class EmbeddingClient(Protocol):
    def embed_text(
        self, texts: list[str], source: str = "visual"
    ) -> TextEmbeddingResponse: ...


class ImageEmbeddingClient(Protocol):
    def embed_images(
        self,
        images: Sequence[Image.Image],
        *,
        source: str = "visual",
        item_ids: list[str] | None = None,
    ) -> EmbeddingResponse: ...


class RemoteEmbeddingAdapter:
    """Use hosted text embeddings and validate their identity and shape."""

    def __init__(
        self,
        client: EmbeddingClient,
        config: EncoderConfig,
        embedding_dim: int,
        source: str = "visual",
    ) -> None:
        self.client = client
        self.config = config
        self.embedding_dim = embedding_dim
        self.source = source

    def encode_text(
        self, texts: list[str], stats: EncodingStats | None = None
    ) -> np.ndarray:
        """Send text in batches bounded by this encoder's configured ceiling."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=self.config.dtype)
        started, batches = perf_counter(), []
        size = self.config.batch_size
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            response = self.client.embed_text(batch, self.source)
            batches.append(self._validate(response, len(batch)))
        vectors = np.vstack(batches)
        if stats is not None:
            elapsed = (perf_counter() - started) * 1_000
            stats.num_encoded += len(texts)
            stats.total_time_ms += elapsed
            stats.batch_times_ms.append(elapsed)
            stats.embedding_dim = self.embedding_dim
        return vectors

    def _validate(
        self, response: TextEmbeddingResponse, count: int
    ) -> np.ndarray:
        if response.model != self.config.model_name:
            raise ValueError("remote embedding checkpoint mismatch")
        if self.config.revision is not None and response.revision != self.config.revision:
            raise ValueError("remote embedding revision mismatch")
        if self.embedding_dim == 0:
            self.embedding_dim = response.dimension
        if response.dimension != self.embedding_dim or not response.normalized:
            raise ValueError("remote embedding metadata mismatch")
        vectors = np.asarray(response.embeddings, dtype=self.config.dtype)
        if vectors.shape != (count, self.embedding_dim):
            raise ValueError("remote embedding shape mismatch")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("remote embedding contains non-finite values")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
            raise ValueError("remote embedding vectors are not L2-normalized")
        return vectors


class RemoteImageEmbeddingAdapter:
    """Use hosted image embeddings without changing ordered frame identity."""

    def __init__(
        self,
        client: ImageEmbeddingClient,
        config: EncoderConfig,
        embedding_dim: int = 0,
    ) -> None:
        self.client = client
        self.config = config
        self.embedding_dim = embedding_dim

    def encode_images(
        self,
        images: list[Image.Image],
        stats: EncodingStats | None = None,
    ) -> np.ndarray:
        """Send images in batches bounded by the visual encoder configuration."""
        if not images:
            return np.empty((0, self.embedding_dim), dtype=self.config.dtype)
        started, batches = perf_counter(), []
        size = self.config.batch_size
        for start in range(0, len(images), size):
            batch = images[start : start + size]
            identifiers = [str(index) for index in range(start, start + len(batch))]
            response = self.client.embed_images(
                batch, source="visual", item_ids=identifiers
            )
            batches.append(self._validate(response, identifiers))
        vectors = np.vstack(batches)
        if stats is not None:
            elapsed = (perf_counter() - started) * 1_000
            stats.num_encoded += len(images)
            stats.total_time_ms += elapsed
            stats.batch_times_ms.append(elapsed)
            stats.embedding_dim = self.embedding_dim
        return vectors

    def _validate(
        self, response: EmbeddingResponse, identifiers: list[str]
    ) -> np.ndarray:
        if response.model != self.config.model_name:
            raise ValueError("remote image embedding checkpoint mismatch")
        if self.config.revision is not None and response.revision != self.config.revision:
            raise ValueError("remote image embedding revision mismatch")
        if response.item_ids != identifiers or not response.normalized:
            raise ValueError("remote image embedding metadata mismatch")
        if self.embedding_dim == 0:
            self.embedding_dim = response.dimension
        vectors = np.asarray(response.embeddings, dtype=self.config.dtype)
        if vectors.shape != (len(identifiers), self.embedding_dim):
            raise ValueError("remote image embedding shape mismatch")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("remote image embedding contains non-finite values")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
            raise ValueError("remote image embedding vectors are not L2-normalized")
        return vectors
