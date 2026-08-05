"""Remote text-embedding adapter."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

import numpy as np

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import TextEmbeddingResponse
from hcmai.embedding.models.stats import EncodingStats


class EmbeddingClient(Protocol):
    def embed_text(
        self, texts: list[str], source: str = "visual"
    ) -> TextEmbeddingResponse: ...


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
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=self.config.dtype)
        started, batches = perf_counter(), []
        size = min(self.config.batch_size, 64)
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
        if self.embedding_dim == 0:
            self.embedding_dim = response.dimension
        if response.dimension != self.embedding_dim or not response.normalized:
            raise ValueError("remote embedding metadata mismatch")
        vectors = np.asarray(response.embeddings, dtype=self.config.dtype)
        if vectors.shape != (count, self.embedding_dim):
            raise ValueError("remote embedding shape mismatch")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("remote embedding contains non-finite values")
        return vectors
