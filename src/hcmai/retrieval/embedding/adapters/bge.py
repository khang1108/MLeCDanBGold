"""Normalized BGE-M3 dense text embeddings for CaptionStore retrieval."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np

from sentence_transformers import SentenceTransformer
from hcmai.common.config import EncoderConfig
from hcmai.retrieval.embedding.models.stats import EncodingStats


class BGEAdapter:
    """Lazy SentenceTransformers adapter for BGE-M3 dense vectors."""

    def __init__(
        self,
        config: EncoderConfig,
        *,
        model: Any | None = None,
        loader: Callable[..., Any] | None = None,
    ) -> None:
        if config.backend != "bge_m3":
            raise ValueError("BGEAdapter requires backend='bge_m3'")
        self.config = config
        self.model = model
        self._loader = loader
        self.embedding_dim = _dimension(model)

    def _load_model(self) -> None:
        if self.model is not None:
            return

        loader = self._loader
        if loader is None:
            loader = SentenceTransformer

        options: dict[str, Any] = {"device": self.config.device}
        if self.config.revision is not None:
            options["revision"] = self.config.revision

        self.model = loader(self.config.model_name, **options)
        self.model.max_seq_length = self.config.max_length
        self.embedding_dim = _dimension(self.model)

    def encode_text(
        self,
        texts: list[str],
        stats: EncodingStats | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=self.config.dtype)

        self._load_model()
        assert self.model is not None

        started = perf_counter()
        vectors = np.asarray(
            self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=self.config.dtype,
        )

        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise ValueError("BGE-M3 returned an invalid embedding shape")
        if not np.isfinite(vectors).all():
            raise ValueError("BGE-M3 returned non-finite embeddings")
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        
        if np.any(norms <= 0):
            raise ValueError("BGE-M3 returned a zero embedding")
        vectors = vectors / norms
        
        self.embedding_dim = int(vectors.shape[1])
        if stats is not None:
            elapsed = (perf_counter() - started) * 1_000
            stats.num_encoded += len(texts)
            stats.total_time_ms += elapsed
            stats.batch_times_ms.append(elapsed)
            stats.embedding_dim = self.embedding_dim
        return vectors.astype(self.config.dtype, copy=False)


def _dimension(model: Any | None) -> int:
    if model is None:
        return 0
    value = model.get_sentence_embedding_dimension()
    return int(value or 0)
