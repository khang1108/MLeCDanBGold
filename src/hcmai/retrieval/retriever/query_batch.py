"""Ordered query-text and embedding batch contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from hcmai.common.observability import StageTrace
from hcmai.retrieval.embedding.pipeline import TextEmbeddingAdapter
from hcmai.common.observability import PipelineStage
from hcmai.common.observability.tracing import StageTimer
from hcmai.retrieval.retriever.cache import EmbeddingCache, EmbeddingCacheKey

SourceFamily = Literal["visual", "text"]


@dataclass(frozen=True, slots=True)
class QueryText:
    """One caller text with stable order and normalized lookup identity."""

    text: str
    normalized_text: str
    position: int
    source_family: SourceFamily


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    """One query vector with encoder and source-family provenance."""

    query: QueryText
    vector: np.ndarray
    model_name: str
    revision: str | None
    normalized: bool

    @property
    def dimension(self) -> int:
        return int(self.vector.shape[0])


@dataclass(frozen=True, slots=True)
class QueryEmbeddingBatch:
    """Ordered embeddings produced by one non-empty encoder call."""

    embeddings: tuple[QueryEmbedding, ...]
    encoding_trace: StageTrace

    def __post_init__(self) -> None:
        if not self.embeddings:
            raise ValueError("query embedding batch must not be empty")
        dimensions = {embedding.dimension for embedding in self.embeddings}
        models = {embedding.model_name for embedding in self.embeddings}
        families = {
            embedding.query.source_family for embedding in self.embeddings
        }
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("query embeddings must have one positive dimension")
        if len(models) != 1 or len(families) != 1:
            raise ValueError("query embeddings must share model and source family")

    @property
    def vectors(self) -> np.ndarray:
        return np.stack([embedding.vector for embedding in self.embeddings])

    @property
    def model_name(self) -> str:
        return self.embeddings[0].model_name

    @property
    def revision(self) -> str | None:
        return self.embeddings[0].revision

    @property
    def source_family(self) -> SourceFamily:
        return self.embeddings[0].query.source_family

    @property
    def dimension(self) -> int:
        return self.embeddings[0].dimension

    @property
    def normalized(self) -> bool:
        return all(embedding.normalized for embedding in self.embeddings)


def encode_query_batch(
    texts: list[str],
    encoder: TextEmbeddingAdapter,
    source_family: SourceFamily,
    cache: EmbeddingCache | None = None,
    prompt_version: str = "query-v1",
) -> QueryEmbeddingBatch:
    """Normalize/deduplicate text, encode once, and restore caller order."""

    queries = tuple(
        QueryText(
            text=text,
            normalized_text=_normalize_text(text),
            position=position,
            source_family=source_family,
        )
        for position, text in enumerate(texts)
    )
    if not queries:
        raise ValueError("texts must not be empty")
    unique_texts = list(dict.fromkeys(query.normalized_text for query in queries))
    timer = StageTimer(PipelineStage.ENCODE.value)
    model_name = encoder.config.model_name
    initial_revision = _encoder_revision(encoder)
    by_text: dict[str, np.ndarray] = {}
    misses: list[str] = []
    for normalized_text in unique_texts:
        key = EmbeddingCacheKey(
            model_name=model_name,
            revision=initial_revision,
            source_family=source_family,
            normalized_query=normalized_text,
            prompt_version=prompt_version,
        )
        cached = cache.get(key) if cache is not None else None
        if cached is None:
            misses.append(normalized_text)
        else:
            by_text[normalized_text] = cached
    if misses:
        encoded = _validated_vectors(encoder.encode_text(misses), len(misses))
        revision = _encoder_revision(encoder)
        if by_text and revision != initial_revision:
            by_text.clear()
            misses = unique_texts
            encoded = _validated_vectors(
                encoder.encode_text(misses),
                len(misses),
            )
            revision = _encoder_revision(encoder)
        for normalized_text, vector in zip(misses, encoded):
            readonly = _readonly(vector)
            by_text[normalized_text] = readonly
            if cache is not None:
                cache.set(
                    EmbeddingCacheKey(
                        model_name=model_name,
                        revision=revision,
                        source_family=source_family,
                        normalized_query=normalized_text,
                        prompt_version=prompt_version,
                    ),
                    readonly,
                )
    else:
        revision = initial_revision
    trace = timer.finish(
        cache_hit=not misses,
        input_count=len(unique_texts),
        output_count=len(unique_texts),
        backend=model_name,
    )
    embeddings = tuple(
        QueryEmbedding(
            query=query,
            vector=by_text[query.normalized_text],
            model_name=model_name,
            revision=revision,
            normalized=_is_normalized(by_text[query.normalized_text]),
        )
        for query in queries
    )
    return QueryEmbeddingBatch(embeddings=embeddings, encoding_trace=trace)


def _normalize_text(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("query text must not be empty")
    return normalized


def _encoder_revision(encoder: TextEmbeddingAdapter) -> str | None:
    value = getattr(encoder, "resolved_revision", None)
    if value is None:
        value = getattr(encoder, "revision", None)
    return str(value) if value is not None else None


def _is_normalized(vector: np.ndarray) -> bool:
    return bool(np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-3, atol=1e-4))


def _validated_vectors(values, expected_count: int) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != expected_count:
        raise ValueError("encoder returned an invalid query embedding batch shape")
    if not np.isfinite(vectors).all():
        raise ValueError("encoder returned non-finite query embeddings")
    return vectors


def _readonly(vector: np.ndarray) -> np.ndarray:
    value = np.array(vector, dtype=np.float32, copy=True)
    value.setflags(write=False)
    return value
