"""Retriever contract consumed by ``RetrievalService``."""

from __future__ import annotations

from typing import Protocol

from hcmai.retrieval.models import RetrievalResult, RetrievalSource
from hcmai.retrieval.retriever.query_batch import (
    QueryEmbeddingBatch,
    SourceFamily,
)


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> RetrievalResult: ...

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 100,
    ) -> list[RetrievalResult]: ...


class VectorRetriever(Retriever, Protocol):
    """Retriever accepted by the concurrent RRF vector-search path."""

    @property
    def source(self) -> RetrievalSource: ...

    @property
    def source_family(self) -> SourceFamily: ...

    def encode(self, query_texts: list[str]) -> QueryEmbeddingBatch: ...

    def search_vectors(
        self,
        query_batch: QueryEmbeddingBatch,
        top_k: int = 100,
    ) -> list[RetrievalResult]: ...
