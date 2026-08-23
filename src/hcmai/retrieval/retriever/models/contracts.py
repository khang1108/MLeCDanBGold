"""Retriever contract consumed by ``RetrievalService``."""

from __future__ import annotations

from typing import Protocol

from hcmai.common.schemas import RetrievalResult, RetrievalSource, TaskType
from hcmai.common.schemas.search import SearchFilters
from hcmai.retrieval.retriever.query_batch import (
    QueryEmbeddingBatch,
    SourceFamily,
)


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> RetrievalResult: ...

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalResult]: ...


class VectorRetriever(Retriever, Protocol):
    """Retriever accepted by the concurrent RRF vector-search path."""

    source: RetrievalSource
    source_family: SourceFamily

    def encode(self, query_texts: list[str]) -> QueryEmbeddingBatch: ...

    def search_vectors(
        self,
        query_batch: QueryEmbeddingBatch,
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalResult]: ...
