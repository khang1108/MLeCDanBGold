"""Retriever contract consumed by ``RetrievalService``."""

from __future__ import annotations

from typing import Protocol

from hcmai.common.schemas import RetrievalCandidate, TaskType
from hcmai.common.schemas.search import SearchFilters


class Retriever(Protocol):
    last_query_encoding_ms: float
    last_index_search_ms: float

    def search(
        self,
        query: str,
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalCandidate]: ...
