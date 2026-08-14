"""Narrow structural contracts that keep VQA components fake-friendly."""

from __future__ import annotations

from typing import Protocol

from hcmai.common.schemas import FrameRecord, RetrievalResult, RetrievalSource, TaskType
from hcmai.common.schemas.search import SearchFilters


class TemporalData(Protocol):
    def neighbors(
        self, frame_id: str, *, window_ms: int, include_self: bool = False
    ) -> list[FrameRecord]: ...


class EvidenceLookup(Protocol):
    def get_evidence(self, frame_id: str, source: RetrievalSource) -> str | None: ...


class AnswerData(TemporalData, EvidenceLookup, Protocol):
    pass


class RetrievalGateway(Protocol):
    def search_batch(
        self,
        queries: list[str],
        top_k: int,
        filters: SearchFilters | None,
        query_type: TaskType,
    ) -> list[RetrievalResult]: ...
