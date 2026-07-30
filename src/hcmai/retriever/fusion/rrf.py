"""Reciprocal-rank fusion over candidates from independent retrievers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hcmai.common.config import FusionConfig
from hcmai.common.schemas import TaskType
from hcmai.common.schemas.retrieval import RetrievalCandidate
from hcmai.common.schemas.search import SearchFilters


class RRFFusionRetriever:
    """Union candidates by frame ID and rank them with reciprocal rank fusion."""

    def __init__(
        self,
        retrievers: Sequence[Any],
        config: FusionConfig,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("RRF fusion requires at least two retrievers")
        if config.method != "rrf":
            raise ValueError(f"Unsupported fusion method {config.method!r}")
        self.retrievers = tuple(retrievers)
        self.config = config
        self.last_query_encoding_ms = 0.0
        self.last_index_search_ms = 0.0

    def search(
        self,
        query: str,
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalCandidate]:
        """Retrieve, merge exact frame IDs, and apply task-specific weights."""

        pool: dict[str, RetrievalCandidate] = {}
        for retriever in self.retrievers:
            for candidate in retriever.search(query, top_k, filters):
                existing = pool.get(candidate.frame_id)
                pool[candidate.frame_id] = (
                    candidate.model_copy(deep=True)
                    if existing is None
                    else _merge(existing, candidate)
                )

        self.last_query_encoding_ms = sum(
            float(getattr(item, "last_query_encoding_ms", 0.0))
            for item in self.retrievers
        )
        self.last_index_search_ms = sum(
            float(getattr(item, "last_index_search_ms", 0.0))
            for item in self.retrievers
        )

        weights = self.config.task_weights[query_type]
        fused = [
            candidate.model_copy(
                update={
                    "fusion_score": sum(
                        weights[source] / (self.config.rrf_k + rank)
                        for source, rank in candidate.source_ranks.items()
                    )
                }
            )
            for candidate in pool.values()
        ]
        fused.sort(key=_sort_key)
        return fused[:top_k]


def _merge(
    existing: RetrievalCandidate,
    incoming: RetrievalCandidate,
) -> RetrievalCandidate:
    """Combine source evidence without rewriting canonical candidate identity."""

    overlapping = set(existing.source_ranks).intersection(incoming.source_ranks)
    if overlapping:
        names = ", ".join(sorted(source.value for source in overlapping))
        raise ValueError(
            f"Duplicate retrieval sources for frame {existing.frame_id!r}: {names}"
        )
    return existing.model_copy(
        update={
            "source_scores": {
                **existing.source_scores,
                **incoming.source_scores,
            },
            "source_ranks": {
                **existing.source_ranks,
                **incoming.source_ranks,
            },
            "metadata": existing.metadata or incoming.metadata,
        },
        deep=True,
    )


def _sort_key(candidate: RetrievalCandidate) -> tuple[float, int, str]:
    """Sort by fused score, strongest source rank, then stable frame identity."""

    best_rank = min(candidate.source_ranks.values(), default=2**31)
    return (-(candidate.fusion_score or 0.0), best_rank, candidate.frame_id)
