"""Reciprocal-rank fusion over candidates from independent retrievers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hcmai.common.config import FusionConfig
from hcmai.common.schemas import (
    RetrievalResult,
    RetrievalSource,
    RetrievalTrace,
    TaskType,
)
from hcmai.common.schemas.retrieval import RetrievalCandidate
from hcmai.common.schemas.search import SearchFilters
from hcmai.observability.tracing import StageTimer
from hcmai.retriever.concurrent import (
    ModalitySearchExecutor,
    ModalitySearchJob,
)


class RRFFusionRetriever:
    """Union candidates by frame ID and rank them with reciprocal rank fusion."""

    def __init__(
        self,
        retrievers: Sequence[Any],
        config: FusionConfig,
        executor: ModalitySearchExecutor | None = None,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("RRF fusion requires at least two retrievers")
        if config.method != "rrf":
            raise ValueError(f"Unsupported fusion method {config.method!r}")
        self.retrievers = tuple(retrievers)
        self.config = config
        configured_sources = {
            source
            for retriever in self.retrievers
            if (source := getattr(retriever, "source", None)) is not None
        }
        missing_required = config.required_sources - configured_sources
        if configured_sources and missing_required:
            names = ", ".join(
                sorted(source.value for source in missing_required)
            )
            raise ValueError(f"Required retrieval sources are not configured: {names}")
        self._executor = executor or ModalitySearchExecutor(
            config.modality_max_workers
        )

    def search(
        self,
        query: str,
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> RetrievalResult:
        """Retrieve, merge exact frame IDs, and apply task-specific weights."""

        return self.search_batch([query], top_k, filters, query_type)[0]

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalResult]:
        """Encode each source family once and fuse every query in order."""

        if not queries:
            return []
        if not all(
            hasattr(retriever, "encode")
            and hasattr(retriever, "search_vectors")
            and hasattr(retriever, "source_family")
            for retriever in self.retrievers
        ):
            return [
                self._search_legacy(query, top_k, filters, query_type)
                for query in queries
            ]

        batches: dict[str, Any] = {}
        encoding_trace = RetrievalTrace()
        jobs: list[ModalitySearchJob] = []
        for retriever in self.retrievers:
            family = retriever.source_family
            if family not in batches:
                batch = retriever.encode(queries)
                batches[family] = batch
                encoding_trace = encoding_trace.merged(
                    RetrievalTrace(
                        stages={batch.encoding_trace.stage: batch.encoding_trace}
                    ),
                    prefix=family,
                )
            jobs.append(
                ModalitySearchJob(
                    source=retriever.source,
                    query_batch=batches[family],
                    index=retriever,
                    top_k=top_k,
                    filters=filters,
                    query_type=query_type,
                )
            )

        modality_results = self._executor.search(
            jobs,
            self.config.required_sources,
        )
        retrievers_by_source = {
            retriever.source: retriever for retriever in self.retrievers
        }
        active_sources = {
            result.source for result in modality_results if result.succeeded
        }

        results: list[RetrievalResult] = []
        for query_index in range(len(queries)):
            children: list[tuple[Any, RetrievalResult]] = []
            trace = encoding_trace
            warnings: list[str] = []
            for modality in modality_results:
                if modality.succeeded:
                    result = modality.query_results[query_index]
                    children.append((retrievers_by_source[modality.source], result))
                    trace = trace.merged(
                        result.trace,
                        prefix=modality.source.value,
                    )
                else:
                    if modality.failure_trace is not None:
                        trace = trace.merged(
                            modality.failure_trace,
                            prefix=modality.source.value,
                        )
                    if modality.warning is not None:
                        warnings.append(modality.warning)
            results.append(
                self._fuse(
                    children,
                    top_k,
                    query_type,
                    trace,
                    warnings=warnings,
                    active_sources=active_sources,
                )
            )
        return results

    def _search_legacy(
        self,
        query: str,
        top_k: int,
        filters: SearchFilters | None,
        query_type: TaskType,
    ) -> RetrievalResult:
        """Support existing retriever adapters without a batch interface."""

        child_results: list[tuple[Any, RetrievalResult]] = []
        trace = RetrievalTrace()
        for index, retriever in enumerate(self.retrievers):
            raw_result = retriever.search(query, top_k, filters, query_type)
            result = (
                raw_result
                if isinstance(raw_result, RetrievalResult)
                else RetrievalResult(candidates=raw_result)
            )
            child_results.append((retriever, result))
            source = getattr(getattr(retriever, "source", None), "value", None)
            trace = trace.merged(
                result.trace,
                prefix=source or f"retriever_{index}",
            )

        return self._fuse(child_results, top_k, query_type, trace)

    def _fuse(
        self,
        child_results: list[tuple[Any, RetrievalResult]],
        top_k: int,
        query_type: TaskType,
        trace: RetrievalTrace,
        *,
        warnings: list[str] | None = None,
        active_sources: set[RetrievalSource] | None = None,
    ) -> RetrievalResult:
        """Fuse one query's modality results without changing identity."""

        result_warnings = [
            warning
            for _, result in child_results
            for warning in result.warnings
        ]
        result_warnings.extend(warnings or [])
        fusion_timer = StageTimer("fusion")
        pool: dict[str, RetrievalCandidate] = {}
        for _, result in child_results:
            for candidate in result.candidates:
                existing = pool.get(candidate.frame_id)
                pool[candidate.frame_id] = (
                    candidate.model_copy(deep=True)
                    if existing is None
                    else _merge(existing, candidate)
                )

        weights = self._active_weights(
            query_type,
            active_sources or _result_sources(child_results),
        )
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
        fusion_trace = fusion_timer.finish()
        trace = trace.merged(
            RetrievalTrace(stages={fusion_trace.stage: fusion_trace})
        )
        return RetrievalResult(
            candidates=fused[:top_k],
            trace=trace,
            warnings=result_warnings,
        )

    def _active_weights(
        self,
        query_type: TaskType,
        active_sources: set[RetrievalSource],
    ) -> dict[RetrievalSource, float]:
        weights = self.config.task_weights[query_type]
        configured_sources = {
            source
            for retriever in self.retrievers
            if (source := getattr(retriever, "source", None)) is not None
        }
        if not configured_sources:
            configured_sources = active_sources
        active_weights = {
            source: weights[source]
            for source in active_sources
        }
        if (
            not self.config.normalize_active_weights
            or active_sources == configured_sources
            or not active_weights
        ):
            return active_weights
        configured_total = sum(weights[source] for source in configured_sources)
        active_total = sum(active_weights.values())
        scale = configured_total / active_total
        return {
            source: weight * scale
            for source, weight in active_weights.items()
        }


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


def _result_sources(
    child_results: list[tuple[Any, RetrievalResult]],
) -> set[RetrievalSource]:
    return {
        source
        for _, result in child_results
        for candidate in result.candidates
        for source in candidate.source_ranks
    }


def _sort_key(candidate: RetrievalCandidate) -> tuple[float, int, str]:
    """Sort by fused score, strongest source rank, then stable frame identity."""

    best_rank = min(candidate.source_ranks.values(), default=2**31)
    return (-(candidate.fusion_score or 0.0), best_rank, candidate.frame_id)
