"""Reciprocal-rank fusion over candidates from independent retrievers."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any

from hcmai.common.config import FusionConfig
from hcmai.common.schemas import (
    RetrievalResult,
    RetrievalSource,
    RetrievalTrace,
)
from hcmai.common.schemas.retrieval import RetrievalCandidate
from hcmai.common.observability import PipelineStage
from hcmai.common.observability.tracing import StageTimer
from hcmai.retrieval.retriever.concurrent import (
    ModalitySearchExecutor,
    ModalitySearchJob,
)
from hcmai.retrieval.retriever.models.contracts import VectorRetriever


class RRFFusionRetriever:
    """Union candidates by frame ID and rank them with reciprocal rank fusion."""

    def __init__(
        self,
        retrievers: Sequence[VectorRetriever],
        config: FusionConfig,
        executor: ModalitySearchExecutor | None = None,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("RRF fusion requires at least two retrievers")
        if config.method != "rrf":
            raise ValueError(f"Unsupported fusion method {config.method!r}")
        self.retrievers: tuple[VectorRetriever, ...] = tuple(retrievers)
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
        if any(
            not all(
                hasattr(retriever, attribute)
                for attribute in ("encode", "search_vectors", "source_family")
            )
            for retriever in self.retrievers
        ):
            raise TypeError(
                "RRFFusionRetriever requires vector retrievers with encode, "
                "search_vectors, and source_family"
            )
        self._executor = executor or ModalitySearchExecutor(
            config.modality_max_workers
        )

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> RetrievalResult:
        """Retrieve, merge exact frame IDs, and apply source weights."""

        return self.search_batch([query], top_k)[0]

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 100,
    ) -> list[RetrievalResult]:
        """Encode each source family once and fuse every query in order."""

        if not queries:
            return []
        started = perf_counter()
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
                    trace,
                    warnings=warnings,
                    active_sources=active_sources,
                )
            )
        first_candidate_ms = (perf_counter() - started) * 1_000
        return [
            result.model_copy(
                update={"time_to_first_candidate_ms": first_candidate_ms}
            )
            for result in results
        ]

    def _fuse(
        self,
        child_results: list[tuple[Any, RetrievalResult]],
        top_k: int,
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
        fusion_timer = StageTimer(PipelineStage.FUSION.value)
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
        fusion_trace = fusion_timer.finish(
            input_count=sum(len(result.candidates) for _, result in child_results),
            output_count=min(top_k, len(fused)),
            backend="rrf",
        )
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
        active_sources: set[RetrievalSource],
    ) -> dict[RetrievalSource, float]:
        weights = self.config.source_weights
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
            # The first candidate owns canonical frame metadata. Preserve it
            # while retaining distinct source provenance such as ASR segments.
            "metadata": {
                **incoming.metadata,
                **existing.metadata,
            },
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
