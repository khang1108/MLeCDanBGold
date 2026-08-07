"""Bounded concurrent execution for independent modality searches."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from hcmai.common.schemas import (
    RetrievalResult,
    RetrievalSource,
    RetrievalTrace,
    StageStatus,
    TaskType,
)
from hcmai.common.schemas.search import SearchFilters
from hcmai.observability.tracing import StageTimer
from hcmai.observability import PipelineStage
from hcmai.retriever.query_batch import QueryEmbeddingBatch


class VectorSearchIndex(Protocol):
    def search_vectors(
        self,
        query_batch: QueryEmbeddingBatch,
        top_k: int,
        filters: SearchFilters | None,
        query_type: TaskType,
    ) -> list[RetrievalResult]: ...


class RequiredModalitySearchError(RuntimeError):
    """A configured required retrieval source could not be searched."""

    def __init__(self, source: RetrievalSource, category: str) -> None:
        super().__init__(
            f"Required {source.value} retrieval failed ({category})"
        )
        self.source = source
        self.category = category


@dataclass(frozen=True, slots=True)
class ModalitySearchJob:
    """One source's index search over a shared ordered query-vector batch."""

    source: RetrievalSource
    query_batch: QueryEmbeddingBatch
    index: VectorSearchIndex
    top_k: int
    filters: SearchFilters | None
    query_type: TaskType


@dataclass(frozen=True, slots=True)
class ModalitySearchResult:
    """Successful results or a categorized source-local failure."""

    source: RetrievalSource
    query_results: tuple[RetrievalResult, ...]
    failure_trace: RetrievalTrace | None = None
    warning: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure_trace is None


class ModalitySearchExecutor:
    """Reuse one bounded worker pool across retrieval requests."""

    def __init__(self, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be greater than zero")
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="hcmai-modality",
        )

    def search(
        self,
        jobs: list[ModalitySearchJob],
        required_sources: set[RetrievalSource],
    ) -> list[ModalitySearchResult]:
        """Run every configured source concurrently and preserve job order."""

        if not jobs:
            return []
        futures: list[
            tuple[
                ModalitySearchJob,
                Future[list[RetrievalResult]],
                StageTimer,
            ]
        ] = []
        for job in jobs:
            timer = StageTimer(PipelineStage.SEARCH.value)
            future = self._pool.submit(
                job.index.search_vectors,
                job.query_batch,
                job.top_k,
                job.filters,
                job.query_type,
            )
            futures.append((job, future, timer))

        results: list[ModalitySearchResult] = []
        required_failure: RequiredModalitySearchError | None = None
        for job, future, timer in futures:
            try:
                query_results = tuple(future.result())
                results.append(
                    ModalitySearchResult(
                        source=job.source,
                        query_results=query_results,
                    )
                )
            except Exception as error:
                category = type(error).__name__
                failure = timer.finish(
                    status=StageStatus.FAILED,
                    error_category=category,
                    input_count=len(job.query_batch.embeddings),
                    output_count=0,
                    backend="faiss_or_exact_subset",
                )
                warning = (
                    f"{job.source.value} retrieval unavailable ({category})"
                )
                results.append(
                    ModalitySearchResult(
                        source=job.source,
                        query_results=(),
                        failure_trace=RetrievalTrace(
                            stages={failure.stage: failure}
                        ),
                        warning=warning,
                    )
                )
                if job.source in required_sources and required_failure is None:
                    required_failure = RequiredModalitySearchError(
                        job.source,
                        category,
                    )
        if required_failure is not None:
            raise required_failure
        return results

    def close(self) -> None:
        """Release worker threads when the owning service shuts down."""

        self._pool.shutdown(wait=True, cancel_futures=False)
