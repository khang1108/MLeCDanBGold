"""KIS projection of shared canonical temporal-search paths.

This workflow splits raw KIS text, delegates retrieval and alignment to the
shared temporal service, and projects each returned path to one representative
frame. It does not own retrieval, reranking, task dispatch, or trace payloads.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from hcmai.api.contracts import SearchLatency, SearchRequest, SearchResponse
from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.corpus import Corpus
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.orchestration.workflows.temporal_search import TemporalSearchService
from hcmai.temporal import split_query_events

if TYPE_CHECKING:
    from hcmai.query_preparation.service import QueryPreparationService


class KISPipeline:
    """Project aligned paths into deterministic KIS representative results."""

    def __init__(
        self,
        corpus: Corpus | None,
        temporal: TemporalSearchService | None,
        query_preparation: QueryPreparationService | None = None,
        max_temporal_event_count: int = DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    ) -> None:
        """Bind canonical materialization and the shared temporal service."""

        self.corpus = corpus
        self.temporal = temporal
        self.query_preparation = query_preparation
        self.max_temporal_event_count = max_temporal_event_count
        self.materializer = SearchMaterializer(corpus) if corpus is not None else None

    def execute(self, request: SearchRequest) -> SearchResponse:
        """Split query text, search temporal paths, and materialize each midpoint.

        The representative is the upper-middle aligned frame. Full frame and
        timestamp arrays are preserved for frontend inspection of the path.
        """

        started = perf_counter()

        query_started = perf_counter()
        if not request.query.strip():
            query_ms = (perf_counter() - query_started) * 1_000
            total_ms = (perf_counter() - started) * 1_000
            return SearchResponse(
                query=request.query,
                events=[],
                dense_events=[] if request.use_dense else None,
                bm25_caption_events=None,
                use_dense=request.use_dense,
                use_bm25=request.use_bm25,
                results=[],
                latency=SearchLatency(
                    query_ms=query_ms,
                    retrieval_ms=0,
                    alignment_ms=0,
                    materialization_ms=0,
                    total_ms=total_ms,
            ),)

        if self.corpus is None or self.materializer is None:
            raise RuntimeError("canonical frame data is not loaded")
        if self.temporal is None:
            raise RuntimeError("temporal search service is not loaded")

        events = split_query_events(request.query)
        if len(events) > self.max_temporal_event_count:
            raise ValueError(
                f"requests may contain at most {self.max_temporal_event_count} temporal events"
            )
        retrieval_events = (
            tuple(request.retrieval_events) if request.retrieval_events is not None else events
        )
        if len(retrieval_events) != len(events):
            raise ValueError("retrieval_events must match the original event count")
        # The published context corpus is Vietnamese. Candidate rewrites may
        # improve Dense recall, but must not replace the literal BM25 query.
        caption_events = events if request.use_bm25 else None
        query_ms = (perf_counter() - query_started) * 1_000

        search = self.temporal.search(
            events,
            retrieval_events=retrieval_events,
            caption_events=caption_events,
            use_dense=request.use_dense,
            use_bm25=request.use_bm25,
            top_k=request.top_k,
        )

        materialization_started = perf_counter()
        results = [self.materializer.build_kis_result(path) for path in search.paths]
        materialization_ms = (perf_counter() - materialization_started) * 1_000
        total_ms = (perf_counter() - started) * 1_000

        return SearchResponse(
            query=request.query,
            events=list(events),
            dense_events=list(retrieval_events) if request.use_dense else None,
            bm25_caption_events=(list(caption_events) if caption_events else None),
            use_dense=request.use_dense,
            use_bm25=request.use_bm25,
            results=results,
            latency=SearchLatency(
                query_ms=query_ms,
                retrieval_ms=search.retrieval_ms,
                alignment_ms=search.alignment_ms,
                materialization_ms=materialization_ms,
                total_ms=total_ms,
        ),)
