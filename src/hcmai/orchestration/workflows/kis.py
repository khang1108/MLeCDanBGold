"""KIS projection of shared canonical temporal-search paths.

This workflow splits raw KIS text, delegates retrieval and alignment to the
shared temporal service, and projects each returned path to one representative
frame. It does not own retrieval, reranking, task dispatch, or trace payloads.
"""

from __future__ import annotations

from time import perf_counter

from hcmai.api.contracts import SearchLatency, SearchRequest, SearchResponse
from hcmai.data.pipeline import DataService
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.orchestration.temporal_search import TemporalSearchService
from hcmai.temporal import split_query_events


class KISPipeline:
    """Project aligned paths into deterministic KIS representative results."""

    def __init__(
        self,
        data: DataService | None,
        alignment: TemporalSearchService | None,
    ) -> None:
        """Bind canonical materialization and the shared temporal service."""

        self.data = data
        self.alignment = alignment
        self.materializer = SearchMaterializer(data) if data is not None else None

    def execute(self, request: SearchRequest) -> SearchResponse:
        """Split query text, search temporal paths, and materialize each midpoint.

        The representative is the upper-middle aligned frame. Full frame and
        timestamp arrays are preserved for frontend inspection of the path.
        """

        if self.data is None or self.materializer is None:
            raise RuntimeError("canonical frame data is not loaded")
        if self.alignment is None:
            raise RuntimeError("temporal search service is not loaded")

        started = perf_counter()

        query_started = perf_counter()
        events = split_query_events(request.query)
        query_ms = (perf_counter() - query_started) * 1_000

        search = self.alignment.search(events, top_k=request.top_k)

        materialization_started = perf_counter()
        results = [
            self.materializer.build_kis_result(path) for path in search.paths
        ]
        materialization_ms = (perf_counter() - materialization_started) * 1_000
        total_ms = (perf_counter() - started) * 1_000

        return SearchResponse(
            query=request.query,
            events=list(events),
            results=results,
            latency=SearchLatency(
                query_ms=query_ms,
                retrieval_ms=search.retrieval_ms,
                alignment_ms=search.alignment_ms,
                materialization_ms=materialization_ms,
                total_ms=total_ms,
            ),
        )
