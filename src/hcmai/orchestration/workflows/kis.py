"""KIS projection of shared canonical temporal-search paths.

This workflow executes temporal retrieval and alignment from resolved KISIntent
events, delegates to the shared temporal service, and projects each returned path
to one representative frame. It does not own query planning, semantic parsing,
retrieval indexes, or reranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from hcmai.api.contracts.search import SearchLatency, SearchResult
from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.corpus import Corpus
from hcmai.kis.models import KISIntent
from hcmai.orchestration.utils.errors import InvalidQueryInputError
from hcmai.orchestration.utils.materializer import SearchMaterializer
from hcmai.orchestration.workflows.temporal_search import TemporalSearchService
from hcmai.retrieval.plan import KISRetrievalPlan


@dataclass(frozen=True, slots=True)
class KISSearchExecution:
    """KIS search results and stage timings projected from aligned paths."""

    results: list[SearchResult]
    latency: SearchLatency


class KISPipeline:
    """Project aligned paths into deterministic KIS representative results."""

    def __init__(
        self,
        corpus: Corpus | None,
        temporal: TemporalSearchService | None,
        max_temporal_event_count: int = DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    ) -> None:
        """Bind canonical materialization and the shared temporal service."""
        self.corpus = corpus
        self.temporal = temporal
        self.max_temporal_event_count = max_temporal_event_count
        self.materializer = SearchMaterializer(corpus) if corpus is not None else None

    def execute(
        self,
        *,
        intent: KISIntent,
        retrieval_plan: KISRetrievalPlan,
        use_dense: bool,
        use_bm25: bool,
        top_k: int,
        intent_ms: float = 0.0,
        translation_ms: float = 0.0,
    ) -> KISSearchExecution:
        """Search temporal paths using resolved intent events and materialize each midpoint.

        The representative is the upper-middle aligned frame. Full frame and
        timestamp arrays are preserved for frontend inspection of the path.
        """
        started = perf_counter()

        if self.corpus is None or self.materializer is None:
            raise RuntimeError("canonical frame data is not loaded")
        if self.temporal is None:
            raise RuntimeError("temporal search service is not loaded")

        if retrieval_plan.event_ids != tuple(event.id for event in intent.events):
            raise InvalidQueryInputError(
                "retrieval plan event IDs must match intent event order"
            )
        if len(retrieval_plan.events) > self.max_temporal_event_count:
            raise ValueError(
                f"requests may contain at most {self.max_temporal_event_count} temporal events"
            )
        try:
            retrieval_plan.validate_text_sources(use_dense=use_dense, use_bm25=use_bm25)
        except ValueError as error:
            raise InvalidQueryInputError(str(error)) from error

        original_events = retrieval_plan.canonical_texts
        retrieval_bundle = retrieval_plan.dense_texts if use_dense else original_events
        caption_events = retrieval_plan.bm25_texts if use_bm25 else None

        search = self.temporal.search(
            original_events,
            retrieval_events=retrieval_bundle,
            caption_events=caption_events,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=top_k,
        )

        materialization_started = perf_counter()
        results = [self.materializer.build_kis_result(path) for path in search.paths]
        materialization_ms = (perf_counter() - materialization_started) * 1_000
        query_ms = intent_ms + translation_ms
        total_ms = (perf_counter() - started) * 1_000 + query_ms

        latency = SearchLatency(
            intent_ms=intent_ms,
            translation_ms=translation_ms,
            query_ms=query_ms,
            retrieval_ms=search.retrieval_ms,
            alignment_ms=search.alignment_ms,
            materialization_ms=materialization_ms,
            total_ms=total_ms,
        )
        return KISSearchExecution(results=results, latency=latency)
