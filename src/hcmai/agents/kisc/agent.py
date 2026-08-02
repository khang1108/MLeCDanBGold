"""Compose bounded conversation interpretation with frame search."""

from __future__ import annotations

from time import perf_counter

from hcmai.common.schemas import SearchRequest
from hcmai.common.schemas.enum import TaskType
from hcmai.common.schemas.kisc import KISCSearchRequest, KISCSearchResponse
from hcmai.common.utils.logging import get_logger
from hcmai.orchestration import SearchEngine

from .resolver import ConversationResolver

logger = get_logger(__name__)


class KISCAgent:
    """Execute one stateless resolve-then-search turn."""

    def __init__(
        self, resolver: ConversationResolver, search_engine: SearchEngine
    ) -> None:
        self.resolver = resolver
        self.search_engine = search_engine

    def search(self, request: KISCSearchRequest) -> KISCSearchResponse:
        """Resolve context once, search once, and apply explicit feedback."""
        started = perf_counter()
        logger.info(
            "KISC turn started history=%d previous_state=%s accepted=%d "
            "rejected=%d message=%r",
            len(request.history), request.previous_state is not None,
            len(request.feedback.accepted_frame_ids),
            len(request.feedback.rejected_frame_ids), _preview(request.current_message),
        )
        logger.info("KISC conversation resolution started")
        state = self.resolver.resolve(
            request.history,
            request.current_message,
            request.feedback,
            request.previous_state,
        )
        resolution_ms = max(0, int((perf_counter() - started) * 1_000))
        logger.info(
            "KISC conversation resolution completed elapsed_ms=%d "
            "standalone_query=%r positive=%d negative=%d uncertain=%d",
            resolution_ms, _preview(state.standalone_query),
            len(state.positive_constraints), len(state.negative_constraints),
            len(state.uncertain_constraints),
        )
        logger.info("KISC frame search started")
        response = self.search_engine.search(
            SearchRequest(
                query=state.standalone_query,
                query_type=TaskType.KISC,
                top_k=request.top_k,
                filters=request.filters,
            )
        )
        rejected = set(state.rejected_frame_ids)
        accepted = {
            frame_id: index for index, frame_id in enumerate(state.accepted_frame_ids)
        }
        results = [item for item in response.results if item.frame_id not in rejected]
        results.sort(
            key=lambda item: (
                item.frame_id not in accepted,
                accepted.get(item.frame_id, item.rank),
            )
        )
        results = [
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(results, start=1)
        ]
        search = response.model_copy(
            update={"results": results, "total_results": len(results)}
        )
        logger.info(
            "KISC turn completed results=%d accepted_promoted=%d "
            "rejected_filtered=%d search_ms=%d",
            len(results), len(accepted), len(response.results) - len(results),
            search.latency_ms.total,
        )
        return KISCSearchResponse(
            interpreted_state=state,
            resolution_latency_ms=resolution_ms,
            search=search,
            warnings=[],
        )


def _preview(value: str, limit: int = 160) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"
