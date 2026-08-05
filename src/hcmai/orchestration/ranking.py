"""Internal retrieval and reranking stage for :class:`SearchService`."""

from __future__ import annotations

from hashlib import sha1
from time import perf_counter

from hcmai.common.schemas import RetrievalCandidate, SearchRequest
from hcmai.common.utils.logging import get_logger
from hcmai.reranking.pipeline import RerankingService
from hcmai.retriever.pipeline import RetrievalService

logger = get_logger(__name__)


def rank_candidates(
    request: SearchRequest,
    retrieval: RetrievalService,
    reranking: RerankingService | None,
    *,
    candidate_count: int,
    rerank_count: int,
    request_id: str,
) -> tuple[list[RetrievalCandidate], int, int]:
    """Retrieve and optionally rerank one bounded candidate list."""

    retrieval_started = perf_counter()
    logger.info("[%s] retrieval started", request_id)
    candidates = retrieval.search(
        query=request.query,
        top_k=candidate_count,
        filters=request.filters,
        query_type=request.query_type,
    )
    retrieval_ms = elapsed_ms(retrieval_started)
    logger.info(
        "[%s] retrieval completed candidates=%d elapsed_ms=%d "
        "encoding_ms=%.1f index_ms=%.1f",
        request_id,
        len(candidates),
        retrieval_ms,
        retrieval.last_query_encoding_ms,
        retrieval.last_index_search_ms,
    )
    sources = sorted({
        source.value
        for candidate in candidates
        for source in candidate.source_ranks
    })
    fused_count = sum(item.fusion_score is not None for item in candidates)
    if fused_count:
        logger.info(
            "[%s] fusion completed sources=%s candidates=%d",
            request_id, sources, fused_count,
        )
    else:
        logger.info(
            "[%s] fusion skipped reason=single_source sources=%s",
            request_id, sources,
        )
    if reranking is None or rerank_count <= 0:
        reason = "not_configured" if reranking is None else "disabled"
        logger.info("[%s] reranking skipped reason=%s", request_id, reason)
        return candidates, retrieval_ms, 0
    started = perf_counter()
    depth = min(rerank_count, len(candidates))
    logger.info("[%s] reranking started candidates=%d", request_id, depth)
    ranked = reranking.rerank(request.query, candidates[:depth])
    ranked.extend(candidates[depth:])
    reranking_ms = elapsed_ms(started)
    logger.info(
        "[%s] reranking completed candidates=%d elapsed_ms=%d",
        request_id,
        depth,
        reranking_ms,
    )
    return ranked, retrieval_ms, reranking_ms


def request_id(request: SearchRequest) -> str:
    """Build the stable request identifier used by pipeline telemetry."""

    payload = f"{request.query_type.value}\0{request.query}\0{request.top_k}".encode()
    return f"search-{sha1(payload).hexdigest()[:12]}"


def elapsed_ms(started: float) -> int:
    return max(0, int((perf_counter() - started) * 1_000))
