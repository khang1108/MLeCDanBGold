"""Internal retrieval and reranking stage for :class:`SearchService`."""

from __future__ import annotations

from hashlib import sha1
from time import perf_counter

from hcmai.common.schemas import (
    RetrievalResult,
    RetrievalTrace,
    SearchRequest,
    StageStatus,
    StageTrace,
)
from hcmai.common.utils.logging import get_logger
from hcmai.observability.tracing import StageTimer, log_stage
from hcmai.observability import PipelineStage
from hcmai.retrieval.reranking.pipeline import RerankingError, RerankingService
from hcmai.retrieval.retriever.pipeline import RetrievalService

logger = get_logger(__name__)


def rank_candidates(
    request: SearchRequest,
    retrieval: RetrievalService,
    reranking: RerankingService | None,
    *,
    candidate_count: int,
    rerank_count: int,
    request_id: str,
) -> tuple[RetrievalResult, int]:
    """Retrieve and optionally rerank one bounded candidate list."""

    retrieval_timer = StageTimer(PipelineStage.SEARCH.value)
    logger.info("[%s] retrieval started", request_id)
    raw_result = retrieval.search(
        query=request.query,
        top_k=candidate_count,
        filters=request.filters,
        query_type=request.query_type,
    )
    result = (
        raw_result
        if isinstance(raw_result, RetrievalResult)
        else RetrievalResult(candidates=raw_result)
    )
    retrieval_stage = retrieval_timer.finish(
        input_count=1,
        output_count=len(result.candidates),
        backend=type(retrieval).__name__,
    )
    if not result.trace.stages:
        result = result.model_copy(
            update={
                "trace": RetrievalTrace(
                    stages={retrieval_stage.stage: retrieval_stage}
                )
            }
        )
    candidates = result.candidates
    for trace in result.trace.stages.values():
        log_stage(
            logger,
            request_id=request_id,
            task_type=request.query_type,
            trace=trace,
        )
    logger.info(
        "[%s] retrieval completed candidates=%d elapsed_ms=%d "
        "encoding_ms=%.1f index_ms=%.1f",
        request_id,
        len(candidates),
        int(retrieval_stage.duration_ms),
        result.trace.duration_for("query_encoding"),
        result.trace.duration_for("index_search"),
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
        skipped = StageTimer(PipelineStage.RERANK.value).finish(
            status=StageStatus.SKIPPED,
            attempt_count=0,
            input_count=len(candidates),
            output_count=len(candidates),
        )
        log_stage(
            logger,
            request_id=request_id,
            task_type=request.query_type,
            trace=skipped,
        )
        return _with_reranking_trace(result, skipped), 0
    reranking_timer = StageTimer(PipelineStage.RERANK.value)
    depth = min(rerank_count, len(candidates))
    logger.info("[%s] reranking started candidates=%d", request_id, depth)
    try:
        ranked = reranking.rerank(request.query, candidates[:depth])
    except RerankingError as error:
        reranking_trace = reranking_timer.finish(
            status=StageStatus.PARTIAL,
            error_category=error.category,
            input_count=depth,
            output_count=len(candidates),
            backend=_reranker_backend(reranking),
            fallback_used=True,
        )
        log_stage(
            logger,
            request_id=request_id,
            task_type=request.query_type,
            trace=reranking_trace,
        )
        logger.warning(
            "[%s] reranking fallback category=%s candidates=%d",
            request_id,
            error.category,
            depth,
        )
        if reranking.config.required:
            raise
        fallback = result.model_copy(
            update={
                "warnings": [
                    *result.warnings,
                    f"reranking fallback ({error.category})",
                ]
            }
        )
        return (
            _with_reranking_trace(fallback, reranking_trace),
            int(reranking_trace.duration_ms),
        )
    ranked.extend(candidates[depth:])
    reranking_trace = reranking_timer.finish(
        input_count=depth,
        output_count=len(ranked),
        backend=_reranker_backend(reranking),
    )
    reranking_ms = int(reranking_trace.duration_ms)
    log_stage(
        logger,
        request_id=request_id,
        task_type=request.query_type,
        trace=reranking_trace,
    )
    logger.info(
        "[%s] reranking completed candidates=%d elapsed_ms=%d",
        request_id,
        depth,
        reranking_ms,
    )
    updated = result.model_copy(update={"candidates": ranked})
    return _with_reranking_trace(updated, reranking_trace), reranking_ms


def request_id(request: SearchRequest) -> str:
    """Build the stable request identifier used by pipeline telemetry."""

    payload = f"{request.query_type.value}\0{request.query}\0{request.top_k}".encode()
    return f"search-{sha1(payload).hexdigest()[:12]}"


def elapsed_ms(started: float) -> int:
    return max(0, int((perf_counter() - started) * 1_000))


def _with_reranking_trace(
    result: RetrievalResult,
    stage: StageTrace,
) -> RetrievalResult:
    trace = result.trace.merged(
        RetrievalTrace(stages={stage.stage: stage})
    )
    return result.model_copy(update={"trace": trace})


def _reranker_backend(reranking: RerankingService) -> str:
    adapter = getattr(reranking, "adapter", reranking)
    return type(adapter).__name__
