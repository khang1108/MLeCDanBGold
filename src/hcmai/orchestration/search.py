"""Orchestration for the online frame-retrieval pipeline.

The search engine deliberately knows only the small interfaces needed from a
retriever, optional reranker, and frame store. Concrete implementations can be
added later without changing the public request and response contracts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha1
from time import perf_counter
from typing import Any
from urllib.parse import quote

from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalSource,
    SearchLatency,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchScores,
)
from hcmai.common.utils.logging import get_logger

logger = get_logger(__name__)


class SearchEngine:
    """Run retrieval, optional reranking, and response materialization."""

    def __init__(
        self,
        frame_store: Any,
        retriever: Any,
        reranker: Any | None = None,
        config: Mapping[str, Any] | None = None,
        evidence_stores: Mapping[RetrievalSource, Any] | None = None,
    ) -> None:
        self.frame_store = frame_store
        self.retriever = retriever
        self.reranker = reranker
        self.config = config or {}
        self.evidence_stores = dict(evidence_stores or {})

    def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a search request using the single configured pipeline."""
        started = perf_counter()
        search_config = self._search_config()
        request_id = self._request_id(request)
        candidate_count = max(
            request.top_k, int(search_config.get("candidate_count", request.top_k))
        )
        rerank_count = int(search_config.get("rerank_count", 0))
        logger.info(
            "[%s] search started query_type=%s top_k=%d candidates=%d "
            "rerank_count=%d reranker=%s query=%r", request_id,
            request.query_type.value,
            request.top_k, candidate_count, rerank_count, self.reranker is not None,
            _preview(request.query),
        )
        candidates, retrieval_ms, reranking_ms = self._rank_candidates(
            request, request_id, candidate_count, rerank_count
        )
        selected = candidates[: request.top_k]
        materialization_started = perf_counter()
        logger.info(
            "[%s] materialization started selected=%d", request_id, len(selected)
        )
        response = self._build_response(request, selected)
        materialization_ms = self._elapsed_ms(materialization_started)
        latency = response.latency_ms.model_copy(update={
            "candidate_retrieval": retrieval_ms,
            "reranking": reranking_ms,
            "materialization": materialization_ms,
            "total": self._elapsed_ms(started),
        })
        response = response.model_copy(update={"latency_ms": latency})
        logger.info(
            "[%s] search completed results=%d total_ms=%d retrieval_ms=%d "
            "reranking_ms=%d materialization_ms=%d top_frames=%s",
            request_id,
            response.total_results,
            latency.total,
            latency.candidate_retrieval,
            latency.reranking,
            latency.materialization,
            [item.frame_id for item in response.results[:5]],
        )
        return response

    def _rank_candidates(
        self, request: SearchRequest, request_id: str,
        candidate_count: int, rerank_count: int,
    ) -> tuple[list[RetrievalCandidate], int, int]:
        retrieval_started = perf_counter()
        logger.info("[%s] retrieval started", request_id)
        candidates = list(self.retriever.search(
            query=request.query, top_k=candidate_count, filters=request.filters,
            query_type=request.query_type,
        ))
        retrieval_ms = self._elapsed_ms(retrieval_started)
        logger.info(
            "[%s] retrieval completed candidates=%d elapsed_ms=%d "
            "encoding_ms=%.1f index_ms=%.1f", request_id, len(candidates),
            retrieval_ms, float(getattr(self.retriever, "last_query_encoding_ms", 0)),
            float(getattr(self.retriever, "last_index_search_ms", 0)),
        )
        sources = sorted({
            source.value
            for candidate in candidates
            for source in candidate.source_ranks
        })
        fused_count = sum(
            candidate.fusion_score is not None for candidate in candidates
        )
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
        if self.reranker is None or rerank_count <= 0:
            reason = "not_configured" if self.reranker is None else "disabled"
            logger.info("[%s] reranking skipped reason=%s", request_id, reason)
            return candidates, retrieval_ms, 0
        started = perf_counter()
        depth = min(rerank_count, len(candidates))
        logger.info("[%s] reranking started candidates=%d", request_id, depth)
        candidates = self.reranker.rerank(
            query=request.query, candidates=candidates[:depth],
        ) + candidates[depth:]
        reranking_ms = self._elapsed_ms(started)
        logger.info(
            "[%s] reranking completed candidates=%d elapsed_ms=%d",
            request_id, depth, reranking_ms,
        )
        return candidates, retrieval_ms, reranking_ms

    def _search_config(self) -> Mapping[str, Any]:
        """Return the single configured search pipeline values."""

        return self.config.get("search", {})

    def _build_response(
        self,
        request: SearchRequest,
        candidates: Sequence[RetrievalCandidate],
    ) -> SearchResponse:
        """Convert internal candidates into the public response contract."""

        results = [
            self._build_result(candidate, rank)
            for rank, candidate in enumerate(candidates, start=1)
        ]
        return SearchResponse(
            request_id=self._request_id(request),
            query=request.query,
            query_type=request.query_type,
            top_k=request.top_k,
            total_results=len(results),
            latency_ms=SearchLatency(total=0),
            results=results,
        )

    def _build_result(
        self, candidate: RetrievalCandidate, rank: int
    ) -> SearchResult:
        """Combine canonical metadata and optional text evidence."""

        frame = self._materialize_frame(candidate)
        encoded_id = quote(candidate.frame_id, safe="")
        fields = {
            RetrievalSource.CAPTION: "caption",
            RetrievalSource.OCR: "ocr_text",
            RetrievalSource.ASR: "asr_text",
        }
        text = {
            field: self._field(frame, field)
            or self._evidence_text(candidate.frame_id, source)
            for source, field in fields.items()
        }
        return SearchResult(
            rank=rank,
            frame_id=candidate.frame_id,
            video_id=self._field(frame, "video_id", ""),
            frame_idx=int(self._field(frame, "frame_idx", 0)),
            timestamp_ms=int(self._field(frame, "timestamp_ms", 0)),
            thumbnail_url=self._field(frame, "thumbnail_url")
            or f"/api/v1/frames/{encoded_id}/thumbnail",
            frame_url=self._field(frame, "frame_url")
            or f"/api/v1/frames/{encoded_id}/image",
            caption=text["caption"],
            ocr_text=text["ocr_text"],
            asr_text=text["asr_text"],
            scores=self._build_scores(candidate),
        )

    def _materialize_frame(self, candidate: RetrievalCandidate) -> Any:
        """Resolve frame metadata from candidate metadata or the frame store."""

        metadata = candidate.metadata or {}
        frame = metadata.get("frame", metadata)
        if self._field(frame, "video_id") is not None:
            return frame

        for method_name in ("get", "get_by_id", "lookup"):
            method = getattr(self.frame_store, method_name, None)
            if method is not None:
                frame = method(candidate.frame_id)
                if frame is not None:
                    return frame

        raise ValueError(
            f"No frame metadata found for candidate frame_id={candidate.frame_id!r}"
        )

    def _evidence_text(
        self, frame_id: str, source: RetrievalSource
    ) -> str | None:
        """Read optional text without making missing enrichment fatal."""

        store = self.evidence_stores.get(source)
        if store is None:
            return None
        try:
            return store.get_text(frame_id)
        except KeyError:
            return None

    @staticmethod
    def _build_scores(candidate: RetrievalCandidate) -> SearchScores:
        """Map source and pipeline scores to the public score contract."""

        source_scores = candidate.source_scores
        values = {
            getattr(key, "value", key): value
            for key, value in source_scores.items()
        }
        final = candidate.final_score
        if final is None:
            final = candidate.reranker_score
        if final is None:
            final = candidate.fusion_score
        if final is None:
            final = values.get("visual", 0.0)

        return SearchScores(
            visual=values.get("visual"),
            caption=values.get("caption"),
            ocr=values.get("ocr"),
            asr=values.get("asr"),
            fusion=candidate.fusion_score,
            reranker=candidate.reranker_score,
            final=final,
        )

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _request_id(request: SearchRequest) -> str:
        """Return a stable placeholder until request IDs are injected."""

        payload = (
            f"{request.query_type.value}\0{request.query}\0{request.top_k}"
        ).encode()
        return f"search-{sha1(payload).hexdigest()[:12]}"

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int((perf_counter() - started) * 1_000))


def _preview(value: str, limit: int = 160) -> str:
    """Keep query logs useful without flooding the terminal."""
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"
