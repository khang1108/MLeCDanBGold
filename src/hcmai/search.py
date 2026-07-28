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

from hcmai.common.utils.logging import get_logger
from hcmai.schema import (
    RetrievalCandidate,
    SearchLatency,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchScores,
)

logger = get_logger(__name__)


class SearchEngine:
    """Run retrieval, optional reranking, and response materialization."""

    def __init__(
        self,
        frame_store: Any,
        retriever: Any,
        reranker: Any | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.frame_store = frame_store
        self.retriever = retriever
        self.reranker = reranker
        self.config = config or {}

    def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a search request using the configured search profile."""
        started = perf_counter()
        profile = self._get_profile(request.search_mode)
        request_id = self._request_id(request)
        visual_count = int(profile.get("visual_candidates", request.top_k))
        rerank_count = int(profile.get("rerank_count", 0))
        logger.info(
            "[%s] search started mode=%s top_k=%d candidates=%d rerank_count=%d "
            "reranker=%s query=%r", request_id, request.search_mode.value,
            request.top_k, visual_count, rerank_count, self.reranker is not None,
            _preview(request.query),
        )
        candidates, retrieval_ms, reranking_ms = self._rank_candidates(
            request, request_id, visual_count, rerank_count
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
        visual_count: int, rerank_count: int,
    ) -> tuple[list[RetrievalCandidate], int, int]:
        retrieval_started = perf_counter()
        logger.info("[%s] retrieval started", request_id)
        candidates = list(self.retriever.search(
            query=request.query, top_k=visual_count, filters=request.filters,
        ))
        retrieval_ms = self._elapsed_ms(retrieval_started)
        logger.info(
            "[%s] retrieval completed candidates=%d elapsed_ms=%d "
            "encoding_ms=%.1f index_ms=%.1f", request_id, len(candidates),
            retrieval_ms, float(getattr(self.retriever, "last_query_encoding_ms", 0)),
            float(getattr(self.retriever, "last_index_search_ms", 0)),
        )
        logger.info("[%s] fusion skipped reason=single_visual_source", request_id)
        if self.reranker is None or rerank_count <= 0:
            reason = "not_configured" if self.reranker is None else "profile_disabled"
            logger.info("[%s] reranking skipped reason=%s", request_id, reason)
            return candidates, retrieval_ms, 0
        started = perf_counter()
        depth = min(rerank_count, len(candidates))
        logger.info("[%s] reranking started candidates=%d", request_id, depth)
        candidates = self.reranker.rerank(
            query=request.query, candidates=candidates[:depth],
        ) + candidates[depth:]
        reranking_ms = self._elapsed_ms(started)
        fallbacks = sum(
            "reranker_fallback" in (item.metadata or {}) for item in candidates[:depth]
        )
        logger.info(
            "[%s] reranking completed candidates=%d fallbacks=%d elapsed_ms=%d",
            request_id, depth, fallbacks, reranking_ms,
        )
        return candidates, retrieval_ms, reranking_ms

    def _get_profile(self, search_mode: Any) -> Mapping[str, Any]:
        """Return the configured profile, with safe defaults for the skeleton."""

        search_config = self.config.get("search", {})
        profiles = search_config.get("profiles", {})
        mode = getattr(search_mode, "value", search_mode)

        profile = profiles.get(mode)
        if profile is None:
            profile = profiles.get("accurate", {})

        return profile

    def _build_response(
        self,
        request: SearchRequest,
        candidates: Sequence[RetrievalCandidate],
    ) -> SearchResponse:
        """Convert internal candidates into the public response contract."""

        results: list[SearchResult] = []
        for rank, candidate in enumerate(candidates, start=1):
            frame = self._materialize_frame(candidate)
            scores = self._build_scores(candidate)
            encoded_id = quote(candidate.frame_id, safe="")
            results.append(
                SearchResult(
                    rank=rank,
                    frame_id=candidate.frame_id,
                    video_id=self._field(frame, "video_id", ""),
                    frame_idx=int(self._field(frame, "frame_idx", 0)),
                    timestamp_ms=int(self._field(frame, "timestamp_ms", 0)),
                    thumbnail_url=self._field(frame, "thumbnail_url")
                    or f"/api/v1/frames/{encoded_id}/thumbnail",
                    frame_url=self._field(frame, "frame_url")
                    or f"/api/v1/frames/{encoded_id}/image",
                    caption=self._field(frame, "caption"),
                    ocr_text=self._field(frame, "ocr_text"),
                    asr_text=self._field(frame, "asr_text"),
                    scores=scores,
                )
            )

        return SearchResponse(
            request_id=self._request_id(request),
            query=request.query,
            search_mode=request.search_mode,
            top_k=request.top_k,
            total_results=len(results),
            latency_ms=SearchLatency(total=0),
            results=results,
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

        payload = f"{request.query}\0{request.top_k}".encode()
        return f"search-{sha1(payload).hexdigest()[:12]}"

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int((perf_counter() - started) * 1_000))


def _preview(value: str, limit: int = 160) -> str:
    """Keep query logs useful without flooding the terminal."""
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"
