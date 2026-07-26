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

from hcmai.schema import (
    RetrievalCandidate,
    SearchLatency,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchScores,
)


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

        retrieval_started = perf_counter()
        candidates = self.retriever.search(
            query=request.query,
            top_k=int(profile.get("visual_candidates", request.top_k)),
            filters=request.filters,
        )
        candidate_retrieval_ms = self._elapsed_ms(retrieval_started)

        reranking_ms = 0
        if self.reranker is not None and profile.get("rerank_count", 0) > 0:
            reranking_started = perf_counter()
            candidates = self.reranker.rerank(
                query=request.query,
                candidates=list(candidates)[: int(profile["rerank_count"])],
            )
            reranking_ms = self._elapsed_ms(reranking_started)

        selected = list(candidates)[: request.top_k]
        materialization_started = perf_counter()
        response = self._build_response(request, selected)
        materialization_ms = self._elapsed_ms(materialization_started)

        latency = response.latency_ms.model_copy(
            update={
                "candidate_retrieval": candidate_retrieval_ms,
                "reranking": reranking_ms,
                "materialization": materialization_ms,
                "total": self._elapsed_ms(started),
            }
        )
        return response.model_copy(update={"latency_ms": latency})

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
            results.append(
                SearchResult(
                    rank=rank,
                    frame_id=candidate.frame_id,
                    video_id=self._field(frame, "video_id", ""),
                    frame_idx=int(self._field(frame, "frame_idx", 0)),
                    timestamp_ms=int(self._field(frame, "timestamp_ms", 0)),
                    thumbnail_url=self._field(frame, "thumbnail_url"),
                    frame_url=self._field(frame, "frame_url"),
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
