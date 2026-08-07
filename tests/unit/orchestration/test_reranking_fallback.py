"""Reranking failures preserve fused retrieval as a valid KIS response."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    SearchRequest,
    StageStatus,
)
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.ranking import rank_candidates
from hcmai.reranking.config import RerankerConfig
from hcmai.reranking.pipeline import (
    RerankerContractError,
    RerankerInvalidScoreError,
    RerankerTimeoutError,
    RerankerUnavailableError,
    RerankingError,
    RerankingService,
)
from hcmai.retriever.pipeline import RetrievalService


def _candidates() -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            frame_id=frame_id,
            source_scores={RetrievalSource.VISUAL: score},
            source_ranks={RetrievalSource.VISUAL: rank},
            fusion_score=score,
            final_score=score,
        )
        for rank, (frame_id, score) in enumerate(
            (("f1", 0.9), ("f2", 0.8)),
            start=1,
        )
    ]


class Retrieval:
    def search(self, query, top_k, filters, query_type):
        del query, top_k, filters, query_type
        return RetrievalResult(candidates=_candidates())


class Data:
    def get_frame(self, frame_id):
        return SimpleNamespace(
            frame_id=frame_id,
            video_id="video-1",
            frame_idx=1 if frame_id == "f1" else 2,
            timestamp_ms=1000 if frame_id == "f1" else 2000,
        )

    def get_evidence(self, frame_id, source):
        del frame_id, source
        return None


class FailingReranker:
    def __init__(self, error: RerankingError, required: bool = False) -> None:
        self.error = error
        self.config = RerankerConfig(required=required)

    def rerank(self, query, candidates):
        del query, candidates
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        RerankerTimeoutError(),
        RerankerUnavailableError(),
        RerankerContractError(),
        RerankerInvalidScoreError(),
    ],
)
def test_optional_failure_returns_valid_kis_in_fused_order(error) -> None:
    service = SearchService(
        cast(DataService, Data()),
        cast(RetrievalService, Retrieval()),
        cast(RerankingService, FailingReranker(error)),
    )

    response = service.search(SearchRequest(query="red bus", top_k=2))

    assert [result.frame_id for result in response.results] == ["f1", "f2"]
    assert response.total_results == 2
    assert response.warnings == [f"reranking fallback ({error.category})"]


def test_fallback_trace_is_partial_and_does_not_expose_backend_detail() -> None:
    error = RerankerUnavailableError("image_load_failure")
    request = SearchRequest(query="query")

    result, _ = rank_candidates(
        request,
        cast(RetrievalService, Retrieval()),
        cast(RerankingService, FailingReranker(error)),
        candidate_count=20,
        rerank_count=10,
        request_id="request-1",
    )

    trace = result.trace.stages["rerank"]
    assert trace.status is StageStatus.PARTIAL
    assert trace.error_category == "image_load_failure"
    assert result.warnings == ["reranking fallback (image_load_failure)"]


def test_required_reranker_failure_is_not_silently_downgraded() -> None:
    service = SearchService(
        cast(DataService, Data()),
        cast(RetrievalService, Retrieval()),
        cast(
            RerankingService,
            FailingReranker(RerankerTimeoutError(), required=True),
        ),
    )

    with pytest.raises(RerankerTimeoutError):
        service.search(SearchRequest(query="query"))
