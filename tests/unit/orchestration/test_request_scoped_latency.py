from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    RetrievalTrace,
    SearchRequest,
    StageStatus,
    StageTrace,
)
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService


def _stage(stage: str, duration_ms: float) -> StageTrace:
    return StageTrace(
        stage=stage,
        started_at=1.0,
        ended_at=1.0 + duration_ms / 1_000,
        duration_ms=duration_ms,
        status=StageStatus.SUCCESS,
    )


class CanonicalData:
    def get_frame(self, frame_id: str):
        return SimpleNamespace(
            frame_id=frame_id,
            video_id="video-1",
            frame_idx=42,
            timestamp_ms=1_000,
        )

    def get_evidence(self, frame_id, source):
        del frame_id, source
        return None


class FullyTracedRetrieval:
    def search(self, query, top_k, filters, query_type):
        del query, top_k, filters, query_type
        stages = {
            "visual.query_encoding": _stage("visual.query_encoding", 2.9),
            "visual.index_search": _stage("visual.index_search", 3.9),
            "fusion": _stage("fusion", 4.9),
        }
        return RetrievalResult(
            candidates=[
                RetrievalCandidate(
                    frame_id="frame-1",
                    source_scores={RetrievalSource.VISUAL: 0.9},
                    source_ranks={RetrievalSource.VISUAL: 1},
                )
            ],
            trace=RetrievalTrace(stages=stages),
            warnings=["caption modality unavailable"],
        )


def test_kis_response_latency_and_warnings_come_from_current_trace() -> None:
    service = SearchService(
        cast(DataService, CanonicalData()),
        cast(RetrievalService, FullyTracedRetrieval()),
    )

    response = service.search(SearchRequest(query="red bus", top_k=1))

    assert response.latency_ms.query_encoding == 2
    assert response.latency_ms.candidate_retrieval == 3
    assert response.latency_ms.fusion == 4
    assert response.warnings == ["caption modality unavailable"]
