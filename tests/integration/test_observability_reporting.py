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
from hcmai.retriever.pipeline import RetrievalService


def _stage(stage: str, duration_ms: float) -> StageTrace:
    return StageTrace(
        stage=stage,
        started_at=1,
        ended_at=1 + duration_ms / 1_000,
        duration_ms=duration_ms,
        status=StageStatus.SUCCESS,
        input_count=1,
        output_count=1,
        backend="fixture",
    )


class Data:
    record_count = 1

    def get_frame(self, frame_id: str):
        return SimpleNamespace(
            frame_id=frame_id,
            video_id="video-1",
            frame_idx=17,
            timestamp_ms=680,
        )

    def get_evidence(self, frame_id, source):
        del frame_id, source
        return None

    def has_evidence(self, source):
        del source
        return False


class Retrieval:
    active_sources = (RetrievalSource.VISUAL,)

    def search(self, query, top_k, filters, query_type):
        del query, top_k, filters, query_type
        stages = {
            "visual.encode": _stage("visual.encode", 2),
            "visual.search": _stage("visual.search", 3),
            "fusion": _stage("fusion", 1),
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
            time_to_first_candidate_ms=4,
        )


def test_search_response_and_health_expose_canonical_observability() -> None:
    service = SearchService(
        cast(DataService, Data()),
        cast(RetrievalService, Retrieval()),
    )

    response = service.search(SearchRequest(query="person cooking", top_k=1))
    health = service.health()

    assert set(response.trace.stages) == {
        "parse",
        "visual.encode",
        "visual.search",
        "fusion",
        "rerank",
        "materialization",
    }
    assert response.latency_ms.time_to_first_candidate == 4
    assert response.latency_ms.time_to_first_submission >= 0
    assert health["capabilities"]["kis"] is True
    assert health["capabilities"]["vqa"] is True
    assert health["capabilities"]["shared_retrieval"] is True
    assert set(health["capabilities"]["remote_inference"]) == {
        "embedding",
        "reranking",
        "multi_image_vqa",
        "structured_parsing",
    }
    assert "latency_histograms_ms" in health["observability"]
