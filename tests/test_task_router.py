from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest

from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalSource,
    SearchRequest,
    SearchResponse,
    TaskType,
)
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import (
    SearchPipelineUnavailableError,
    SearchService,
)
from hcmai.orchestration.task_router import PipelineRegistry
from hcmai.retriever.pipeline import RetrievalService


@dataclass(frozen=True)
class StubPipeline:
    task_type: TaskType

    def execute(self, request: SearchRequest) -> SearchResponse:
        raise AssertionError(f"unexpected execution for {request.query_type.value}")


class Data:
    record_count = 1

    def get_frame(self, frame_id: str):
        if frame_id != "f1":
            raise KeyError(frame_id)
        return SimpleNamespace(
            frame_id="f1",
            video_id="official-video",
            frame_idx=42,
            timestamp_ms=1_000,
        )

    def get_evidence(self, frame_id, source):
        del frame_id, source
        return None

    def has_evidence(self, source) -> bool:
        del source
        return False


class Retrieval:
    last_query_encoding_ms = 1
    last_index_search_ms = 2

    def __init__(self) -> None:
        self.query_types: list[TaskType] = []

    def search(self, query, top_k, filters, query_type):
        del query, top_k, filters
        self.query_types.append(query_type)
        return [
            RetrievalCandidate(
                frame_id="f1",
                source_scores={RetrievalSource.VISUAL: 0.5},
                final_score=0.5,
            )
        ]


def test_register_and_get_pipeline() -> None:
    pipeline = StubPipeline(TaskType.KIS)
    registry = PipelineRegistry()

    registry.register(pipeline)

    assert registry.get(TaskType.KIS) is pipeline
    assert registry.capability_report((TaskType.KIS, TaskType.VQA)) == {
        "kis": True,
        "vqa": False,
    }


def test_duplicate_registration_is_rejected() -> None:
    registry = PipelineRegistry([StubPipeline(TaskType.KIS)])

    with pytest.raises(ValueError, match="'kis' is already registered"):
        registry.register(StubPipeline(TaskType.KIS))


def test_get_unsupported_task_raises_key_error() -> None:
    registry = PipelineRegistry([StubPipeline(TaskType.KIS)])

    with pytest.raises(KeyError):
        registry.get(TaskType.VQA)


@pytest.mark.parametrize("query_type", [TaskType.KIS, TaskType.VKIS])
def test_kis_pipeline_preserves_search_response_behavior(
    query_type: TaskType,
) -> None:
    retrieval = Retrieval()
    service = SearchService(
        cast(DataService, Data()), cast(RetrievalService, retrieval)
    )

    response = service.search(
        SearchRequest(query="red bus", query_type=query_type, top_k=5)
    )

    assert response.query == "red bus"
    assert response.query_type is query_type
    assert response.top_k == 5
    assert response.total_results == 1
    assert response.results[0].frame_id == "f1"
    assert response.results[0].video_id == "official-video"
    assert response.results[0].frame_idx == 42
    assert retrieval.query_types == [query_type]


def test_missing_pipeline_maps_to_typed_service_error() -> None:
    service = SearchService(None, None)

    with pytest.raises(SearchPipelineUnavailableError, match="vqa"):
        service.search(SearchRequest(query="question", query_type=TaskType.VQA))


def test_health_task_availability_is_derived_from_registry() -> None:
    registry = PipelineRegistry([StubPipeline(TaskType.VQA)])
    service = SearchService(
        cast(DataService, Data()),
        cast(RetrievalService, Retrieval()),
        pipeline_registry=registry,
    )

    assert service.health()["capabilities"]["query_types"] == {
        "kis": False,
        "vkis": False,
        "vqa": True,
        "trake": False,
    }


def test_trake_is_registered_by_default() -> None:
    service = SearchService(
        cast(DataService, Data()), cast(RetrievalService, Retrieval())
    )

    assert service.pipeline_registry.get(TaskType.TRAKE).task_type is TaskType.TRAKE
    assert service.health()["capabilities"]["query_types"]["trake"] is True
