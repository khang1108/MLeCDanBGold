from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from hcmai.common.schemas import (
    FrameRecord,
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
    UnsupportedSearchTaskError,
)
from hcmai.orchestration.workflows.base import TaskPipelineRequestError
from hcmai.orchestration.task_router import PipelineRegistry
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.video_scores import VideoEventScores


@dataclass(frozen=True)
class StubPipeline:
    task_type: TaskType

    def execute(self, request: SearchRequest) -> SearchResponse:
        raise AssertionError(f"unexpected execution for {request.query_type.value}")


@dataclass(frozen=True)
class RaisingPipeline:
    task_type: TaskType
    error: Exception

    def execute(self, request: SearchRequest) -> SearchResponse:
        del request
        raise self.error


class Data:
    record_count = 1

    def get_frame(self, frame_id: str):
        if frame_id != "f1":
            raise KeyError(frame_id)
        return FrameRecord(
            frame_id="f1",
            video_id="official-video",
            frame_idx=42,
            timestamp_ms=1_000,
            image_path="f1.jpg",
            width=640,
            height=360,
        )

    def get_evidence(self, frame_id, source):
        del frame_id, source
        return None

    def has_evidence(self, source) -> bool:
        del source
        return False


class Retrieval:
    def __init__(self) -> None:
        self.event_batches: list[list[str]] = []

    def score_event_videos(self, events, filters=None, **kwargs):
        """Return one canonical frame for each KIS event-scoring request."""

        del filters, kwargs
        self.event_batches.append(list(events))
        return [
            VideoEventScores(
                video_id="official-video",
                frame_ids=np.array(["f1"], dtype=object),
                frame_idx=np.array([42]),
                timestamps_ms=np.array([1_000]),
                scores=np.full((len(events), 1), 0.5),
            )
        ]


def test_register_and_get_pipeline() -> None:
    pipeline = StubPipeline(TaskType.KIS)
    registry = PipelineRegistry()

    registry.register(pipeline)

    assert registry.get(TaskType.KIS) is pipeline
    assert registry.capability_report((TaskType.KIS, TaskType.TRAKE)) == {
        "kis": True,
        "trake": False,
    }


def test_duplicate_registration_is_rejected() -> None:
    registry = PipelineRegistry([StubPipeline(TaskType.KIS)])

    with pytest.raises(ValueError, match="'kis' is already registered"):
        registry.register(StubPipeline(TaskType.KIS))


def test_get_unsupported_task_raises_key_error() -> None:
    registry = PipelineRegistry([StubPipeline(TaskType.KIS)])

    with pytest.raises(KeyError):
        registry.get(TaskType.TRAKE)


def test_kis_pipeline_preserves_search_response_behavior(
) -> None:
    retrieval = Retrieval()
    service = SearchService(
        cast(DataService, Data()), cast(RetrievalService, retrieval)
    )

    response = service.search(
        SearchRequest(query="red bus", query_type=TaskType.KIS, top_k=5)
    )

    assert response.query == "red bus"
    assert response.query_type is TaskType.KIS
    assert response.top_k == 5
    assert response.total_results == 1
    assert response.results[0].frame_ids == ["f1"]
    assert response.results[0].video_id == "official-video"
    assert response.results[0].frame_idx == 42
    assert retrieval.event_batches == [["red bus"]]


def test_missing_pipeline_maps_to_typed_service_error() -> None:
    service = SearchService(None, None, pipeline_registry=PipelineRegistry())

    with pytest.raises(SearchPipelineUnavailableError, match="kis"):
        service.search(SearchRequest(query="question", query_type=TaskType.KIS))


def test_pipeline_request_error_maps_to_unsupported_task() -> None:
    registry = PipelineRegistry(
        [
            RaisingPipeline(
                TaskType.KIS,
                TaskPipelineRequestError("request does not match pipeline"),
            )
        ]
    )
    service = SearchService(None, None, pipeline_registry=registry)

    with pytest.raises(UnsupportedSearchTaskError, match="does not match"):
        service.search(SearchRequest(query="question", query_type=TaskType.KIS))


def test_unexpected_pipeline_value_error_is_not_misclassified() -> None:
    registry = PipelineRegistry(
        [RaisingPipeline(TaskType.KIS, ValueError("invalid embedding shape"))]
    )
    service = SearchService(None, None, pipeline_registry=registry)

    with pytest.raises(ValueError, match="invalid embedding shape"):
        service.search(SearchRequest(query="question", query_type=TaskType.KIS))


def test_health_task_availability_is_derived_from_registry() -> None:
    registry = PipelineRegistry([StubPipeline(TaskType.TRAKE)])
    service = SearchService(
        cast(DataService, Data()),
        cast(RetrievalService, Retrieval()),
        pipeline_registry=registry,
    )

    assert service.health()["capabilities"]["query_types"] == {
        "kis": False,
        "trake": True,
    }


def test_trake_is_registered_by_default() -> None:
    service = SearchService(
        cast(DataService, Data()), cast(RetrievalService, Retrieval())
    )

    assert service.pipeline_registry.get(TaskType.TRAKE).task_type is TaskType.TRAKE
    assert service.pipeline_registry.capability_report() == {
        "kis": True,
        "trake": True,
    }
    assert service.health()["capabilities"]["query_types"] == {
        "kis": True,
        "trake": True,
    }
