"""Regression tests for reranking in the default temporal KIS path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    FrameEvidence,
    FrameRecord,
    RetrievalSource,
    RetrievalTrace,
    SceneCandidate,
    SearchRequest,
    StageStatus,
    TaskType,
)
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.retrieval.reranking.config import RerankerConfig
from hcmai.retrieval.reranking.pipeline import RerankerUnavailableError
from hcmai.retrieval.retriever.pipeline import RetrievalService


class _Data:
    """Small canonical frame store used by the orchestration fixture."""

    def __init__(self) -> None:
        self.frames = {
            frame_id: FrameRecord(
                frame_id=frame_id,
                video_id="video-1",
                frame_idx=index,
                timestamp_ms=index * 1_000,
                image_path=f"{frame_id}.jpg",
                width=640,
                height=360,
            )
            for index, frame_id in enumerate(("frame-a", "frame-b"), start=1)
        }

    def get_frame(self, frame_id: str) -> FrameRecord:
        return self.frames[frame_id]

    def get_evidence(self, frame_id: str, source: RetrievalSource) -> None:
        del frame_id, source
        return None


class _TemporalCore:
    """Return two deterministic scenes without invoking the retriever."""

    def __init__(self, scenes: tuple[SceneCandidate, ...]) -> None:
        self.scenes = scenes

    def localize(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(
            search_id="search-1",
            version=1,
            scenes=self.scenes,
            diff=SimpleNamespace(mode=SimpleNamespace(value="new")),
            warnings=(),
            diagnostics={
                "candidate_pool_size": 50,
                "top_m_evidence": 5,
                "scene_top_p_global": 30,
            },
            trace=RetrievalTrace(),
            time_to_first_candidate_ms=None,
        )


class _Reranker:
    """Return the second candidate first and record the bounded request."""

    adapter = SimpleNamespace()

    def __init__(self, *, error: Exception | None = None) -> None:
        self.config = RerankerConfig(required=False)
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, candidates: list[Any]) -> list[Any]:
        self.calls.append((query, [item.frame_id for item in candidates]))
        if self.error is not None:
            raise self.error
        return [
            candidates[1].model_copy(
                update={"reranker_score": 0.95, "final_score": 0.95}
            ),
            candidates[0].model_copy(
                update={"reranker_score": 0.05, "final_score": 0.05}
            ),
        ]


def _scene(data: _Data, frame_id: str, score: float) -> SceneCandidate:
    frame = data.get_frame(frame_id)
    return SceneCandidate(
        scene_id=f"scene-{frame_id}",
        video_id=frame.video_id,
        start_ms=frame.timestamp_ms,
        end_ms=frame.timestamp_ms,
        evidence=(
            FrameEvidence(
                frame=frame,
                source_scores={RetrievalSource.VISUAL: score},
                score=score,
            ),
        ),
        final_score=score,
    )


def _pipeline(
    data: _Data,
    reranker: _Reranker,
    *,
    config: SearchConfig | None = None,
) -> KISPipeline:
    scenes = (
        _scene(data, "frame-a", 0.8),
        _scene(data, "frame-b", 0.7),
    )
    return KISPipeline(
        TaskType.KIS,
        cast(DataService, data),
        cast(RetrievalService, object()),
        config or SearchConfig(candidate_count=2),
        temporal_core=cast(Any, _TemporalCore(scenes)),
        reranking=cast(Any, reranker),
    )


def test_default_temporal_kis_calls_reranker_and_materializes_its_order() -> None:
    """The default positive rerank budget must affect the public result order."""

    data = _Data()
    reranker = _Reranker()

    response = _pipeline(data, reranker).execute(
        SearchRequest(query="red bus", top_k=2)
    )

    assert reranker.calls == [("red bus", ["frame-a", "frame-b"])]
    assert response.results[0].frame_ids[0] == "frame-b"
    assert response.results[0].scores.reranker == 0.95
    assert response.trace.stages["rerank"].status is StageStatus.SUCCESS
    assert response.latency_ms.reranking >= 0


def test_rerank_count_zero_skips_configured_reranker() -> None:
    """An explicit zero budget remains an opt-out for ablations and tests."""

    data = _Data()
    reranker = _Reranker()
    pipeline = _pipeline(
        data,
        reranker,
        config=SearchConfig(candidate_count=2, rerank_count=0),
    )

    response = pipeline.execute(SearchRequest(query="red bus", top_k=2))

    assert reranker.calls == []
    assert [item.frame_ids[0] for item in response.results] == [
        "frame-a",
        "frame-b",
    ]
    assert response.trace.stages["rerank"].status is StageStatus.SKIPPED


def test_search_service_injects_reranker_into_kis_registry() -> None:
    """The composition root must pass the configured service to both KIS heads."""

    data = _Data()
    reranker = _Reranker()
    service = SearchService(
        cast(DataService, data),
        cast(RetrievalService, object()),
        reranking=cast(Any, reranker),
    )

    kis = service.pipeline_registry.get(TaskType.KIS)
    vkis = service.pipeline_registry.get(TaskType.VKIS)

    assert getattr(kis, "reranking") is reranker
    assert getattr(vkis, "reranking") is reranker


def test_optional_reranker_failure_falls_back_to_temporal_order() -> None:
    """A bounded provider failure keeps search available when not required."""

    data = _Data()
    reranker = _Reranker(error=RerankerUnavailableError("unavailable"))

    response = _pipeline(data, reranker).execute(
        SearchRequest(query="red bus", top_k=2)
    )

    assert [item.frame_ids[0] for item in response.results] == [
        "frame-a",
        "frame-b",
    ]
    assert response.trace.stages["rerank"].status is StageStatus.PARTIAL
    assert response.trace.stages["rerank"].fallback_used is True
    assert response.warnings == ["reranking fallback (unavailable)"]
