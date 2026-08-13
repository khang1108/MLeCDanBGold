"""Smoke tests for four-source weighted reciprocal-rank fusion."""

from hcmai.common.config import FusionConfig
from hcmai.common.schemas import (
    RetrievalResult,
    RetrievalSource,
    RetrievalTrace,
    StageStatus,
    StageTrace,
    TaskType,
)
from hcmai.common.schemas.retrieval import RetrievalCandidate
from hcmai.common.schemas.search import SearchFilters
from hcmai.retrieval.retriever.fusion import RRFFusionRetriever


class FakeRetriever:
    def __init__(self, candidates, *, encode_ms: float, search_ms: float) -> None:
        self.candidates = candidates
        self.trace = RetrievalTrace(
            stages={
                "query_encoding": _trace("query_encoding", encode_ms),
                "index_search": _trace("index_search", search_ms),
            }
        )
        self.call = None

    def search(self, query, top_k, filters, query_type=TaskType.KIS):
        self.call = (query, top_k, filters, query_type)
        return RetrievalResult(
            candidates=self.candidates[:top_k],
            trace=self.trace,
        )


def _trace(stage: str, duration_ms: float) -> StageTrace:
    return StageTrace(
        stage=stage,
        started_at=1.0,
        ended_at=1.0 + duration_ms / 1_000,
        duration_ms=duration_ms,
        status=StageStatus.SUCCESS,
    )


def _candidate(frame_id, source, rank, score):
    return RetrievalCandidate(
        frame_id=frame_id,
        source_scores={source: score},
        source_ranks={source: rank},
        metadata={"frame": {"frame_id": frame_id}},
    )


def test_rrf_unions_disjoint_frames_and_rewards_source_agreement() -> None:
    visual = FakeRetriever(
        [
            _candidate("shared", RetrievalSource.VISUAL, 1, 0.9),
            _candidate("visual-only", RetrievalSource.VISUAL, 2, 0.8),
        ],
        encode_ms=2.0,
        search_ms=3.0,
    )
    caption = FakeRetriever(
        [
            _candidate("caption-only", RetrievalSource.CAPTION, 1, 0.95),
            _candidate("shared", RetrievalSource.CAPTION, 2, 0.85),
        ],
        encode_ms=5.0,
        search_ms=7.0,
    )
    filters = SearchFilters(video_ids=["video-1"])
    config = FusionConfig(method="rrf", rrf_k=60)
    retriever = RRFFusionRetriever([visual, caption], config)

    results = retriever.search("cook", top_k=3, filters=filters)

    assert [item.frame_id for item in results] == [
        "shared", "caption-only", "visual-only",
    ]
    assert results[0].source_ranks == {
        RetrievalSource.VISUAL: 1,
        RetrievalSource.CAPTION: 2,
    }
    assert results[0].fusion_score == 1 / 61 + 1 / 62
    assert visual.call == caption.call == ("cook", 3, filters, TaskType.KIS)
    assert results.trace.duration_for("query_encoding") == 7.0
    assert results.trace.duration_for("index_search") == 10.0
    assert retriever.config is config


def test_rrf_uses_task_specific_weights_across_all_modalities() -> None:
    retrievers = [
        FakeRetriever(
            [_candidate("visual", RetrievalSource.VISUAL, 1, 0.9)],
            encode_ms=0,
            search_ms=0,
        ),
        *[
            FakeRetriever(
                [_candidate("text-shared", source, 1, 0.9)],
                encode_ms=0,
                search_ms=0,
            )
            for source in (
                RetrievalSource.CAPTION,
                RetrievalSource.OCR,
                RetrievalSource.ASR,
            )
        ],
    ]
    task_weights = FusionConfig().task_weights
    task_weights[TaskType.VQA] = {
        source: (5.0 if source == RetrievalSource.VISUAL else 1.0)
        for source in RetrievalSource
    }
    fusion = RRFFusionRetriever(retrievers, FusionConfig(task_weights=task_weights))
    kis = fusion.search("query", top_k=2, query_type=TaskType.KIS)
    vqa = fusion.search("query", top_k=2, query_type=TaskType.VQA)

    assert kis[0].frame_id == "text-shared"
    assert vqa[0].frame_id == "visual"
    assert vqa[0].fusion_score == 5 / 61
