from types import SimpleNamespace
from typing import cast

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    QueryLanguage,
    QuerySuggestion,
    RetrievalCandidate,
    RetrievalResult,
    SearchRequest,
    TaskType,
)
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipelines.kis import KISPipeline
from hcmai.retriever.pipeline import RetrievalService


class CanonicalData:
    frames = {
        "a1": ("video-a", 10, 1_000),
        "a2": ("video-a", 11, 1_300),
        "a3": ("video-a", 40, 8_000),
        "b1": ("video-b", 20, 2_000),
        "c1": ("video-c", 30, 3_000),
    }

    def get_frame(self, frame_id):
        video_id, frame_idx, timestamp_ms = self.frames[frame_id]
        return SimpleNamespace(
            frame_id=frame_id,
            video_id=video_id,
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
        )

    def get_evidence(self, frame_id, source):
        del frame_id, source
        return None


class BatchRetrieval:
    def __init__(self):
        self.calls = []

    def search_batch(self, queries, top_k, filters, query_type):
        self.calls.append((queries, top_k, filters, query_type))
        original = ["a1", "a2", "a3", "b1", "c1"]
        generated = ["b1", "c1", "a1"]
        return [
            RetrievalResult(candidates=[
                RetrievalCandidate(frame_id=frame_id)
                for frame_id in (original if index == 0 else generated)
            ])
            for index in range(len(queries))
        ]


class Suggestions:
    def suggest(self, request):
        return SimpleNamespace(suggestions=[
            QuerySuggestion(
                suggestion_id=f"s{index}",
                query=f"red bus 7 view {index}",
                language=QueryLanguage.ENGLISH,
                focus="literal",
            )
            for index in range(request.count)
        ])


def test_golden_kis_path_batches_variants_and_preserves_canonical_identity():
    retrieval = BatchRetrieval()
    pipeline = KISPipeline(
        TaskType.KIS,
        cast(DataService, CanonicalData()),
        cast(RetrievalService, retrieval),
        None,
        SearchConfig(candidate_count=10, rerank_count=0, temporal_window_ms=500),
        suggestion_service=Suggestions(),
    )

    first = pipeline.execute(SearchRequest(query="red bus 7", top_k=4))
    second = pipeline.execute(SearchRequest(query="red bus 7", top_k=4))

    assert [item.frame_id for item in first.results] == ["a1", "b1", "c1", "a3"]
    assert [(item.video_id, item.frame_idx) for item in first.results] == [
        ("video-a", 10),
        ("video-b", 20),
        ("video-c", 30),
        ("video-a", 40),
    ]
    assert [item.frame_id for item in first.results] == [
        item.frame_id for item in second.results
    ]
    assert len(retrieval.calls) == 2
    assert len(retrieval.calls[0][0]) == 6
