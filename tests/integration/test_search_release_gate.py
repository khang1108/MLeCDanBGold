from concurrent.futures import ThreadPoolExecutor
from typing import cast

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    FrameRecord,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    SearchRequest,
)
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService


class TinyData:
    record_count = 1

    def get_frame(self, frame_id):
        assert frame_id == "frame-1"
        return FrameRecord(
            frame_id="frame-1", video_id="video-1", frame_idx=7,
            timestamp_ms=1_000, image_path="unused.jpg", width=4, height=4,
        )

    def get_evidence(self, frame_id, source):
        return None


class TinyRetrieval:
    active_sources = (RetrievalSource.VISUAL,)

    def search(self, query, top_k, filters, query_type):
        return RetrievalResult(candidates=[RetrievalCandidate(
            frame_id="frame-1",
            source_scores={RetrievalSource.VISUAL: 1.0},
            source_ranks={RetrievalSource.VISUAL: 1},
            final_score=1.0,
        )])


def _service():
    return SearchService(
        cast(DataService, TinyData()),
        cast(RetrievalService, TinyRetrieval()),
        config=SearchConfig(candidate_count=1, rerank_count=0),
    )


def test_tiny_corpus_survives_100_repeated_requests():
    service = _service()
    for index in range(100):
        response = service.search(SearchRequest(query=f"query {index}", top_k=1))
        assert response.results[0].frame_idx == 7
        assert response.trace.stages


def test_tiny_corpus_survives_20_concurrent_requests_with_independent_traces():
    service = _service()
    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(
            lambda index: service.search(
                SearchRequest(query=f"concurrent {index}", top_k=1)
            ),
            range(20),
        ))

    assert len({response.request_id for response in responses}) == 20
    assert all(response.results[0].frame_idx == 7 for response in responses)
    assert all(response.trace.stages for response in responses)
