from concurrent.futures import ThreadPoolExecutor
from typing import cast

import numpy as np

from hcmai.api.contracts import SearchRequest
from hcmai.common.config import SearchConfig
from hcmai.corpus import Corpus, Frame
from hcmai.retrieval.models import RetrievalSource
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class TinyCorpus:
    def __len__(self):
        return 1

    def frame(self, frame_id):
        assert frame_id == "frame-1"
        return Frame(
            frame_id="frame-1", video_id="video-1", frame_idx=7,
            timestamp_ms=1_000, image_path="unused.jpg",
        )

    def caption(self, frame_id):
        del frame_id
        return None
    ocr = caption

    def objects(self, frame_id):
        del frame_id
        return ()

    def title(self, video_id):
        del video_id
        return None
    def transcript(self, video_id, start_ms, end_ms):
        del video_id, start_ms, end_ms
        return None


class TinyRetrieval:
    active_sources = (RetrievalSource.VISUAL,)

    def score_event_videos(self, events, **kwargs):
        """Return one immutable score matrix for every concurrent request."""

        del kwargs
        return [
            VideoEventScores(
                video_id="video-1",
                frame_ids=np.array(["frame-1"], dtype=object),
                frame_idx=np.array([7]),
                timestamps_ms=np.array([1_000]),
                scores=np.ones((len(events), 1)),
            )
        ]


def _service():
    return SearchService(
        cast(Corpus, TinyCorpus()),
        cast(RetrievalService, TinyRetrieval()),
        config=SearchConfig(),
    )


def test_tiny_corpus_survives_100_repeated_requests():
    service = _service()
    for index in range(100):
        response = service.search_kis(
            SearchRequest(query=f"query {index}", top_k=1)
        )
        assert response.results[0].frame_idx == 7
        assert response.latency.total_ms >= 0


def test_tiny_corpus_survives_20_concurrent_requests_with_independent_traces():
    service = _service()
    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(
            lambda index: service.search_kis(
                SearchRequest(query=f"concurrent {index}", top_k=1)
            ),
            range(20),
        ))

    assert all(response.results[0].frame_idx == 7 for response in responses)
    assert len({id(response.latency) for response in responses}) == 20
