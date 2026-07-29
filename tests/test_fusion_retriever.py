"""Smoke tests for visual-caption reciprocal rank fusion."""

from __future__ import annotations

from hcmai.common.config import FusionConfig
from hcmai.common.schemas import RetrievalSource
from hcmai.common.schemas.retrieval import RetrievalCandidate
from hcmai.common.schemas.search import SearchFilters
from hcmai.retriever.fusion import RRFFusionRetriever


class FakeRetriever:
    def __init__(self, candidates, *, encode_ms: float, search_ms: float) -> None:
        self.candidates = candidates
        self.last_query_encoding_ms = encode_ms
        self.last_index_search_ms = search_ms
        self.call = None

    def search(self, query, top_k, filters):
        self.call = (query, top_k, filters)
        return self.candidates[:top_k]


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
        "shared", "caption-only", "visual-only"
    ]
    assert results[0].source_ranks == {
        RetrievalSource.VISUAL: 1,
        RetrievalSource.CAPTION: 2,
    }
    assert results[0].fusion_score == 1 / 61 + 1 / 62
    assert visual.call == caption.call == ("cook", 3, filters)
    assert retriever.last_query_encoding_ms == 7.0
    assert retriever.last_index_search_ms == 10.0
    assert retriever.config is config
