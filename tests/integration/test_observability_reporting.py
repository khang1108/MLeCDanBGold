from __future__ import annotations

from typing import cast

import numpy as np

from hcmai.common.schemas import (
    RetrievalSource,
    FrameRecord,
    SearchRequest,
)
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class Data:
    record_count = 1

    def get_frame(self, frame_id: str):
        return FrameRecord(
            frame_id=frame_id,
            video_id="video-1",
            frame_idx=17,
            timestamp_ms=680,
            image_path=f"{frame_id}.jpg",
            width=640,
            height=360,
        )

    def get_evidence(self, frame_id, source):
        del frame_id, source
        return None

    def has_evidence(self, source):
        del source
        return False


class Retrieval:
    active_sources = (RetrievalSource.VISUAL,)

    def score_event_videos(self, events, filters=None, **kwargs):
        """Return one canonical score column for stateless KIS telemetry."""

        del filters, kwargs
        return [
            VideoEventScores(
                video_id="video-1",
                frame_ids=np.array(["frame-1"], dtype=object),
                frame_idx=np.array([17]),
                timestamps_ms=np.array([680]),
                scores=np.full((len(events), 1), 0.9),
            )
        ]


def test_search_response_and_health_expose_canonical_observability() -> None:
    service = SearchService(
        cast(DataService, Data()),
        cast(RetrievalService, Retrieval()),
    )

    response = service.search(SearchRequest(query="person cooking", top_k=1))
    health = service.health()

    assert set(response.trace.stages) == {
        "parse",
        "localization",
        "materialization",
    }
    assert response.trace.stages["localization"].backend == "monotonic_dp"
    assert response.latency_ms.time_to_first_candidate >= 0
    assert response.latency_ms.time_to_first_submission >= 0
    assert health["capabilities"]["kis"] is True
    assert health["capabilities"]["trake"] is True
    assert health["capabilities"]["query_types"] == {
        "kis": True,
        "trake": True,
    }
    assert health["capabilities"]["shared_retrieval"] is True
    assert set(health["capabilities"]["remote_inference"]) == {
        "embedding",
        "reranking",
        "structured_parsing",
    }
    assert "latency_histograms_ms" in health["observability"]
