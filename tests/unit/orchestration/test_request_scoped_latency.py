from __future__ import annotations

from typing import cast

import numpy as np

from hcmai.common.schemas import FrameRecord, SearchRequest
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class CanonicalData:
    def get_frame(self, frame_id: str):
        return FrameRecord(
            frame_id=frame_id,
            video_id="video-1",
            frame_idx=42,
            timestamp_ms=1_000,
            image_path=f"{frame_id}.jpg",
            width=640,
            height=360,
        )

    def get_evidence(self, frame_id, source):
        del frame_id, source
        return None


class FullyTracedRetrieval:
    def score_event_videos(self, events, filters=None, **kwargs):
        """Return one canonical frame for the new single alignment stage."""

        del filters, kwargs
        return [
            VideoEventScores(
                video_id="video-1",
                frame_ids=np.array(["frame-1"], dtype=object),
                frame_idx=np.array([42]),
                timestamps_ms=np.array([1_000]),
                scores=np.full((len(events), 1), 0.9),
            )
        ]


def test_kis_response_latency_records_one_alignment_stage() -> None:
    service = SearchService(
        cast(DataService, CanonicalData()),
        cast(RetrievalService, FullyTracedRetrieval()),
    )

    response = service.search(SearchRequest(query="red bus", top_k=1))

    assert response.trace.stages["localization"].backend == "monotonic_dp"
    assert response.latency_ms.temporal_refinement >= 0
    assert response.latency_ms.reranking == 0
