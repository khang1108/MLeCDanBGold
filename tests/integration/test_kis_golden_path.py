"""Golden-path coverage for stateless KIS temporal alignment."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from hcmai.common.schemas import FrameRecord, SearchRequest
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class CanonicalData:
    """Small canonical frame authority for a three-event KIS path."""

    frames = {
        frame_id: FrameRecord(
            frame_id=frame_id,
            video_id="video-a",
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            image_path=f"{frame_id}.jpg",
            width=640,
            height=360,
        )
        for frame_id, frame_idx, timestamp_ms in (
            ("a1", 10, 1_000),
            ("a2", 11, 1_300),
            ("a3", 40, 8_000),
        )
    }

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Resolve one frame only through the canonical fixture mapping."""

        return self.frames[frame_id]

    def get_evidence(self, frame_id, source):
        """Keep optional specialist evidence absent from this visual fixture."""

        del frame_id, source
        return None


class BatchRetrieval:
    """Return an exact visual score matrix for the explicit event list."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], object]] = []

    def score_event_videos(self, events, filters=None, **kwargs):
        """Expose one score row per event and retain task-provided filters."""

        del kwargs
        self.calls.append((list(events), filters))
        return [
            VideoEventScores(
                video_id="video-a",
                frame_ids=np.array(["a1", "a2", "a3"], dtype=object),
                frame_idx=np.array([10, 11, 40]),
                timestamps_ms=np.array([1_000, 1_300, 8_000]),
                scores=np.eye(len(events), 3, dtype=np.float32),
            )
        ]


def test_golden_kis_projects_middle_path_frame_and_preserves_identity() -> None:
    """Return KIS's deterministic midpoint without discarding the path IDs."""

    retrieval = BatchRetrieval()
    response = SearchService(
        cast(DataService, CanonicalData()),
        cast(RetrievalService, retrieval),
    ).search(
        SearchRequest(
            query="first. second. third.",
            events=["first", "second", "third"],
            top_k=4,
        )
    )

    assert response.results[0].frame_id == "a2"
    assert response.results[0].frame_ids == ["a1", "a2", "a3"]
    assert response.results[0].video_id == "video-a"
    assert response.results[0].frame_idx == 11
    assert response.results[0].scores.final == pytest.approx(2.93)
    assert retrieval.calls == [(["first", "second", "third"], None)]


class ManyFramesData:
    """Canonical data fixture with one valid single-event path per video."""

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Materialize the integer suffix as one video's canonical frame."""

        index = int(frame_id.removeprefix("frame-"))
        return FrameRecord(
            frame_id=frame_id,
            video_id=f"video-{index}",
            frame_idx=index,
            timestamp_ms=index * 10_000,
            image_path=f"{frame_id}.jpg",
            width=640,
            height=360,
        )

    def get_evidence(self, frame_id, source):
        """Avoid specialist artifact requirements for the top-k fixture."""

        del frame_id, source
        return None


class ManyFramesRetrieval:
    """Produce 100 independently rankable one-event video candidates."""

    def score_event_videos(self, events, filters=None, **kwargs):
        """Keep fixture event count honest while returning deterministic paths."""

        del filters, kwargs
        assert len(events) == 1
        return [
            VideoEventScores(
                video_id=f"video-{index}",
                frame_ids=np.array([f"frame-{index}"], dtype=object),
                frame_idx=np.array([index]),
                timestamps_ms=np.array([index * 10_000]),
                scores=np.array([[1.0 - index / 1_000]], dtype=np.float32),
            )
            for index in range(100)
        ]


def test_kis_top_k_controls_materialized_path_count() -> None:
    """Keep the public KIS result budget independent of candidate video count."""

    service = SearchService(
        cast(DataService, ManyFramesData()),
        cast(RetrievalService, ManyFramesRetrieval()),
    )

    response_20 = service.search(SearchRequest(query="many frames", top_k=20))
    response_100 = service.search(SearchRequest(query="many frames", top_k=100))

    assert response_20.total_results == 20
    assert len(response_20.results) == 20
    assert response_100.total_results == 100
    assert len(response_100.results) == 100
