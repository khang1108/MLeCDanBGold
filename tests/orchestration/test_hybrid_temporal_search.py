"""Tests for routing fused evidence into frozen temporal DP."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from hcmai.common.config import AlignmentConfig
from hcmai.orchestration.workflows.temporal_search import TemporalSearchService
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class FakeCorpus:
    @staticmethod
    def frame(frame_id: str) -> Any:
        return SimpleNamespace(frame_id=frame_id, video_id="v1", frame_idx=1, timestamp_ms=100)


class RecordingScorer:
    def __init__(self) -> None:
        self.call: tuple[object, ...] | None = None

    def score_events(
        self, original: object, retrieval: object, **kwargs: object
    ) -> list[VideoEventScores]:
        self.call = (original, retrieval, kwargs)
        return [
            VideoEventScores(
                video_id="v1",
                frame_ids=np.asarray(["f1"]),
                frame_idx=np.asarray([1]),
                timestamps_ms=np.asarray([100]),
                scores=np.asarray([[0.7]], dtype=np.float32),
        )]


def test_temporal_search_routes_representations_to_evidence_scorer(monkeypatch: Any) -> None:
    """Forward fused score rows into DP without query rewriting."""

    scorer = RecordingScorer()
    received: list[VideoEventScores] = []

    def fake_rank_paths(scores: list[VideoEventScores], **kwargs: object) -> list[object]:
        received.extend(scores)
        return []

    monkeypatch.setattr(
        "hcmai.orchestration.workflows.temporal_search.rank_paths", fake_rank_paths
    )
    service = TemporalSearchService(cast(Any, FakeCorpus()), scorer, AlignmentConfig())

    result = service.search(
        ("vi",),
        retrieval_events=("en",),
        caption_events=("caption",),
        use_dense=True,
        use_bm25=True,
        top_k=3,
    )

    assert scorer.call == (
        ("vi",),
        ("en",),
        {"caption_events": ("caption",), "use_dense": True, "use_bm25": True},
    )
    assert received[0].scores[0, 0] == np.float32(0.7)
    assert result.paths == ()


def test_temporal_event_limit_runs_before_evidence_scoring() -> None:
    """Reject oversized event matrices before either encoder can run."""

    scorer = RecordingScorer()
    service = TemporalSearchService(
        cast(Any, FakeCorpus()), scorer, AlignmentConfig(), max_temporal_event_count=1
    )

    with pytest.raises(ValueError, match="at most 1 temporal events"):
        service.search(("first", "second"), top_k=1)

    assert scorer.call is None
