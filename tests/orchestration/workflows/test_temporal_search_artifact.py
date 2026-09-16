"""Tests for TemporalSearchArtifact and score-preserving search."""

from unittest.mock import Mock
import numpy as np
import pytest

from hcmai.common.config import AlignmentConfig
from hcmai.corpus.models import Frame
from hcmai.orchestration.workflows.temporal_search import TemporalSearchService
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from hcmai.retrieval.retriever.video_scores import VideoEventScores


@pytest.fixture
def corpus():
    c = Mock()
    frames = {
        "v1_f1": Frame(
            video_id="v1",
            frame_id="v1_f1",
            frame_idx=10,
            timestamp_ms=1000,
            image_path="/tmp/v1_f1.jpg",
            fps=25.0,
        ),
        "v1_f2": Frame(
            video_id="v1",
            frame_id="v1_f2",
            frame_idx=20,
            timestamp_ms=2000,
            image_path="/tmp/v1_f2.jpg",
            fps=25.0,
        ),
    }
    c.frame.side_effect = lambda fid: frames[fid]
    return c


@pytest.fixture
def evidence():
    ev = Mock()
    scores = (
        VideoEventScores(
            video_id="v1",
            frame_ids=np.array(["v1_f1", "v1_f2"]),
            frame_idx=np.array([10, 20]),
            timestamps_ms=np.array([1000, 2000], dtype=np.int64),
            scores=np.array([[0.9, 0.8]], dtype=np.float32),
        ),
    )
    ev.score_plan.return_value = scores
    return ev


@pytest.fixture
def plan():
    return KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "a woman enters", "a woman enters", "a woman enters"),
        )
    )


@pytest.fixture
def temporal_service(corpus, evidence):
    config = AlignmentConfig()
    return TemporalSearchService(corpus, evidence, config)


def test_search_plan_artifact_scores_once_and_ranks_same_scores(
    temporal_service, plan, monkeypatch
):
    calls = 0
    original = temporal_service.score_plan

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(temporal_service, "score_plan", counted)
    artifact = temporal_service.search_plan_artifact(plan, top_k=5)

    assert calls == 1
    assert artifact.result.paths
    assert artifact.video_scores
    assert artifact.decoder_config == temporal_service.snapshot_decoder_config()

    scores_by_id = {v.video_id: v for v in artifact.video_scores}
    for path in artifact.result.paths:
        assert path.video_id in scores_by_id
        video = scores_by_id[path.video_id]
        known_frames = set(video.frame_ids)
        for fid in path.frame_ids:
            assert fid in known_frames


def test_score_plan_validation_returns_video_event_scores(temporal_service, plan):
    scores, retrieval_ms = temporal_service.score_plan(plan)
    assert len(scores) == 1
    assert isinstance(scores[0], VideoEventScores)
    assert scores[0].video_id == "v1"
    assert retrieval_ms >= 0.0
