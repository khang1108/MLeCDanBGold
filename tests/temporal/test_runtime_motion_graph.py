"""Tests for parallel runtime integration and minimal experiment config (SP-09, SP-10).

Verifies Phase 6:
- Task 6.1: search_plan remains static baseline; search_plan_motion_graph provides parallel path.
- Task 6.2: AlignmentConfig supports decoder: static | motion_graph, candidate_k, transition_weight.
"""

from unittest.mock import Mock
import numpy as np
import pytest

from hcmai.common.config import AlignmentConfig
from hcmai.corpus.models import Frame
from hcmai.orchestration.workflows.search.temporal import (
    DecoderConfigSnapshot,
    TemporalSearchService,
)
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.transition_decoder import (
    FrameEmbeddingAccessor,
    rank_motion_graph_paths,
)


def test_alignment_config_minimal_schema():
    """Verify AlignmentConfig schema for static and motion_graph decoders (Task 6.2)."""
    # 1. Default is static baseline
    default_cfg = AlignmentConfig()
    assert default_cfg.decoder == "static"
    assert default_cfg.candidate_k == 32
    assert default_cfg.transition_weight == 0.25

    # 2. Configured for motion graph
    mg_cfg = AlignmentConfig(
        decoder="motion_graph",
        candidate_k=16,
        transition_weight=0.5,
    )
    assert mg_cfg.decoder == "motion_graph"
    assert mg_cfg.candidate_k == 16
    assert mg_cfg.transition_weight == 0.5

    # 3. Invalid inputs rejected
    with pytest.raises(ValueError):
        AlignmentConfig(candidate_k=0)

    with pytest.raises(ValueError):
        AlignmentConfig(transition_weight=-0.1)


def test_rank_motion_graph_paths_diversification():
    """Verify rank_motion_graph_paths ranks multi-video candidates with level-wise diversification."""
    v1_scores = np.array([
        [0.8, 0.2],
        [0.2, 0.8],
    ])
    v2_scores = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
    ])

    v1 = VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["v1_0", "v1_1"]),
        frame_idx=np.array([0, 1], dtype=np.int64),
        timestamps_ms=np.array([1000, 2000], dtype=np.int64),
        scores=v1_scores,
    )
    v2 = VideoEventScores(
        video_id="v2",
        frame_ids=np.array(["v2_0", "v2_1"]),
        frame_idx=np.array([0, 1], dtype=np.int64),
        timestamps_ms=np.array([1000, 2000], dtype=np.int64),
        scores=v2_scores,
    )

    frame_embs = {
        "v1": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "v2": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }
    event_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    rows = rank_motion_graph_paths(
        [v1, v2],
        frame_embeddings=frame_embs,
        event_embeddings=event_embs,
        candidate_k=2,
        transition_weight=0.0,
        max_rows=2,
    )

    assert len(rows) == 2
    # v2 has higher score (0.9+0.9 = 1.8) than v1 (0.8+0.8 = 1.6)
    assert rows[0].video_id == "v2"
    assert rows[1].video_id == "v1"


def test_temporal_search_service_parallel_paths(monkeypatch):
    """Verify TemporalSearchService provides search_plan_motion_graph alongside search_plan_artifact (Task 6.1)."""
    corpus = Mock()
    frames = {
        "v1_f0": Frame(frame_id="v1_f0", video_id="v1", frame_idx=0, timestamp_ms=1000, image_path="/tmp/0.jpg"),
        "v1_f1": Frame(frame_id="v1_f1", video_id="v1", frame_idx=1, timestamp_ms=2000, image_path="/tmp/1.jpg"),
    }
    corpus.frame.side_effect = lambda fid: frames[fid]

    evidence = Mock()
    scores = (
        VideoEventScores(
            video_id="v1",
            frame_ids=np.array(["v1_f0", "v1_f1"]),
            frame_idx=np.array([0, 1]),
            timestamps_ms=np.array([1000, 2000], dtype=np.int64),
            scores=np.array([[0.8, 0.1], [0.1, 0.8]], dtype=np.float32),
        ),
    )
    evidence.score_plan.return_value = scores
    evidence.visual_index = {"v1": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}

    plan = KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "start", "start", "start"),
            KISRetrievalEvent("E2", "finish", "finish", "finish"),
        )
    )

    # 1. Baseline static decoder
    static_service = TemporalSearchService(corpus, evidence, AlignmentConfig(decoder="static"))
    static_art = static_service.search_plan_artifact(plan, top_k=1)
    assert static_art.decoder_config.decoder == "static"
    assert len(static_art.result.paths) == 1

    # 2. Parallel explicit motion graph call
    event_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mg_art = static_service.search_plan_motion_graph(
        plan,
        event_embeddings=event_embs,
        candidate_k=2,
        transition_weight=0.25,
        top_k=1,
    )
    assert mg_art.decoder_config.decoder == "motion_graph"
    assert mg_art.decoder_config.candidate_k == 2
    assert mg_art.decoder_config.transition_weight == 0.25
    assert len(mg_art.result.paths) == 1

    # 3. Service configured with decoder: motion_graph routes automatically
    monkeypatch.setattr(
        "hcmai.orchestration.workflows.search.temporal.encode_query_events",
        lambda texts: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    mg_service = TemporalSearchService(corpus, evidence, AlignmentConfig(decoder="motion_graph", candidate_k=2))
    auto_art = mg_service.search_plan_artifact(plan, top_k=1)
    assert auto_art.decoder_config.decoder == "motion_graph"
    assert len(auto_art.result.paths) == 1
