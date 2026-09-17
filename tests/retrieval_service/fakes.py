"""Shared fakes for retrieval service tests."""

from unittest.mock import Mock
import numpy as np

from hcmai.corpus.models import Frame
from hcmai.orchestration.workflows.temporal_search import (
    DecoderConfigSnapshot,
    SelectedVideoScoreResult,
    TemporalSearchArtifact,
    TemporalSearchResult,
)
from hcmai.retrieval.evidence.components import TemporalScoreComponent
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import AlignedPath


def make_fake_corpus(frames: dict[str, Frame] | None = None) -> Mock:
    """Create a minimal corpus fake with canonical frame records."""
    if frames is None:
        frames = {
            "v1_f1": Frame("v1", "v1_f1", 10, 1000, "/tmp/v1_f1.jpg", 25.0),
            "v1_f2": Frame("v1", "v1_f2", 20, 2000, "/tmp/v1_f2.jpg", 25.0),
            "v2_f1": Frame("v2", "v2_f1", 10, 1000, "/tmp/v2_f1.jpg", 25.0),
            "v2_f2": Frame("v2", "v2_f2", 20, 2000, "/tmp/v2_f2.jpg", 25.0),
        }
    corpus = Mock()
    corpus.frame.side_effect = lambda fid: frames[fid]
    corpus.__len__ = Mock(return_value=len(frames))
    return corpus


def make_fake_video_scores(video_id: str, frame_ids: list[str]) -> VideoEventScores:
    """Create a two-frame VideoEventScores fixture."""
    count = len(frame_ids)
    return VideoEventScores(
        video_id=video_id,
        frame_ids=np.array(frame_ids),
        frame_idx=np.array([10 * (i + 1) for i in range(count)], dtype=np.int64),
        timestamps_ms=np.array([1000 * (i + 1) for i in range(count)], dtype=np.int64),
        scores=np.array([[0.8, 0.9]], dtype=np.float32),
    )


def make_fake_temporal_search_service() -> Mock:
    """Create a fake TemporalSearchService with Mock tracking."""
    temporal = Mock()
    decoder_config = DecoderConfigSnapshot(
        lambda_gap=0.1,
        event_power=1.0,
        cluster_delta=0.5,
        path_min_separation_ms=1000,
    )
    v1_scores = make_fake_video_scores("video-1", ["v1_f1", "v1_f2"])
    v2_scores = make_fake_video_scores("video-2", ["v2_f1", "v2_f2"])

    path1 = AlignedPath("video-1", 0.95, ("v1_f1",), (10,), (1000,))
    path2 = AlignedPath("video-2", 0.90, ("v2_f1",), (10,), (1000,))

    search_result = TemporalSearchResult(
        paths=(path1, path2),
        retrieval_ms=10.0,
        alignment_ms=2.0,
    )
    artifact = TemporalSearchArtifact(
        result=search_result,
        video_scores=(v1_scores, v2_scores),
        decoder_config=decoder_config,
    )

    temporal.search_plan_artifact.return_value = artifact
    temporal.search.return_value = search_result
    temporal.snapshot_decoder_config.return_value = decoder_config

    def fake_score_video(plan, *, video_id, **kwargs):
        if video_id == "video-1":
            return SelectedVideoScoreResult(v1_scores, 5.0, decoder_config)
        elif video_id == "video-2":
            return SelectedVideoScoreResult(v2_scores, 5.0, decoder_config)
        raise KeyError(f"video {video_id!r} not found")

    temporal.score_video.side_effect = fake_score_video
    return temporal


def make_fake_image_scorer() -> Mock:
    """Create a fake ImageQueryTemporalScorer."""
    scorer = Mock()
    scorer.score_events.return_value = TemporalScoreComponent(
        name="visual_image",
        raw_scores=np.array([[0.7, 0.8]], dtype=np.float32),
    )
    return scorer
