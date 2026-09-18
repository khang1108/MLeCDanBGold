"""Tests for event-conditioned complete-path dynamic programming."""

import itertools
import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video_conditioned


def brute_force(video, focus_event_index, focus_position, allowed, lambda_gap=0.0):
    best = None
    n_events, n_frames = video.scores.shape
    timestamps = np.asarray(video.timestamps_ms, dtype=np.float64)
    for positions in itertools.combinations(range(n_frames), n_events):
        if positions[focus_event_index] != focus_position:
            continue
        if any(not allowed[e, p] for e, p in enumerate(positions)):
            continue
        score = sum(float(video.scores[e, p]) for e, p in enumerate(positions))
        if lambda_gap > 0 and n_events > 1:
            score -= lambda_gap * (timestamps[positions[-1]] - timestamps[positions[0]])
        if best is None or score > best[0]:
            best = (score, positions)
    return best


@pytest.fixture
def video_scores_fixture():
    scores = np.array(
        [
            [0.9, 0.2, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.8, 0.3, 0.9, 0.2, 0.1],
            [0.1, 0.1, 0.7, 0.2, 0.8, 0.9],
        ],
        dtype=np.float32,
    )
    return VideoEventScores(
        video_id="v_test",
        frame_ids=np.array([f"f_{i}" for i in range(6)]),
        frame_idx=np.array([i * 10 for i in range(6)]),
        timestamps_ms=np.array([i * 1000 for i in range(6)], dtype=np.int64),
        scores=scores,
    )


def test_conditioned_paths_match_bruteforce_for_focus_event(video_scores_fixture):
    video = video_scores_fixture
    allowed = np.ones_like(video.scores, dtype=bool)
    paths = align_video_conditioned(
        video,
        1,
        allowed=allowed,
        lambda_gap=0.0,
        event_power=1.0,
        cluster_delta=0.0,
        max_paths=4,
        min_separation_ms=0,
    )
    assert len(paths) > 0
    for item in paths:
        expected = brute_force(video, 1, item.focus_frame_position, allowed, lambda_gap=0.0)
        assert expected is not None
        assert abs(item.score - expected[0]) < 1e-6
        assert tuple(item.path.frame_idx) == tuple(int(video.frame_idx[p]) for p in expected[1])


def test_conditioned_paths_with_gap_penalty_matches_bruteforce(video_scores_fixture):
    video = video_scores_fixture
    allowed = np.ones_like(video.scores, dtype=bool)
    lambda_gap = 1e-4
    paths = align_video_conditioned(
        video,
        1,
        allowed=allowed,
        lambda_gap=lambda_gap,
        event_power=1.0,
        cluster_delta=0.0,
        max_paths=4,
        min_separation_ms=0,
    )
    assert len(paths) > 0
    for item in paths:
        expected = brute_force(video, 1, item.focus_frame_position, allowed, lambda_gap=lambda_gap)
        assert expected is not None
        assert abs(item.score - expected[0]) < 1e-6
        assert tuple(item.path.frame_idx) == tuple(int(video.frame_idx[p]) for p in expected[1])


def test_highest_unary_does_not_equal_highest_conditioned_path():
    """Verify conditioned alternatives rank by complete path score rather than unary frame score."""
    # At focus event 1:
    # Frame 1 has score 0.99, but event 0 and 2 have terrible scores around it.
    # Frame 3 has score 0.60, but event 0 and 2 have 0.95 and 0.95.
    scores = np.array(
        [
            [0.10, 0.10, 0.95, 0.10, 0.10],  # Event 0
            [0.10, 0.99, 0.10, 0.60, 0.10],  # Event 1 (Focus)
            [0.10, 0.10, 0.10, 0.10, 0.95],  # Event 2
        ],
        dtype=np.float32,
    )
    video = VideoEventScores(
        video_id="v_unary",
        frame_ids=np.array([f"f_{i}" for i in range(5)]),
        frame_idx=np.array([i * 10 for i in range(5)]),
        timestamps_ms=np.array([i * 1000 for i in range(5)], dtype=np.int64),
        scores=scores,
    )
    allowed = np.ones_like(scores, dtype=bool)
    paths = align_video_conditioned(
        video,
        1,
        allowed=allowed,
        lambda_gap=0.0,
        event_power=1.0,
        cluster_delta=0.0,
        max_paths=2,
        min_separation_ms=0,
    )
    assert len(paths) == 2
    # The top complete path should choose focus position 3 (score: 0.95 + 0.60 + 0.95 = 2.50)
    # over focus position 1 (score: 0.10 + 0.99 + 0.95 = 2.04 or 0.10 + 0.99 + 0.10 = 1.19)
    assert paths[0].focus_frame_position == 3
    assert paths[0].score > paths[1].score
    assert paths[1].focus_frame_position == 1


def test_temporal_search_service_decode_event_alternatives(video_scores_fixture):
    from hcmai.common.config import AlignmentConfig
    from hcmai.corpus import Frame
    from hcmai.orchestration.workflows.search.temporal import TemporalSearchService

    class FakeCorpus:
        def __init__(self, frames):
            self._map = {f.frame_id: f for f in frames}

        def frame(self, frame_id):
            return self._map[frame_id]

    video = video_scores_fixture
    frames = [
        Frame(
            video_id=video.video_id,
            frame_id=str(fid),
            frame_idx=int(idx),
            timestamp_ms=int(t),
            image_path="/tmp/f.jpg",
        )
        for fid, idx, t in zip(
            video.frame_ids, video.frame_idx, video.timestamps_ms, strict=True
        )
    ]
    service = TemporalSearchService(
        corpus=FakeCorpus(frames),
        evidence=None,
        config=AlignmentConfig(
            lambda_gap=0.0,
            event_power=1.0,
            cluster_delta=0.0,
            path_min_separation_ms=0,
        ),
    )
    allowed = np.ones_like(video.scores, dtype=bool)
    alternatives = service.decode_event_alternatives(
        video,
        allowed=allowed,
        focus_event_index=1,
        max_paths=3,
        min_separation_ms=0,
    )
    assert len(alternatives) > 0
    for alt in alternatives:
        assert alt.path.video_id == video.video_id
        assert len(alt.path.frame_ids) == video.scores.shape[0]
        assert alt.focus_timestamp_ms == int(video.timestamps_ms[alt.focus_frame_position])

