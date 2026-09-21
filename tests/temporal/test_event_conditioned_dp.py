"""Tests for event-conditioned complete-path dynamic programming."""

import itertools
import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video_conditioned


def test_conditioned_path_never_reuses_frame_for_focus_e0():
    """Two events, two frames: no valid path can reuse the same frame.

    Bug: searchsorted(source, frames) returned t as a valid successor of t,
    producing path (0, 0) with score 1.2 instead of the only valid strict
    path (0, 1) with score 0.7.
    """
    # E0 scores: frame0=0.4, frame1=0.6
    # E1 scores: frame0=0.8, frame1=0.3
    # Invalid optimum (same frame): E0@frame0, E1@frame0 → score=1.2
    # Valid strict optimum:         E0@frame0, E1@frame1 → score=0.7
    scores = np.array([[0.4, 0.6], [0.8, 0.3]], dtype=np.float32)
    video = VideoEventScores(
        video_id="v_strict",
        frame_ids=np.array(["f0", "f1"]),
        frame_idx=np.array([0, 1]),
        timestamps_ms=np.array([0, 1000], dtype=np.int64),
        scores=scores,
    )
    allowed = np.ones_like(scores, dtype=bool)
    paths = align_video_conditioned(
        video, focus_event_index=0, allowed=allowed,
        lambda_gap=0.0, max_paths=4
    )
    # Must return at least one path (the valid one)
    assert len(paths) >= 1
    for p in paths:
        # frame_idx values are the organizer-facing indices (0, 10, 20, …)
        # but for this fixture frame_idx = [0, 1]; positions in score matrix
        # are the indices of the frame_idx array.
        positions = [int(np.where(video.frame_idx == idx)[0][0]) for idx in p.path.frame_idx]
        for a, b in zip(positions, positions[1:]):
            assert a < b, (
                f"Path {p.path.frame_idx} violates strict chronology: "
                f"score matrix positions {positions}"
            )


def test_conditioned_path_score_matches_brute_force_focus_e0_all_sizes():
    """Exhaustive random correctness: focus on first event across small sizes."""
    rng = np.random.default_rng(2026)
    mismatches = 0
    total = 0
    for n_events, n_frames in [(2, 2), (2, 3), (3, 3), (3, 4)]:
        for _ in range(30):
            s = rng.random((n_events, n_frames))
            video = VideoEventScores(
                video_id="v_rand",
                frame_ids=np.array([f"f{i}" for i in range(n_frames)]),
                frame_idx=np.arange(n_frames),
                timestamps_ms=np.arange(n_frames, dtype=np.int64) * 1000,
                scores=s.astype(np.float32),
            )
            allowed = np.ones((n_events, n_frames), dtype=bool)
            paths = align_video_conditioned(
                video, focus_event_index=0, allowed=allowed,
                lambda_gap=0.0, max_paths=n_frames
            )
            for p in paths:
                focus_pos = p.focus_frame_position
                # Brute-force: all strictly-increasing combinations with E0 at focus_pos
                best = None
                for combo in itertools.combinations(range(n_frames), n_events):
                    if combo[0] != focus_pos:
                        continue
                    score = sum(float(s[e, combo[e]]) for e in range(n_events))
                    if best is None or score > best:
                        best = score
                if best is not None:
                    total += 1
                    if abs(p.score - best) > 1e-5:
                        mismatches += 1
    assert mismatches == 0, f"{mismatches}/{total} mismatch(es) found"


def brute_force(
    video,
    focus_event_index,
    focus_position,
    allowed,
    lambda_gap=0.0,
    cluster_delta=0.0,
):
    """Reference exhaustive evaluator enforcing strict chronology and cluster constraints."""
    from hcmai.temporal.dp import cluster_starts

    best = None
    scores = np.asarray(video.scores, dtype=np.float64)
    n_events, n_frames = scores.shape
    starts = (
        cluster_starts(scores, cluster_delta)
        if cluster_delta > 0.0
        else np.arange(n_frames)
    )
    # If the video does not contain at least n_events distinct clusters, no valid path exists
    if int(np.count_nonzero(starts == np.arange(n_frames))) < n_events:
        return None

    timestamps = np.asarray(video.timestamps_ms, dtype=np.float64)
    for positions in itertools.combinations(range(n_frames), n_events):
        if positions[focus_event_index] != focus_position:
            continue
        if any(not allowed[e, p] for e, p in enumerate(positions)):
            continue
        # Cluster constraint: successor must not share the current frame's cluster
        if any(starts[positions[e + 1]] < positions[e] + 1 for e in range(n_events - 1)):
            continue
        score = sum(float(scores[e, p]) for e, p in enumerate(positions))
        if lambda_gap > 0 and n_events > 1:
            score -= lambda_gap * (timestamps[positions[-1]] - timestamps[positions[0]])
        if best is None or score > best[0]:
            best = (score, positions)
    return best


def test_conditioned_dp_brute_force_property_across_configurations():
    """Exhaustive property test: align_video_conditioned matches brute force across seeds, deltas, and gaps."""
    from hcmai.temporal.dp import align_video, cluster_starts

    rng = np.random.default_rng(2026_09_19)
    configs = [
        # (n_events, n_frames, cluster_delta, lambda_gap)
        (2, 3, 0.0, 0.0),
        (2, 4, 0.25, 0.0),
        (2, 5, 0.35, 1e-4),
        (3, 4, 0.0, 1e-3),
        (3, 5, 0.20, 0.0),
        (3, 6, 0.30, 1e-4),
    ]

    total_tested = 0
    for n_events, n_frames, cluster_delta, lambda_gap in configs:
        for _ in range(8):
            s = rng.random((n_events, n_frames), dtype=np.float32)
            video = VideoEventScores(
                video_id="v_prop",
                frame_ids=np.array([f"f{i}" for i in range(n_frames)]),
                frame_idx=np.arange(n_frames),
                timestamps_ms=np.arange(n_frames, dtype=np.int64) * 1000,
                scores=s,
            )
            allowed = np.ones((n_events, n_frames), dtype=bool)
            starts = (
                cluster_starts(s, cluster_delta)
                if cluster_delta > 0.0
                else np.arange(n_frames)
            )

            for focus_e in range(n_events):
                paths = align_video_conditioned(
                    video,
                    focus_event_index=focus_e,
                    allowed=allowed,
                    lambda_gap=lambda_gap,
                    cluster_delta=cluster_delta,
                    max_paths=n_frames,
                )
                for p in paths:
                    total_tested += 1
                    focus_pos = p.focus_frame_position
                    expected = brute_force(
                        video,
                        focus_e,
                        focus_pos,
                        allowed,
                        lambda_gap=lambda_gap,
                        cluster_delta=cluster_delta,
                    )
                    assert expected is not None, (
                        f"DP returned path {p.path.frame_idx} for focus {focus_e}@{focus_pos} "
                        f"but brute force found no valid path under cluster_delta={cluster_delta}"
                    )
                    assert abs(p.score - expected[0]) < 1e-5, (
                        f"Score mismatch for focus {focus_e}@{focus_pos}: "
                        f"DP={p.score} vs BruteForce={expected[0]}"
                    )
                    # Verify strict chronology
                    pos_list = [int(idx) for idx in p.path.frame_idx]
                    for a, b in zip(pos_list, pos_list[1:]):
                        assert a < b, f"Chronology violated in {pos_list}"
                        if cluster_delta > 0.0:
                            assert starts[b] >= a + 1, (
                                f"Cluster constraint violated between frame {a} (starts={starts[a]}) "
                                f"and {b} (starts={starts[b]})"
                            )

            # Check that top-1 conditioned path across all focus positions matches unconditioned align_video
            uncond_paths = align_video(
                video,
                allowed=allowed,
                lambda_gap=lambda_gap,
                cluster_delta=cluster_delta,
                paths=1,
            )
            cond_e0_paths = align_video_conditioned(
                video,
                focus_event_index=0,
                allowed=allowed,
                lambda_gap=lambda_gap,
                cluster_delta=cluster_delta,
                max_paths=n_frames,
            )
            if uncond_paths:
                assert cond_e0_paths, "Unconditioned found path but conditioned found none"
                assert abs(uncond_paths[0].score - cond_e0_paths[0].score) < 1e-5
                assert tuple(uncond_paths[0].frame_idx) == tuple(cond_e0_paths[0].path.frame_idx)

    assert total_tested > 100, f"Expected >100 configurations tested, got {total_tested}"


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

