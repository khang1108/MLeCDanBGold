"""Baseline-equivalence randomized tests for motion-aware candidate lattice decoding.

Verifies Task 2.2 / SP-03 from the SOICT motion graph implementation plan:
When candidate_k = number_of_frames and transition_weight = 0.0,
decode_candidate_lattice must numerically and structurally reproduce the frozen
baseline align_video across 100+ randomized cases with M in {2, 3, 4}.
"""

import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video
from hcmai.temporal.transition_decoder import (
    decode_candidate_lattice,
    select_event_candidates,
)


def _make_random_video(
    rng: np.random.Generator,
    n_events: int,
    n_frames: int,
) -> VideoEventScores:
    # Generate distinct continuous positive scores
    scores = rng.uniform(0.05, 0.95, size=(n_events, n_frames))
    # Monotonically increasing timestamps with random positive deltas
    intervals = rng.integers(500, 3000, size=n_frames)
    timestamps = np.cumsum(intervals)

    return VideoEventScores(
        video_id=f"rand-video-{n_events}ev-{n_frames}fr",
        frame_ids=np.array([f"f_{i}" for i in range(n_frames)]),
        frame_idx=np.arange(n_frames, dtype=np.int64),
        timestamps_ms=timestamps.astype(np.int64),
        scores=scores,
    )


@pytest.mark.parametrize("n_events", [2, 3, 4])
def test_randomized_baseline_equivalence(n_events: int):
    """Test 40 randomized configurations per M (120 total) for exact baseline equivalence."""
    rng = np.random.default_rng(seed=42 + n_events * 1000)

    for case_idx in range(40):
        # Vary frame count between 6 and 25
        n_frames = rng.integers(max(n_events + 2, 6), 26)
        video = _make_random_video(rng, n_events, n_frames)

        # Randomize parameters
        lambda_gap = float(rng.choice([0.0, 1e-6, 1e-5, 5e-5]))
        paths = int(rng.choice([1, 2, 3]))
        min_sep = int(rng.choice([0, 1000, 2000]))

        baseline_paths = align_video(
            video,
            lambda_gap=lambda_gap,
            paths=paths,
            min_separation_ms=min_sep,
        )

        # 1. Full candidate decode via candidate_k = n_frames
        lattice_paths_k = decode_candidate_lattice(
            video,
            candidate_k=n_frames,
            transition_weight=0.0,
            lambda_gap=lambda_gap,
            paths=paths,
            min_separation_ms=min_sep,
        )

        assert len(baseline_paths) == len(lattice_paths_k), (
            f"Case {case_idx} (M={n_events}, N={n_frames}): "
            f"path count mismatch {len(baseline_paths)} vs {len(lattice_paths_k)}"
        )

        for b_p, l_p in zip(baseline_paths, lattice_paths_k, strict=True):
            assert b_p.frame_idx == l_p.frame_idx, (
                f"Case {case_idx} (M={n_events}, N={n_frames}): "
                f"frame_idx mismatch {b_p.frame_idx} vs {l_p.frame_idx}"
            )
            assert b_p.frame_ids == l_p.frame_ids
            assert np.isclose(b_p.score, l_p.score, atol=1e-7), (
                f"Case {case_idx} (M={n_events}, N={n_frames}): "
                f"score mismatch {b_p.score} vs {l_p.score}"
            )

        # 2. Decode with external candidate layers from select_event_candidates
        ext_candidates = select_event_candidates(video, candidate_k=n_frames)
        lattice_paths_ext = decode_candidate_lattice(
            video,
            candidates=ext_candidates,
            transition_weight=0.0,
            lambda_gap=lambda_gap,
            paths=paths,
            min_separation_ms=min_sep,
        )

        assert len(baseline_paths) == len(lattice_paths_ext)
        for b_p, l_p in zip(baseline_paths, lattice_paths_ext, strict=True):
            assert b_p.frame_idx == l_p.frame_idx
            assert np.isclose(b_p.score, l_p.score, atol=1e-7)


def test_baseline_equivalence_with_dummy_transitions_weight_zero():
    """Verify that random transition matrices with transition_weight=0.0 match baseline DP."""
    rng = np.random.default_rng(seed=999)
    n_events = 3
    n_frames = 12
    video = _make_random_video(rng, n_events, n_frames)

    # Supply random dummy transitions of full shape (12, 12)
    dummy_transitions = [
        rng.uniform(-1.0, 1.0, size=(n_frames, n_frames)),
        rng.uniform(-1.0, 1.0, size=(n_frames, n_frames)),
    ]

    baseline_paths = align_video(video, lambda_gap=1e-5, paths=2)
    lattice_paths = decode_candidate_lattice(
        video,
        candidate_k=n_frames,
        transitions=dummy_transitions,
        transition_weight=0.0,
        lambda_gap=1e-5,
        paths=2,
    )

    assert len(baseline_paths) == len(lattice_paths)
    for b_p, l_p in zip(baseline_paths, lattice_paths, strict=True):
        assert b_p.frame_idx == l_p.frame_idx
        assert np.isclose(b_p.score, l_p.score, atol=1e-7)
