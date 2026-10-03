"""Tests for Candidate Recall@K evaluation (SP-11)."""

import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.candidate_recall import (
    EventAnnotation,
    evaluate_candidate_recall,
    format_candidate_recall_table,
    is_candidate_hit,
)


def _make_test_video() -> VideoEventScores:
    # 2 events, 6 frames
    # Event 0: scores highest at frame 1 (score 0.9), frame 0 (score 0.8), frame 2 (score 0.7)
    # Event 1: scores highest at frame 5 (score 0.9), frame 4 (score 0.8), frame 3 (score 0.7)
    scores = np.array([
        [0.8, 0.9, 0.7, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7, 0.8, 0.9],
    ])
    return VideoEventScores(
        video_id="v_eval",
        frame_ids=np.array([f"f_{i}" for i in range(6)]),
        frame_idx=np.array([0, 10, 20, 30, 40, 50], dtype=np.int64),
        timestamps_ms=np.array([1000, 2000, 3000, 4000, 5000, 6000], dtype=np.int64),
        scores=scores,
    )


def test_is_candidate_hit():
    video = _make_test_video()

    # Annotation by frame_idx: targets frame_idx 10
    ann_idx = EventAnnotation(video_id="v_eval", event_index=0, target_frame_idxs=(10,))
    assert is_candidate_hit(np.array([1]), video, ann_idx)  # frame 1 has frame_idx 10
    assert not is_candidate_hit(np.array([0, 2]), video, ann_idx)

    # Annotation by timestamp window: 1500 to 2500 ms
    ann_window = EventAnnotation(video_id="v_eval", event_index=0, start_ms=1500, end_ms=2500)
    assert is_candidate_hit(np.array([1]), video, ann_window)  # frame 1 is 2000ms
    assert not is_candidate_hit(np.array([0, 2]), video, ann_window)


def test_evaluate_candidate_recall_monotonicity():
    """Verify CandidateRecall increases with K as candidate coverage expands."""
    video = _make_test_video()

    # Ground truth: event 0 is at frame 2 (frame_idx 20), event 1 is at frame 3 (frame_idx 30)
    # Notice: for event 0, scores rank: [1, 0, 2, ...].
    # So frame 2 is rank 3! It enters candidate set only when K >= 3!
    # For event 1, scores rank: [5, 4, 3, ...].
    # So frame 3 is rank 3! It enters candidate set only when K >= 3!
    annotations = [
        EventAnnotation(video_id="v_eval", event_index=0, target_frame_idxs=(20,)),
        EventAnnotation(video_id="v_eval", event_index=1, target_frame_idxs=(30,)),
    ]

    results = evaluate_candidate_recall(
        annotations,
        video_scores={"v_eval": video},
        k_values=(1, 2, 4),
    )

    assert len(results) == 3
    # At K=1: candidate layers have 1 frame each (frame 1 for ev0, frame 5 for ev1) -> 0 hits
    assert results[0].k == 1
    assert results[0].hit_events == 0
    assert results[0].recall == 0.0

    # At K=2: candidate layers have 2 frames each (frames 1, 0 for ev0; frames 5, 4 for ev1) -> 0 hits
    assert results[1].k == 2
    assert results[1].hit_events == 0
    assert results[1].recall == 0.0

    # At K=4: candidate layers have 4 frames each (frames 1, 0, 2 for ev0; frames 5, 4, 3 for ev1) -> 2 hits!
    assert results[2].k == 4
    assert results[2].hit_events == 2
    assert results[2].recall == 1.0

    # Format table
    table = format_candidate_recall_table(results)
    assert "| K | Candidate Recall |" in table
    assert "100.00%" in table
