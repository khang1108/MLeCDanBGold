"""Path-consistent proposals repair a wrong event in fewer actions than local ones."""

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
import pytest

from scripts.evaluation.correction_sim import SimConfig, auroc, event_margins, simulate


def _video():
    # E0 is clearly at f2. E1 is locally strongest at f0/f1, which lie *before* E0,
    # so they can never be part of a valid path. The truth for E1 is f5.
    return VideoEventScores(
        video_id="v",
        frame_ids=np.array([f"f{i}" for i in range(6)]),
        frame_idx=np.arange(6),
        timestamps_ms=np.arange(6, dtype=np.int64) * 10_000,
        scores=np.array([
            [0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
            [0.8, 0.7, 0.1, 0.6, 0.5, 0.4],
        ]),
    )


GT = [(20_000, 20_000), (50_000, 50_000)]


def _cfg(proposals, order="sequential", propagate=True):
    return SimConfig(
        proposals=proposals, order=order, k=3,
        separation_ms=5_000, budget=5, lambda_gap=0.0, propagate=propagate,
    )


def _core(result):
    return {k: result[k] for k in ("correct_at_start", "solved", "actions")}


def test_path_proposals_need_fewer_actions():
    path = simulate(_video(), GT, _cfg("path"))
    local = simulate(_video(), GT, _cfg("local"))

    # Both: action 1 keeps the correct E0.
    # path:  action 2 offers f3,f4,f5 for E1 and uses f5.
    # local: offers f0,f1,f3 (none correct), rejects f3, then f4, then solves.
    assert _core(path) == {"correct_at_start": False, "solved": True, "actions": 2}
    assert _core(local) == {"correct_at_start": False, "solved": True, "actions": 3}


def test_margin_order_skips_the_confident_event():
    # E0 margin is 0.7, E1 margin is 0.1, so margin order inspects E1 first
    # and repairs it with one Use instead of spending an action on Keep.
    result = simulate(_video(), GT, _cfg("path", order="margin"))

    assert _core(result) == {"correct_at_start": False, "solved": True, "actions": 1}


def test_correct_first_decode_costs_nothing():
    result = simulate(_video(), [(20_000, 20_000), (30_000, 30_000)], _cfg("path"))

    assert _core(result) == {"correct_at_start": True, "solved": True, "actions": 0}


def test_fixing_one_event_repairs_its_neighbour_only_with_propagation():
    # First decode is f1,f2 (both wrong). Using f3 for E0 forces E1 past it onto f4.
    video = VideoEventScores(
        video_id="v",
        frame_ids=np.array([f"f{i}" for i in range(5)]),
        frame_idx=np.arange(5),
        timestamps_ms=np.arange(5, dtype=np.int64) * 10_000,
        scores=np.array([
            [0.1, 0.8, 0.1, 0.6, 0.1],
            [0.1, 0.1, 0.7, 0.1, 0.6],
        ]),
    )
    gt = [(30_000, 30_000), (40_000, 40_000)]

    joint = simulate(video, gt, _cfg("path"))
    frozen = simulate(video, gt, _cfg("path", propagate=False))

    assert joint["actions"] == 1 and joint["auto_fixed"] == 1
    assert frozen["actions"] == 2 and frozen["auto_fixed"] == 0
    assert joint["solved"] and frozen["solved"]


def test_auroc_counts_smaller_margins_on_wrong_events():
    assert auroc([0.1, 0.2], [0.5, 0.9]) == 1.0
    assert auroc([0.5], [0.5]) == 0.5
    assert np.isnan(auroc([], [0.3]))


def test_ambiguous_event_gets_smaller_margin():
    # Best path: E0@f0 (0.9) + E1@f1 (0.5) = 1.4.
    # Moving E1 to f3 costs 0.02 (1.38); moving E0 off f0 costs 0.82 (0.58).
    video = VideoEventScores(
        video_id="v",
        frame_ids=np.array(["f0", "f1", "f2", "f3"]),
        frame_idx=np.array([0, 1, 2, 3]),
        timestamps_ms=np.array([0, 10_000, 20_000, 30_000], dtype=np.int64),
        scores=np.array([[0.9, 0.1, 0.1, 0.1], [0.1, 0.5, 0.1, 0.48]]),
    )

    margins = event_margins(video, lambda_gap=0.0, min_separation_ms=5_000)

    assert margins[0] == pytest.approx(0.82)
    assert margins[1] == pytest.approx(0.02)
