"""Tests for benchmark rank accounting under paths_per_video above one."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "benchmark" / "score.py"
_spec = importlib.util.spec_from_file_location("benchmark_score", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)


def _label(video_id: str, *timestamps_ms: int) -> dict:
    return {
        "candidates": [
            {"video_id": video_id, "timestamp_ms": value} for value in timestamps_ms
        ]
    }


def _path(video_id: str, *timestamps_ms: int) -> dict:
    return {"video_id": video_id, "timestamps_ms": list(timestamps_ms)}


def test_video_rank_counts_distinct_videos_not_path_positions() -> None:
    # One wrong video occupying five rows must cost the gold video one place, not five.
    paths = [_path("WRONG", 1_000 * index) for index in range(5)]
    paths.append(_path("GOLD", 60_000))

    video_rank, frame_rank = score._hit_ranks(_label("GOLD", 60_000), paths, 2_000)

    assert video_rank == 2
    assert frame_rank == 6


def test_frame_hit_accepts_the_tolerance_window() -> None:
    paths = [_path("GOLD", 61_500)]

    assert score._hit_ranks(_label("GOLD", 60_000), paths, 2_000)[1] == 1
    assert score._hit_ranks(_label("GOLD", 60_000), paths, 1_000)[1] is None


def test_video_rank_survives_a_gold_video_whose_moments_all_miss() -> None:
    paths = [_path("GOLD", 0), _path("GOLD", 300_000)]

    video_rank, frame_rank = score._hit_ranks(_label("GOLD", 600_000), paths, 2_000)

    assert video_rank == 1
    assert frame_rank is None
