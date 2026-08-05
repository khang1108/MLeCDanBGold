"""Smoke test for TRAKE ranking, diversification, and CSV export."""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.agents.trake import TrakePath, VideoEventScores, rank_paths, write_submission


def _video(video_id: str, boost: float) -> VideoEventScores:
    scores = np.array(
        [[boost, 0.0, 0.0, 0.0], [0.0, 0.4, 0.3, 0.2]], dtype=np.float32
    )
    return VideoEventScores(
        video_id=video_id,
        frame_ids=tuple(f"{video_id}_{position}" for position in range(4)),
        frame_idx=(100, 200, 300, 400),
        timestamps_ms=np.arange(4, dtype=np.float64) * 1000.0,
        scores=scores,
    )


def test_best_paths_of_every_video_outrank_any_second_best() -> None:
    rows = rank_paths([_video("L01_V001", 0.5), _video("L02_V002", 0.9)], 0.0, 4)
    assert [row.video_id for row in rows] == [
        "L02_V002",
        "L01_V001",
        "L02_V002",
        "L01_V001",
    ]
    assert rows[0].frame_idx == (100, 200)
    assert rows[2].frame_idx == (100, 300)


def test_rows_are_written_without_a_header_or_extension(tmp_path) -> None:
    output = write_submission(
        [TrakePath("L10_V001.mp4", 1.0, (1200, 1850, 2100))],
        tmp_path / "submission" / "query-1-trake.csv",
    )
    assert output.read_text(encoding="utf-8") == "L10_V001,1200,1850,2100\n"


def test_mixed_event_counts_are_rejected(tmp_path) -> None:
    rows = [TrakePath("a", 1.0, (1, 2)), TrakePath("b", 0.5, (1, 2, 3))]
    with pytest.raises(ValueError, match="mix event counts"):
        write_submission(rows, tmp_path / "bad.csv")
