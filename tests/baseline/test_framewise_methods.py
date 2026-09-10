"""Synthetic tests for full-frame baseline ranking methods."""

from __future__ import annotations

import numpy as np
import pytest

from baseline.contracts import MethodOptions
from baseline.methods.framewise import (
    DanteStyleUnaryDPMethod,
    GlobalQueryMethod,
    IndependentEventsMethod,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import DPPath


def _video(video_id: str, scores: list[list[float]], *, offset: int = 0) -> VideoEventScores:
    matrix = np.asarray(scores, dtype=np.float32)
    frame_count = matrix.shape[1]
    return VideoEventScores(
        video_id=video_id,
        frame_ids=np.asarray([f"{video_id}-f{i}" for i in range(frame_count)]),
        frame_idx=np.arange(offset, offset + frame_count, dtype=np.int64),
        timestamps_ms=np.arange(frame_count, dtype=np.int64) * 1_000,
        scores=matrix,
    )


def test_global_query_ranks_one_best_frame_per_video_deterministically() -> None:
    method = GlobalQueryMethod()
    videos = [
        _video("v2", [[0.3, 0.8, 0.1]]),
        _video("v1", [[0.8, 0.2]]),
        _video("v1", [[0.7, 0.1]], offset=10),
    ]

    paths = method.rank(videos, top_k=10)

    assert [path.video_id for path in paths] == ["v1", "v2"]
    assert paths[0].frame_ids == ("v1-f0",)
    assert paths[1].frame_ids == ("v2-f1",)
    assert paths[0].chronological_event_order is None


def test_global_query_requires_exactly_one_score_row() -> None:
    with pytest.raises(ValueError, match="exactly one score row"):
        GlobalQueryMethod().rank([_video("v1", [[1.0], [2.0]])], top_k=1)


def test_independent_events_means_event_maxima_and_allows_frame_reuse() -> None:
    method = IndependentEventsMethod()
    video = _video(
        "v1",
        [
            [0.1, 0.9, 0.2],
            [0.0, 0.8, 0.3],
            [0.7, 0.1, 0.2],
        ],
    )

    path = method.rank([video], top_k=1)[0]

    assert path.score == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert path.frame_ids == ("v1-f1", "v1-f1", "v1-f0")
    assert path.timestamps_ms == (1_000, 1_000, 0)
    assert path.chronological_event_order is None


def test_independent_events_score_does_not_depend_on_timestamps() -> None:
    first = _video("v1", [[0.2, 0.7], [0.8, 0.1]])
    second = VideoEventScores(
        video_id=first.video_id,
        frame_ids=first.frame_ids,
        frame_idx=first.frame_idx,
        timestamps_ms=np.asarray([100_000, 900_000]),
        scores=first.scores,
    )

    assert IndependentEventsMethod().rank([first], top_k=1)[0].score == pytest.approx(
        IndependentEventsMethod().rank([second], top_k=1)[0].score
    )


def test_dante_style_method_delegates_to_current_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    video = _video("v1", [[5.0, 0.0], [0.0, 4.0]], offset=20)
    calls: list[tuple[object, ...]] = []

    def fake_align(
        value: VideoEventScores,
        lambda_gap: float,
        paths: int,
        event_power: float,
        cluster_delta: float,
    ) -> list[DPPath]:
        calls.append((value, lambda_gap, paths, event_power, cluster_delta))
        return [
            DPPath(
                video_id="v1",
                score=8.5,
                frame_idx=(20, 21),
                frame_ids=("v1-f0", "v1-f1"),
            )
        ]

    monkeypatch.setattr("baseline.methods.framewise.align_video", fake_align)
    options = MethodOptions(lambda_gap=0.2, event_power=2.0, cluster_delta=0.3)

    path = DanteStyleUnaryDPMethod(options).rank([video], top_k=1)[0]

    assert calls == [(video, 0.2, 1, 2.0, 0.3)]
    assert path.frame_idxs == (20, 21)
    assert path.timestamps_ms == (0, 1_000)
    assert path.chronological_event_order == (0, 1)


def test_dante_style_method_matches_hand_checkable_strict_path() -> None:
    video = _video(
        "v1",
        [
            [9.0, 1.0, 0.0, 0.0],
            [0.0, 8.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 7.0],
        ],
    )

    path = DanteStyleUnaryDPMethod(MethodOptions(lambda_gap=0.0)).rank([video], top_k=1)[0]

    assert path.frame_idxs == (0, 1, 3)
    assert path.score == pytest.approx(24.0)


def test_framewise_methods_reject_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        GlobalQueryMethod().rank([_video("v1", [[1.0]])], top_k=0)
