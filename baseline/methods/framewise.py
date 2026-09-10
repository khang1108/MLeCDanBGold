"""Full-frame unary controls for retrieval and temporal alignment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from baseline.contracts import BaselinePath, MethodOptions
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import DPPath, align_video


class GlobalQueryMethod:
    """Rank each video by its strongest frame for one global query row."""

    name = "global_query"

    def rank(
        self,
        videos: Sequence[VideoEventScores],
        *,
        top_k: int,
    ) -> list[BaselinePath]:
        _validate_top_k(top_k)
        paths: list[BaselinePath] = []
        for video in videos:
            scores = _validated_scores(video)
            if scores.shape[0] != 1:
                raise ValueError("global_query requires exactly one score row per video")
            position = int(np.argmax(scores[0]))
            paths.append(_path_from_positions(video, (position,), float(scores[0, position])))
        return _rank_unique(paths, top_k)


class IndependentEventsMethod:
    """Aggregate independent event maxima without order or distinctness."""

    name = "independent_events"

    def rank(
        self,
        videos: Sequence[VideoEventScores],
        *,
        top_k: int,
    ) -> list[BaselinePath]:
        _validate_top_k(top_k)
        paths: list[BaselinePath] = []
        for video in videos:
            scores = _validated_scores(video)
            positions = tuple(int(position) for position in np.argmax(scores, axis=1))
            maxima = scores[np.arange(scores.shape[0]), np.asarray(positions)]
            paths.append(
                _path_from_positions(
                    video,
                    positions,
                    float(np.mean(maxima, dtype=np.float64)),
                )
            )
        return _rank_unique(paths, top_k)


class DanteStyleUnaryDPMethod:
    """Use the repository's strict monotonic unary DP objective."""

    name = "dante_style_unary_dp"

    def __init__(self, options: MethodOptions) -> None:
        self.options = options

    def rank(
        self,
        videos: Sequence[VideoEventScores],
        *,
        top_k: int,
    ) -> list[BaselinePath]:
        _validate_top_k(top_k)
        paths: list[BaselinePath] = []
        for video in videos:
            _validated_scores(video)
            rows = align_video(
                video,
                self.options.lambda_gap,
                1,
                self.options.event_power,
                self.options.cluster_delta,
            )
            if rows:
                paths.append(_path_from_dp(video, rows[0]))
        return _rank_unique(paths, top_k)


def _validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")


def _validated_scores(video: VideoEventScores) -> np.ndarray:
    scores = np.asarray(video.scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("video scores must be a non-empty two-dimensional matrix")
    frame_count = scores.shape[1]
    if not (
        len(video.frame_ids)
        == len(video.frame_idx)
        == len(video.timestamps_ms)
        == frame_count
    ):
        raise ValueError("video score metadata must match the score columns")
    if not np.isfinite(scores).all():
        raise ValueError("video scores must be finite")
    return scores


def _path_from_positions(
    video: VideoEventScores,
    positions: tuple[int, ...],
    score: float,
    *,
    chronological_event_order: tuple[int, ...] | None = None,
) -> BaselinePath:
    return BaselinePath(
        video_id=video.video_id,
        score=score,
        frame_ids=tuple(str(video.frame_ids[position]) for position in positions),
        frame_idxs=tuple(int(video.frame_idx[position]) for position in positions),
        timestamps_ms=tuple(int(round(float(video.timestamps_ms[position]))) for position in positions),
        chronological_event_order=chronological_event_order,
    )


def _path_from_dp(video: VideoEventScores, row: DPPath) -> BaselinePath:
    positions = {str(frame_id): position for position, frame_id in enumerate(video.frame_ids)}
    try:
        ordered_positions = tuple(positions[frame_id] for frame_id in row.frame_ids)
    except KeyError as error:
        raise ValueError("decoded frame is absent from video score metadata") from error
    actual_frame_idx = tuple(int(video.frame_idx[position]) for position in ordered_positions)
    if actual_frame_idx != row.frame_idx:
        raise ValueError("decoded frame_idx conflicts with video score metadata")
    return _path_from_positions(
        video,
        ordered_positions,
        row.score,
        chronological_event_order=tuple(range(len(ordered_positions))),
    )


def _rank_unique(paths: Sequence[BaselinePath], top_k: int) -> list[BaselinePath]:
    """Sort deterministically and retain at most one row for each video."""

    ordered = sorted(
        paths,
        key=lambda path: (-path.score, path.video_id, path.frame_ids),
    )
    selected: list[BaselinePath] = []
    seen: set[str] = set()
    for path in ordered:
        if path.video_id in seen:
            continue
        selected.append(path)
        seen.add(path.video_id)
        if len(selected) == top_k:
            break
    return selected
