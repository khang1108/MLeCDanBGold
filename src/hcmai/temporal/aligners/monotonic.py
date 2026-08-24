"""Canonical compatibility adapter over the stable TRAKE monotonic DP."""

from __future__ import annotations

from hashlib import sha1
from typing import Any

from hcmai.common.schemas import (
    OrderedPathCandidate,
    TemporalAlignmentMode,
    TemporalQueryPlan,
)
from hcmai.data.pipeline import DataService
from hcmai.retrieval.retriever.video_scores import VideoEventScores

from .monotonic_dp import rank_paths


class MonotonicOrderedPathAligner:
    """Run the existing DP and materialize every frame through DataService."""

    def __init__(self, data: DataService, settings: Any) -> None:
        self.data = data
        self.settings = settings

    def align(
        self,
        plan: TemporalQueryPlan,
        video_scores: tuple[VideoEventScores, ...],
        *,
        max_paths: int,
    ) -> tuple[OrderedPathCandidate, ...]:
        if plan.alignment_mode is not TemporalAlignmentMode.ORDERED_PATH:
            raise ValueError("monotonic alignment requires an ordered-path plan")
        for video in video_scores:
            self._validate_video_scores(plan, video)
        rows = rank_paths(
            video_scores,
            self.settings.lambda_gap,
            max_paths,
            self.settings.event_power,
            self.settings.cluster_delta,
        )
        unit_ids = tuple(unit.unit_id for unit in plan.units)
        candidates = []
        for row in rows:
            frames = tuple(self.data.get_frame(frame_id) for frame_id in row.frame_ids)
            if tuple(frame.frame_id for frame in frames) != row.frame_ids:
                raise ValueError("ordered path frame_id conflicts with canonical data")
            if any(frame.video_id != row.video_id for frame in frames):
                raise ValueError("ordered path video_id conflicts with canonical data")
            digest = sha1(
                f"{row.video_id}\0{'|'.join(row.frame_ids)}".encode()
            ).hexdigest()[:16]
            candidates.append(OrderedPathCandidate(
                path_id=f"path-{digest}",
                video_id=row.video_id,
                frames=frames,
                query_unit_ids=unit_ids,
                score=row.score,
                reason_labels=("monotonic_dynamic_programming",),
            ))
        return tuple(candidates)

    def _validate_video_scores(
        self,
        plan: TemporalQueryPlan,
        video: VideoEventScores,
    ) -> None:
        """Reject dense matrices that conflict with canonical frame metadata."""

        frame_count = len(video.frame_ids)
        if video.scores.shape != (len(plan.units), frame_count):
            raise ValueError("dense score matrix shape does not match query plan")
        if not (
            len(video.frame_idx) == frame_count
            and len(video.timestamps_ms) == frame_count
        ):
            raise ValueError("dense score metadata arrays must have equal lengths")
        for position, frame_id in enumerate(video.frame_ids):
            frame = self.data.get_frame(str(frame_id))
            if frame.video_id != video.video_id:
                raise ValueError("dense score frame has mixed canonical video identity")
            if frame.frame_idx != int(video.frame_idx[position]):
                raise ValueError("dense score frame_idx conflicts with canonical data")
            if frame.timestamp_ms != round(float(video.timestamps_ms[position])):
                raise ValueError("dense score timestamp conflicts with canonical data")
