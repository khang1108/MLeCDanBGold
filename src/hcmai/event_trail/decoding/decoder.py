"""Selected-video temporal constraint decoder.

This module owns projecting user anchors, rejections, and windows onto
pure temporal frame masks and executing DP decoding for a single video.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from hcmai.orchestration.workflows.search.temporal import (
    DecoderConfigSnapshot,
    TemporalSearchService,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.constraints import (
    Conditions,
    Interval,
    build_mask,
)
from hcmai.temporal.dp import AlignedPath


@dataclass(frozen=True, slots=True)
class ConstraintSnapshot:
    """Immutable snapshot of user-applied constraints for a video."""

    anchors: tuple[str | None, ...]
    rejected_cells: tuple[tuple[Interval, ...], ...]
    window: Interval | None


@dataclass(frozen=True, slots=True)
class DecodeOutcome:
    """Outcome of decoding one video under constraints."""

    status: Literal["ok", "contradictory_conditions", "no_indexed_frames", "no_valid_path"]
    path: AlignedPath | None
    constraint_ms: float
    dp_ms: float


def timestamp_for_frame(video: VideoEventScores, frame_id: str) -> int:
    """Find the canonical timestamp in ms for a given frame ID in video scores."""
    indices = np.where(video.frame_ids == frame_id)[0]
    if len(indices) == 0:
        raise ValueError(f"Frame {frame_id} not found in video {video.video_id}")
    return int(video.timestamps_ms[indices[0]])


class TemporalConstraintDecoder:
    """Decode selected-video paths under user-provided temporal constraints."""

    def __init__(self, temporal: TemporalSearchService) -> None:
        self.temporal = temporal

    def rejection_cell(self, video: VideoEventScores, frame_id: str) -> Interval:
        """Derive the rejection interval in integer milliseconds for a frame.

        For strictly increasing neighbors around timestamp t:
        - left boundary: floor((t_prev + t) / 2) + 1;
        - right boundary: floor((t + t_next) / 2);
        - first timestamp group starts at full-video window start;
        - last timestamp group ends at full-video window end.
        Equal timestamp runs are treated as one occurrence and use nearest
        strictly smaller/larger timestamps as midpoint neighbors.
        A video with one distinct timestamp rejects exactly that timestamp.
        """
        target_t = timestamp_for_frame(video, frame_id)
        unique_ts = np.unique(video.timestamps_ms)

        if len(unique_ts) == 1:
            return (target_t, target_t)

        i = int(np.where(unique_ts == target_t)[0][0])

        if i == 0:
            left = int(video.timestamps_ms[0])
        else:
            t_prev = int(unique_ts[i - 1])
            left = ((t_prev + target_t) // 2) + 1

        if i == len(unique_ts) - 1:
            right = int(video.timestamps_ms[-1])
        else:
            t_next = int(unique_ts[i + 1])
            right = (target_t + t_next) // 2

        return (left, right)

    def decode(
        self,
        video: VideoEventScores,
        constraints: ConstraintSnapshot,
        decoder_config: DecoderConfigSnapshot,
    ) -> DecodeOutcome:
        """Decode paths for one video under given constraints and config."""
        c_start = perf_counter()

        confirmed: list[Interval | None] = []
        for anchor_fid in constraints.anchors:
            if anchor_fid is None:
                confirmed.append(None)
            else:
                t = timestamp_for_frame(video, anchor_fid)
                confirmed.append((t, t))

        window = constraints.window or (
            int(video.timestamps_ms[0]),
            int(video.timestamps_ms[-1]),
        )

        conditions = Conditions(
            window=window,
            confirmed=tuple(confirmed),
            rejected=constraints.rejected_cells,
        )

        try:
            mask, domain_status = build_mask(video.timestamps_ms, conditions)
        except ValueError:
            constraint_ms = (perf_counter() - c_start) * 1_000.0
            return DecodeOutcome(
                status="contradictory_conditions",
                path=None,
                constraint_ms=constraint_ms,
                dp_ms=0.0,
            )

        constraint_ms = (perf_counter() - c_start) * 1_000.0

        if domain_status in ("contradictory_conditions", "no_indexed_frames"):
            return DecodeOutcome(
                status=domain_status,
                path=None,
                constraint_ms=constraint_ms,
                dp_ms=0.0,
            )

        dp_start = perf_counter()
        paths = self.temporal.decode_video(
            video,
            allowed=mask,
            decoder_config=decoder_config,
        )
        dp_ms = (perf_counter() - dp_start) * 1_000.0

        if paths:
            return DecodeOutcome(
                status="ok",
                path=paths[0],
                constraint_ms=constraint_ms,
                dp_ms=dp_ms,
            )

        return DecodeOutcome(
            status="no_valid_path",
            path=None,
            constraint_ms=constraint_ms,
            dp_ms=dp_ms,
        )

    def repair(
        self,
        video: VideoEventScores,
        constraints: ConstraintSnapshot,
        decoder_config: DecoderConfigSnapshot,
        current_path: AlignedPath | None,
        target_event_index: int,
    ) -> DecodeOutcome:
        """Decode paths for one video under local temporal repair constraints."""
        c_start = perf_counter()

        if constraints.anchors[target_event_index] is not None:
            raise ValueError(
                f"Target event index {target_event_index} is still anchored; "
                "must clear anchor before repair."
            )

        start_idx, end_idx = repair_block(constraints.anchors, target_event_index)

        t_left: int | None = None
        t_right: int | None = None

        if start_idx > 0 and constraints.anchors[start_idx - 1] is not None:
            t_left = timestamp_for_frame(video, constraints.anchors[start_idx - 1])

        if end_idx < len(constraints.anchors) - 1 and constraints.anchors[end_idx + 1] is not None:
            t_right = timestamp_for_frame(video, constraints.anchors[end_idx + 1])

        if t_left is not None and t_right is not None and t_left >= t_right:
            constraint_ms = (perf_counter() - c_start) * 1_000.0
            return DecodeOutcome(
                status="contradictory_conditions",
                path=None,
                constraint_ms=constraint_ms,
                dp_ms=0.0,
            )

        confirmed: list[Interval | None] = []
        for anchor_fid in constraints.anchors:
            if anchor_fid is None:
                confirmed.append(None)
            else:
                t = timestamp_for_frame(video, anchor_fid)
                confirmed.append((t, t))

        window = constraints.window or (
            int(video.timestamps_ms[0]),
            int(video.timestamps_ms[-1]),
        )

        rejected_cells = constraints.rejected_cells
        if len(rejected_cells) != len(constraints.anchors):
            rejected_cells = tuple(() for _ in constraints.anchors)

        conditions = Conditions(
            window=window,
            confirmed=tuple(confirmed),
            rejected=rejected_cells,
        )

        try:
            mask, domain_status = build_mask(video.timestamps_ms, conditions)
        except ValueError:
            constraint_ms = (perf_counter() - c_start) * 1_000.0
            return DecodeOutcome(
                status="contradictory_conditions",
                path=None,
                constraint_ms=constraint_ms,
                dp_ms=0.0,
            )

        if domain_status in ("contradictory_conditions", "no_indexed_frames"):
            constraint_ms = (perf_counter() - c_start) * 1_000.0
            return DecodeOutcome(
                status=domain_status,
                path=None,
                constraint_ms=constraint_ms,
                dp_ms=0.0,
            )

        for e in range(start_idx, end_idx + 1):
            if t_left is not None:
                mask[e] &= (video.timestamps_ms > t_left)
            if t_right is not None:
                mask[e] &= (video.timestamps_ms < t_right)

        fixed_assignments: dict[int, str] = {}
        if current_path is not None:
            for e in range(len(video.scores)):
                if e < start_idx or e > end_idx:
                    if e < len(current_path.frame_ids):
                        fixed_assignments[e] = current_path.frame_ids[e]

        for e, anchor_fid in enumerate(constraints.anchors):
            if anchor_fid is not None:
                fixed_assignments[e] = anchor_fid

        for e, frame_id in fixed_assignments.items():
            mask[e] &= (video.frame_ids == frame_id)

        if any(mask[e].sum() == 0 for e in range(len(video.scores))):
            constraint_ms = (perf_counter() - c_start) * 1_000.0
            return DecodeOutcome(
                status="no_valid_path",
                path=None,
                constraint_ms=constraint_ms,
                dp_ms=0.0,
            )

        constraint_ms = (perf_counter() - c_start) * 1_000.0

        dp_start = perf_counter()
        paths = self.temporal.decode_video(
            video,
            allowed=mask,
            decoder_config=decoder_config,
        )
        dp_ms = (perf_counter() - dp_start) * 1_000.0

        if paths:
            return DecodeOutcome(
                status="ok",
                path=paths[0],
                constraint_ms=constraint_ms,
                dp_ms=dp_ms,
            )

        return DecodeOutcome(
            status="no_valid_path",
            path=None,
            constraint_ms=constraint_ms,
            dp_ms=dp_ms,
        )


def repair_block(
    anchors: Sequence[str | None],
    target_event_index: int,
) -> tuple[int, int]:
    """Return inclusive unconfirmed block indices [start_idx, end_idx] bounded by nearest anchors."""
    n = len(anchors)
    if target_event_index < 0 or target_event_index >= n:
        raise ValueError(
            f"target_event_index {target_event_index} out of bounds for {n} events"
        )

    left_anchor = -1
    for i in range(target_event_index - 1, -1, -1):
        if anchors[i] is not None:
            left_anchor = i
            break
    start_idx = left_anchor + 1

    right_anchor = n
    for i in range(target_event_index + 1, n):
        if anchors[i] is not None:
            right_anchor = i
            break
    end_idx = right_anchor - 1

    return (start_idx, end_idx)
