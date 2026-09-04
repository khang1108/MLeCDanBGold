"""Project timestamped ASR segment evidence onto canonical keyframes.

This module owns only deterministic timeline-to-frame projection. It never
creates frame identifiers and does not score, fuse, or rewrite canonical frame
metadata supplied by ``FrameStore``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Mapping, cast

from hcmai.corpus.models import Frame

ProjectionKind = Literal["inside_segment", "nearest_midpoint"]


@dataclass(frozen=True, slots=True)
class SegmentFrameProjection:
    """Canonical frame selected for one timestamped ASR segment.

    ``distance_ms`` is measured from the segment midpoint even when the frame
    lies inside the half-open segment interval.
    """

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    distance_ms: int
    kind: ProjectionKind


class SegmentFrameProjector:
    """Map segment intervals to existing canonical frames deterministically."""

    def __init__(
        self,
        frames: Sequence[Frame] | object,
        max_projection_gap_ms: int = 5_000,
    ) -> None:
        """Bind canonical frame projections and an inclusive gap limit."""

        if (
            isinstance(max_projection_gap_ms, bool)
            or not isinstance(max_projection_gap_ms, Integral)
            or max_projection_gap_ms < 0
        ):
            raise ValueError("max_projection_gap_ms must be a non-negative integer")
        records = cast(Iterable[Any], getattr(frames, "iter_frames", lambda: frames)())
        grouped: defaultdict[str, list[Frame]] = defaultdict(list)
        for frame in records:
            grouped[frame.video_id].append(frame)
        self.frames_by_video = {
            video_id: tuple(
                sorted(
                    values,
                    key=lambda frame: (
                        frame.timestamp_ms,
                        frame.frame_idx,
                        frame.frame_id,
                    ),
                )
            )
            for video_id, values in grouped.items()
        }
        self.max_projection_gap_ms = int(max_projection_gap_ms)

    def project_row(
        self,
        row: Mapping[str, object],
    ) -> SegmentFrameProjection | None:
        """Project a mapping containing ``video_id``, ``start_ms``, and ``end_ms``."""

        return self.project(
            row["video_id"],
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
        )

    def project(
        self,
        video_id: object,
        *,
        start_ms: object,
        end_ms: object,
    ) -> SegmentFrameProjection | None:
        """Select the nearest valid canonical frame for a half-open segment.

        Frames within ``[start_ms, end_ms)`` take priority. If none exists, the
        nearest frame to the midpoint is accepted only when its distance is no
        greater than ``max_projection_gap_ms``. Empty and unknown videos return
        ``None``.
        """

        start, end = _validated_interval(start_ms, end_ms)
        canonical_video_id = _validated_video_id(video_id)
        if canonical_video_id is None:
            return None
        frames = self.frames_by_video.get(canonical_video_id, ())
        if not frames:
            return None

        midpoint = (start + end) // 2
        inside = tuple(
            frame for frame in frames if start <= frame.timestamp_ms < end
        )
        if inside:
            chosen = min(inside, key=lambda frame: _selection_key(frame, midpoint))
            return _projection(chosen, midpoint, "inside_segment")

        chosen = min(frames, key=lambda frame: _selection_key(frame, midpoint))
        distance = abs(chosen.timestamp_ms - midpoint)
        if distance > self.max_projection_gap_ms:
            return None
        return _projection(chosen, midpoint, "nearest_midpoint")


def _validated_video_id(video_id: object) -> str | None:
    """Validate raw segment identity without coercing it into a string."""

    if not isinstance(video_id, str):
        raise ValueError("video_id must be a string")
    return video_id if video_id.strip() else None


def _validated_interval(start_ms: object, end_ms: object) -> tuple[int, int]:
    """Return an integral non-negative interval with positive duration."""

    if isinstance(start_ms, bool) or not isinstance(start_ms, Integral):
        raise ValueError("segment interval coordinates must be integers")
    if isinstance(end_ms, bool) or not isinstance(end_ms, Integral):
        raise ValueError("segment interval coordinates must be integers")
    start, end = int(start_ms), int(end_ms)
    if start < 0 or end <= start:
        raise ValueError(
            "segment interval must have non-negative start_ms and end_ms > start_ms"
        )
    return start, end


def _selection_key(
    frame: Frame,
    midpoint_ms: int,
) -> tuple[int, int, int, str]:
    """Return the stable midpoint-distance ordering required for projection."""

    return (
        abs(frame.timestamp_ms - midpoint_ms),
        frame.timestamp_ms,
        frame.frame_idx,
        frame.frame_id,
    )


def _projection(
    frame: Frame,
    midpoint_ms: int,
    kind: ProjectionKind,
) -> SegmentFrameProjection:
    """Copy canonical identity into an immutable projection result."""

    return SegmentFrameProjection(
        frame_id=frame.frame_id,
        video_id=frame.video_id,
        frame_idx=frame.frame_idx,
        timestamp_ms=frame.timestamp_ms,
        distance_ms=abs(frame.timestamp_ms - midpoint_ms),
        kind=kind,
    )


__all__ = ["SegmentFrameProjection", "SegmentFrameProjector"]
