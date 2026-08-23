"""Expand bounded VQA evidence scenes for temporal-answer retries.

This module owns the small temporal neighbor operation used after a VQA answer
attempt needs more context. It does not localize scenes or perform retrieval;
the shared ``TemporalEvidenceCore`` owns initial scene localization.
"""

from __future__ import annotations

from hcmai.common.schemas import FrameRecord

from ..domain.models import EvidenceBundle
from ..domain.ports import TemporalData


def expand_neighbor_window(
    bundle: EvidenceBundle,
    data: TemporalData,
    *,
    expansion_ms: int = 8_000,
    max_frames: int = 12,
) -> EvidenceBundle | None:
    """Return a larger same-video scene when temporal answer context is sparse.

    ``None`` means that the expansion did not discover any new canonical frame.
    Existing scene evidence is always retained and frames remain chronologically
    ordered before the sampling budget is applied.
    """

    if expansion_ms <= 0:
        raise ValueError("expansion_ms must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    scene = bundle.scene
    expanded_by_id = {frame.frame_id: frame for frame in bundle.image_frames}
    for source in scene.evidence:
        radius = max(
            source.frame.timestamp_ms - scene.start_ms,
            scene.end_ms - source.frame.timestamp_ms,
        ) + expansion_ms
        for frame in data.neighbors(
            source.frame.frame_id, window_ms=radius, include_self=True
        ):
            if frame.video_id == scene.video_id:
                expanded_by_id[frame.frame_id] = frame
    if set(expanded_by_id) == set(bundle.image_frame_ids):
        return None

    available = sorted(
        expanded_by_id.values(),
        key=lambda frame: (frame.timestamp_ms, frame.frame_idx, frame.frame_id),
    )
    start, end = available[0].timestamp_ms, available[-1].timestamp_ms
    sampled = _sample(available, start, end, max_frames, set(bundle.image_frame_ids))
    return EvidenceBundle(
        scene=scene.model_copy(update={
            "scene_id": f"{scene.video_id}:{start}-{end}:expanded",
            "start_ms": start,
            "end_ms": end,
        }),
        image_frames=sampled,
    )


def _sample(
    frames: list[FrameRecord],
    start: int,
    end: int,
    limit: int,
    required: set[str],
) -> tuple[FrameRecord, ...]:
    """Sample a chronological frame set while retaining required evidence."""

    available = [frame for frame in frames if start <= frame.timestamp_ms <= end]
    if len(available) <= limit:
        return tuple(available)
    positions = (
        {
            round(index * (len(available) - 1) / (limit - 1))
            for index in range(limit)
        }
        if limit > 1
        else {0}
    )
    selected = {
        frame.frame_id: frame
        for frame in available
        if frame.frame_id in required
    }
    for index in sorted(positions):
        if len(selected) == limit:
            break
        selected.setdefault(available[index].frame_id, available[index])
    if len(selected) < limit:
        for frame in available:
            selected.setdefault(frame.frame_id, frame)
            if len(selected) == limit:
                break
    return tuple(
        sorted(
            selected.values(),
            key=lambda frame: (frame.timestamp_ms, frame.frame_idx, frame.frame_id),
        )
    )
