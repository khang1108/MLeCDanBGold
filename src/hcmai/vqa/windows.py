"""Bounded temporal window construction and one-shot neighbor expansion."""

from __future__ import annotations

from .contracts import TemporalData
from .models import BranchCandidate, TemporalWindow, VideoEvidenceCandidate


def build_windows(
    videos: list[VideoEvidenceCandidate],
    data: TemporalData,
    *,
    duration_ms: int = 15_000,
    max_frames: int = 8,
) -> list[TemporalWindow]:
    if duration_ms <= 0 or max_frames <= 0:
        raise ValueError("duration_ms and max_frames must be positive")
    raw: list[TemporalWindow] = []
    half = duration_ms // 2
    for video in videos:
        for source in video.frames:
            nearby = [
                frame for frame in data.neighbors(
                    source.frame.frame_id, window_ms=half, include_self=True
                )
                if frame.video_id == video.video_id
            ]
            if not nearby:
                nearby = [source.frame]
            start = min(frame.timestamp_ms for frame in nearby)
            end = max(frame.timestamp_ms for frame in nearby)
            sampled = _sample(nearby, start, end, max_frames, {source.frame.frame_id})
            raw.append(TemporalWindow(
                window_id=f"{video.video_id}:{start}-{end}", video_id=video.video_id,
                start_ms=start, end_ms=end, source_frames=(source,), sampled_frames=sampled,
                score=source.score,
            ))
    return _merge_overlaps(raw, max_frames, duration_ms)


def expand_neighbor_window(
    window: TemporalWindow, data: TemporalData, *, expansion_ms: int = 8_000, max_frames: int = 12
) -> TemporalWindow | None:
    if expansion_ms <= 0:
        raise ValueError("expansion_ms must be positive")
    expanded_by_id = {frame.frame_id: frame for frame in window.sampled_frames}
    for source in window.source_frames:
        radius = max(
            source.frame.timestamp_ms - window.start_ms,
            window.end_ms - source.frame.timestamp_ms,
        ) + expansion_ms
        for frame in data.neighbors(
            source.frame.frame_id, window_ms=radius, include_self=True
        ):
            if frame.video_id == window.video_id:
                expanded_by_id[frame.frame_id] = frame
    if set(expanded_by_id) == set(window.frame_ids):
        return None
    available = sorted(
        expanded_by_id.values(),
        key=lambda frame: (frame.timestamp_ms, frame.frame_idx, frame.frame_id),
    )
    start, end = available[0].timestamp_ms, available[-1].timestamp_ms
    sampled = _sample(available, start, end, max_frames, set(window.frame_ids))
    return TemporalWindow(
        window_id=f"{window.video_id}:{start}-{end}:fallback", video_id=window.video_id,
        start_ms=start, end_ms=end, source_frames=window.source_frames,
        sampled_frames=sampled, score=window.score,
    )


def _sample(frames, start: int, end: int, limit: int, required: set[str]):
    available = [frame for frame in frames if start <= frame.timestamp_ms <= end]
    if len(available) <= limit:
        return tuple(available)
    positions = {round(index * (len(available) - 1) / (limit - 1)) for index in range(limit)} if limit > 1 else {0}
    required_frames = [frame for frame in available if frame.frame_id in required][:limit]
    selected = {frame.frame_id: frame for frame in required_frames}
    for index in sorted(positions):
        if len(selected) == limit:
            break
        selected.setdefault(available[index].frame_id, available[index])
    if len(selected) < limit:
        for frame in available:
            selected.setdefault(frame.frame_id, frame)
            if len(selected) == limit:
                break
    return tuple(sorted(selected.values(), key=lambda frame: (frame.timestamp_ms, frame.frame_idx, frame.frame_id)))


def _merge_overlaps(
    windows: list[TemporalWindow], max_frames: int, duration_ms: int
) -> list[TemporalWindow]:
    merged: list[TemporalWindow] = []
    for window in sorted(windows, key=lambda item: (item.video_id, item.start_ms, item.end_ms, -item.score)):
        prior = merged[-1] if merged else None
        end = window.end_ms if prior is None else max(prior.end_ms, window.end_ms)
        if (
            prior is None
            or prior.video_id != window.video_id
            or window.start_ms > prior.end_ms
            or end - prior.start_ms > duration_ms
        ):
            merged.append(window)
            continue
        source_map = {item.frame.frame_id: item for item in prior.source_frames + window.source_frames}
        if len(source_map) > max_frames:
            merged.append(window)
            continue
        frame_map = {frame.frame_id: frame for frame in prior.sampled_frames + window.sampled_frames}
        frames = _sample(
            sorted(frame_map.values(), key=lambda frame: (frame.timestamp_ms, frame.frame_idx, frame.frame_id)),
            prior.start_ms,
            end,
            max_frames,
            set(source_map),
        )
        merged[-1] = TemporalWindow(
            window_id=f"{prior.video_id}:{prior.start_ms}-{end}",
            video_id=prior.video_id, start_ms=prior.start_ms, end_ms=end,
            source_frames=tuple(source_map.values()), sampled_frames=frames,
            score=max(prior.score, window.score) + 0.05 * min(len(source_map), 4),
        )
    return sorted(merged, key=lambda item: (-item.score, item.video_id, item.start_ms, item.window_id))
