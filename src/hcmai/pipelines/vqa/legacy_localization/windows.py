"""Fallback temporal-window construction and neighbor expansion."""

from __future__ import annotations

from hcmai.common.schemas import SceneCandidate

from ..domain.models import EvidenceBundle, VideoEvidenceCandidate
from ..domain.ports import TemporalData


def build_windows(
    videos: list[VideoEvidenceCandidate],
    data: TemporalData,
    *,
    duration_ms: int = 15_000,
    max_frames: int = 8,
) -> list[EvidenceBundle]:
    if duration_ms <= 0 or max_frames <= 0:
        raise ValueError("duration_ms and max_frames must be positive")
    raw: list[EvidenceBundle] = []
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
            raw.append(EvidenceBundle(
                scene=SceneCandidate(
                    scene_id=f"{video.video_id}:{start}-{end}",
                    video_id=video.video_id,
                    start_ms=start,
                    end_ms=end,
                    evidence=(source,),
                    unit_scores=dict(source.unit_scores),
                    final_score=source.score,
                ),
                image_frames=sampled,
            ))
    return _merge_overlaps(raw, max_frames)


def expand_neighbor_window(
    bundle: EvidenceBundle,
    data: TemporalData,
    *,
    expansion_ms: int = 8_000,
    max_frames: int = 12,
) -> EvidenceBundle | None:
    if expansion_ms <= 0:
        raise ValueError("expansion_ms must be positive")
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
            "scene_id": f"{scene.video_id}:{start}-{end}:fallback",
            "start_ms": start,
            "end_ms": end,
        }),
        image_frames=sampled,
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


def _merge_overlaps(bundles: list[EvidenceBundle], max_frames: int) -> list[EvidenceBundle]:
    merged: list[EvidenceBundle] = []
    for bundle in sorted(
        bundles,
        key=lambda item: (
            item.scene.video_id,
            item.scene.start_ms,
            item.scene.end_ms,
            -item.scene.final_score,
        ),
    ):
        prior = merged[-1] if merged else None
        if (
            prior is None
            or prior.scene.video_id != bundle.scene.video_id
            or bundle.scene.start_ms > prior.scene.end_ms
        ):
            merged.append(bundle)
            continue
        frame_map = {
            frame.frame_id: frame
            for frame in prior.image_frames + bundle.image_frames
        }
        source_map = {
            item.frame.frame_id: item
            for item in prior.scene.evidence + bundle.scene.evidence
        }
        frames = tuple(sorted(frame_map.values(), key=lambda frame: (frame.timestamp_ms, frame.frame_idx, frame.frame_id))[:max_frames])
        unit_scores = dict(prior.scene.unit_scores)
        for unit_id, score in bundle.scene.unit_scores.items():
            unit_scores[unit_id] = max(unit_scores.get(unit_id, float("-inf")), score)
        end_ms = max(prior.scene.end_ms, bundle.scene.end_ms)
        final_score = max(
            prior.scene.final_score, bundle.scene.final_score
        ) + 0.05 * min(len(source_map), 4)
        merged[-1] = EvidenceBundle(
            scene=SceneCandidate(
                scene_id=f"{prior.scene.video_id}:{prior.scene.start_ms}-{end_ms}",
                video_id=prior.scene.video_id,
                start_ms=prior.scene.start_ms,
                end_ms=end_ms,
                evidence=tuple(source_map.values()),
                unit_scores=unit_scores,
                coverage_score=final_score - max(
                    item.score for item in source_map.values()
                ),
                final_score=final_score,
            ),
            image_frames=frames,
        )
    return sorted(
        merged,
        key=lambda item: (
            -item.scene.final_score,
            item.scene.video_id,
            item.scene.start_ms,
            item.scene.scene_id,
        ),
    )
