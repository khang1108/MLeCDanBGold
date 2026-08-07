"""Deterministic temporal and video diversity policies for KIS."""

from __future__ import annotations

from dataclasses import dataclass

from hcmai.common.schemas import RetrievalCandidate


@dataclass(frozen=True, slots=True)
class KISRankingConfig:
    temporal_window_ms: int = 3_000
    early_diversity_depth: int = 20
    max_per_video_early: int = 3

    def __post_init__(self) -> None:
        if self.temporal_window_ms < 0:
            raise ValueError("temporal_window_ms must not be negative")
        if self.early_diversity_depth < 1:
            raise ValueError("early_diversity_depth must be positive")
        if self.max_per_video_early < 1:
            raise ValueError("max_per_video_early must be positive")


def shape_kis_candidates(
    candidates: list[RetrievalCandidate],
    data,
    config: KISRankingConfig,
    *,
    minimum_results: int = 0,
) -> list[RetrievalCandidate]:
    """Remove local duplicates then diversify without changing candidate IDs."""

    deduplicated = _temporal_deduplicate(candidates, data, config.temporal_window_ms)
    diversified = _diversify_videos(
        deduplicated,
        data,
        config.early_diversity_depth,
        config.max_per_video_early,
    )
    selected_ids = {candidate.frame_id for candidate in diversified}
    for candidate in candidates:
        if len(diversified) >= minimum_results:
            break
        if candidate.frame_id not in selected_ids:
            diversified.append(candidate)
            selected_ids.add(candidate.frame_id)
    return diversified


def _temporal_deduplicate(candidates, data, window_ms):
    selected: list[RetrievalCandidate] = []
    selected_times: dict[str, list[tuple[int, int]]] = {}
    for candidate in candidates:
        frame = data.get_frame(candidate.frame_id)
        neighbors = selected_times.setdefault(frame.video_id, [])
        suppressed_by = next(
            (
                selected_index
                for timestamp, selected_index in neighbors
                if abs(frame.timestamp_ms - timestamp) <= window_ms
            ),
            None,
        )
        if suppressed_by is None:
            neighbors.append((frame.timestamp_ms, len(selected)))
            selected.append(candidate)
            continue
        kept = selected[suppressed_by]
        metadata = dict(kept.metadata)
        alternates = list(metadata.get("temporal_alternate_frame_ids", []))
        alternates.append(candidate.frame_id)
        metadata["temporal_alternate_frame_ids"] = alternates
        selected[suppressed_by] = kept.model_copy(update={"metadata": metadata})
    return selected


def _diversify_videos(candidates, data, depth, max_per_video):
    if len(candidates) < 2:
        return candidates
    remaining = list(candidates)
    output = [remaining.pop(0)]
    first_video = data.get_frame(output[0].frame_id).video_id
    counts = {first_video: 1}
    target_depth = min(depth, len(candidates))
    while len(output) < target_depth and remaining:
        selected_index = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if counts.get(data.get_frame(candidate.frame_id).video_id, 0)
                < max_per_video
            ),
            0,
        )
        candidate = remaining.pop(selected_index)
        video_id = data.get_frame(candidate.frame_id).video_id
        counts[video_id] = counts.get(video_id, 0) + 1
        output.append(candidate)
    output.extend(remaining)
    return output
