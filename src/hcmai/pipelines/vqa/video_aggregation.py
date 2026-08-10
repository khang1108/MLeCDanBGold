"""Aggregate frame evidence before expensive VQA inference."""

from __future__ import annotations

from collections import defaultdict

from hcmai.common.schemas import RetrievalSource

from .models import BranchCandidate, VideoEvidenceCandidate


def aggregate_videos(
    candidates: list[BranchCandidate], *, top_videos: int = 10, neighborhood_ms: int = 15_000
) -> list[VideoEvidenceCandidate]:
    if top_videos < 1 or neighborhood_ms < 1:
        raise ValueError("top_videos and neighborhood_ms must be positive")
    groups: dict[str, list[BranchCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.frame.video_id].append(candidate)
    videos: list[VideoEvidenceCandidate] = []
    for video_id, frames in groups.items():
        ordered = sorted(frames, key=lambda item: (-item.score, item.frame.timestamp_ms, item.frame.frame_id))
        modalities = {source for frame in frames for source in frame.source_scores}
        neighborhoods = {frame.frame.timestamp_ms // neighborhood_ms for frame in frames}
        clue_hits = sum(
            source in (RetrievalSource.OCR, RetrievalSource.ASR)
            for frame in frames for source in frame.source_scores
        )
        consistent = sum(frame.score for frame in ordered[:3]) / min(3, len(ordered))
        coverage = 0.05 * len(modalities) + 0.03 * len(neighborhoods)
        duplicate_penalty = 0.02 * max(0, len(frames) - len(neighborhoods) * 2)
        score = ordered[0].score + 0.25 * consistent + coverage - duplicate_penalty
        videos.append(VideoEvidenceCandidate(
            video_id=video_id,
            frames=tuple(ordered),
            score=score,
            best_event_rank=_best_rank(ordered, "event"),
            best_question_rank=_best_rank(ordered, "question"),
            modality_count=len(modalities),
            neighborhood_count=len(neighborhoods),
            clue_coverage=min(1.0, clue_hits / max(1, len(frames))),
        ))
    return sorted(videos, key=lambda item: (-item.score, item.video_id))[:top_videos]


def _best_rank(frames: list[BranchCandidate], branch: str) -> int | None:
    ranks = [index for index, frame in enumerate(frames, 1) if branch in frame.branch_scores]
    return min(ranks, default=None)
