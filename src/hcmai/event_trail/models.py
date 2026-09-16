"""Domain models for EventTrail snapshots and sessions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from hcmai.orchestration.workflows.temporal_search import DecoderConfigSnapshot
from hcmai.retrieval.retriever.video_scores import VideoEventScores


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    """Immutable mapping from KIS product result ID to temporal path evidence."""

    result_id: str
    video_id: str
    initial_path: tuple[str, ...]
    path_score: float


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    """Immutable evidence snapshot captured from one KIS search execution."""

    snapshot_id: str
    kis_revision: int
    scoring_revision: str
    event_ids: tuple[str, ...]
    decoder_config: DecoderConfigSnapshot
    results: dict[str, SnapshotResult]
    video_evidence: dict[str, VideoEventScores]
    created_at: datetime
    expires_at: datetime


def freeze_video_scores(video: VideoEventScores) -> VideoEventScores:
    """Create an immutable copy of VideoEventScores with read-only NumPy arrays."""
    arrays = {}
    for name in ("frame_ids", "frame_idx", "timestamps_ms", "scores"):
        value = getattr(video, name).copy()
        value.setflags(write=False)
        arrays[name] = value
    return replace(video, **arrays)
