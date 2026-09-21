"""Domain models for EventTrail snapshots and sessions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from hcmai.event_trail.decoding.decoder import ConstraintSnapshot
from hcmai.orchestration.workflows.search.temporal import DecoderConfigSnapshot
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.constraints import Interval
from hcmai.temporal.dp import AlignedPath


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


@dataclass(frozen=True, slots=True)
class SubmissionSelection:
    """Explicitly selected frame for competition submission."""

    event_id: str
    frame_id: str
    frame_idx: int
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class TrailCheckpoint:
    """Snapshot of session constraints and selection for undo history."""

    constraints: ConstraintSnapshot
    submission_selection: SubmissionSelection | None


@dataclass(frozen=True, slots=True)
class EventTrailSession:
    """Immutable state for one active EventTrail exploration session."""

    session_id: str
    snapshot_id: str
    result_id: str
    video_id: str
    kis_revision: int
    scoring_revision: str
    event_ids: tuple[str, ...]
    video_evidence: VideoEventScores
    decoder_config: DecoderConfigSnapshot
    trail_revision: int
    constraints: ConstraintSnapshot
    current_path: AlignedPath | None
    last_valid_path: AlignedPath | None
    history: tuple[TrailCheckpoint, ...]
    submission_selection: SubmissionSelection | None
    status: Literal["active", "exhausted"]
    search_session_id: str | None = None
    focused_event_id: str | None = None
    alternatives: tuple[TemporalMode, ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalMode:
    """Bounded temporal mode representing one alternative occurrence family."""

    mode_id: str
    event_id: str
    representative_frame_id: str
    representative_frame_idx: int
    representative_timestamp_ms: int
    interval: Interval
    score: float
    path: AlignedPath
    is_current: bool = False


@dataclass(frozen=True, slots=True)
class ApproveEvent:
    """Anchor the currently aligned candidate frame for an event."""

    event_id: str


@dataclass(frozen=True, slots=True)
class UseFrame:
    """Anchor a specific frame for an event and designate it for submission."""

    event_id: str
    frame_id: str


@dataclass(frozen=True, slots=True)
class DeclineCandidate:
    """Exclude the temporal neighborhood cell around an event's candidate."""

    event_id: str


@dataclass(frozen=True, slots=True)
class ClearAnchor:
    """Release any hard anchor on an event."""

    event_id: str


@dataclass(frozen=True, slots=True)
class SetWindow:
    """Restrict alignment to a sub-range [start_ms, end_ms]."""

    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class ClearWindow:
    """Reset the alignment window to the full video."""

    pass


@dataclass(frozen=True, slots=True)
class RepairEvent:
    """Repair candidates within bounded event block bounded by nearest anchors."""

    event_id: str


@dataclass(frozen=True, slots=True)
class Undo:
    """Revert the most recent constraint mutation."""

    pass


@dataclass(frozen=True, slots=True)
class KeepOccurrence:
    """Anchor the currently aligned candidate occurrence for an event."""

    event_id: str


@dataclass(frozen=True, slots=True)
class UseAlternative:
    """Anchor a selected complete-path alternative's representative frame for an event."""

    event_id: str
    alternative_id: str


@dataclass(frozen=True, slots=True)
class RejectMode:
    """Exclude the entire bounded temporal mode interval for an event."""

    event_id: str
    mode_id: str


TrailAction = (
    ApproveEvent
    | UseFrame
    | DeclineCandidate
    | ClearAnchor
    | SetWindow
    | ClearWindow
    | RepairEvent
    | Undo
    | KeepOccurrence
    | UseAlternative
    | RejectMode
)


@dataclass(frozen=True, slots=True)
class EventCandidate:
    """Projected candidate frame for one event in a trail path."""

    event_id: str
    frame_id: str
    frame_idx: int
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class CandidateDiff:
    """Diff describing how one event's candidate moved between revisions."""

    event_id: str
    before_frame_id: str | None
    after_frame_id: str | None
    before_timestamp_ms: int | None
    after_timestamp_ms: int | None


@dataclass(frozen=True, slots=True)
class TrailTransition:
    """Summary of changes and latency resulting from one committed action."""

    action_event_id: str | None
    direct_changed_event_ids: tuple[str, ...]
    indirect_changed_event_ids: tuple[str, ...]
    candidate_diffs: tuple[CandidateDiff, ...]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class TemporalModeView:
    """Projected candidate mode for client-facing TrailView."""

    mode_id: str
    event_id: str
    representative_frame_id: str
    representative_frame_idx: int
    representative_timestamp_ms: int
    interval: Interval
    score: float
    path: tuple[EventCandidate, ...]
    is_current: bool = False


@dataclass(frozen=True, slots=True)
class TrailView:
    """UI-ready presentation projection of an EventTrail session."""

    session_id: str
    result_id: str
    video_id: str
    kis_revision: int
    trail_revision: int
    status: Literal["active", "exhausted"]
    path: tuple[EventCandidate, ...] | None
    last_valid_path: tuple[EventCandidate, ...] | None
    approved_event_ids: tuple[str, ...]
    rejected_counts: dict[str, int]
    window: Interval | None
    submission_selection: SubmissionSelection | None
    transition: TrailTransition | None
    focused_event_id: str | None = None
    alternatives: tuple[TemporalModeView, ...] = ()
    constraints: ConstraintSnapshot | None = None
