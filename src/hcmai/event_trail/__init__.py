"""EventTrail backend module exports."""

from hcmai.event_trail.actions.diff import materialize_snapshot_path
from hcmai.event_trail.config import EventTrailSettings
from hcmai.event_trail.decoding import (
    ConstraintSnapshot,
    DecodeOutcome,
    TemporalConstraintDecoder,
)
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import (
    ApproveEvent,
    ClearAnchor,
    ClearWindow,
    DeclineCandidate,
    EventCandidate,
    EventTrailSession,
    EvidenceSnapshot,
    SetWindow,
    SnapshotResult,
    SubmissionSelection,
    TrailAction,
    TrailCheckpoint,
    TrailTransition,
    TrailView,
    Undo,
    UseFrame,
    freeze_video_scores,
)
from hcmai.event_trail.service import EventTrailService
from hcmai.event_trail.storage import (
    EventTrailSessionStore,
    EvidenceSnapshotStore,
    SessionSlot,
)

__all__ = [
    "ApproveEvent",
    "ClearAnchor",
    "ClearWindow",
    "ConstraintSnapshot",
    "DeclineCandidate",
    "DecodeOutcome",
    "EventCandidate",
    "EventTrailError",
    "EventTrailService",
    "EventTrailSession",
    "EventTrailSessionStore",
    "EventTrailSettings",
    "EvidenceSnapshot",
    "EvidenceSnapshotStore",
    "SessionSlot",
    "SetWindow",
    "SnapshotResult",
    "SubmissionSelection",
    "TemporalConstraintDecoder",
    "TrailAction",
    "TrailCheckpoint",
    "TrailTransition",
    "TrailView",
    "Undo",
    "UseFrame",
    "freeze_video_scores",
    "materialize_snapshot_path",
]
