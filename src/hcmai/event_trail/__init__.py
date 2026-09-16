"""EventTrail backend module exports."""

from hcmai.event_trail.config import EventTrailSettings
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import (
    EvidenceSnapshot,
    SnapshotResult,
    freeze_video_scores,
)
from hcmai.event_trail.decoder import (
    ConstraintSnapshot,
    DecodeOutcome,
    TemporalConstraintDecoder,
)
from hcmai.event_trail.store import EvidenceSnapshotStore

__all__ = [
    "ConstraintSnapshot",
    "DecodeOutcome",
    "EventTrailError",
    "EventTrailSettings",
    "EvidenceSnapshot",
    "EvidenceSnapshotStore",
    "SnapshotResult",
    "TemporalConstraintDecoder",
    "freeze_video_scores",
]
