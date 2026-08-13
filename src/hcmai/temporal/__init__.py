"""Shared progressive temporal-evidence core for KIS and VQA."""

from .core import ProgressiveLocalizationResult, TemporalEvidenceCore
from .evidence import ProgressiveEvidenceState, retrieval_to_evidence
from .query import SnapshotDiffMode, SnapshotDiffResult, diff_snapshot
from .state import (
    ProgressiveSearchState,
    ProgressiveStateConflictError,
    ProgressiveStateStore,
    StaleProgressiveStateError,
)

__all__ = [
    "ProgressiveEvidenceState",
    "ProgressiveLocalizationResult",
    "ProgressiveSearchState",
    "ProgressiveStateConflictError",
    "ProgressiveStateStore",
    "SnapshotDiffMode",
    "SnapshotDiffResult",
    "StaleProgressiveStateError",
    "TemporalEvidenceCore",
    "diff_snapshot",
    "retrieval_to_evidence",
]
