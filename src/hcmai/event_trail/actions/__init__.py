"""Actions subpackage for EventTrail state transitions, diffing, and dispatch."""

from hcmai.event_trail.actions.diff import (
    build_trail_view,
    compute_candidate_diffs,
    materialize_snapshot_path,
)
from hcmai.event_trail.actions.handler import dispatch_action
from hcmai.event_trail.actions.transitions import (
    apply_approve,
    apply_clear_anchor,
    apply_clear_window,
    apply_decline,
    apply_repair,
    apply_set_window,
    apply_undo,
    apply_use_frame,
)

__all__ = [
    "apply_approve",
    "apply_clear_anchor",
    "apply_clear_window",
    "apply_decline",
    "apply_repair",
    "apply_set_window",
    "apply_undo",
    "apply_use_frame",
    "build_trail_view",
    "compute_candidate_diffs",
    "dispatch_action",
    "materialize_snapshot_path",
]
