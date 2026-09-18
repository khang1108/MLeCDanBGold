"""Action dispatcher for EventTrail exploration sessions.

This module inspects incoming TrailAction commands, validates targeted event IDs,
and routes execution to corresponding state transition handlers.
"""

from __future__ import annotations

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
from hcmai.event_trail.decoding.decoder import TemporalConstraintDecoder
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import (
    ApproveEvent,
    ClearAnchor,
    ClearWindow,
    DeclineCandidate,
    RepairEvent,
    SetWindow,
    TrailAction,
    TrailView,
    Undo,
    UseFrame,
)
from hcmai.event_trail.storage.session import SessionSlot


def dispatch_action(
    slot: SessionSlot,
    action: TrailAction,
    decoder: TemporalConstraintDecoder,
    started: float,
) -> TrailView:
    """Validate and route an incoming TrailAction to its state transition handler."""
    session = slot.session

    if isinstance(action, Undo):
        return apply_undo(slot, decoder, started)

    if isinstance(action, (ApproveEvent, UseFrame, DeclineCandidate, ClearAnchor, RepairEvent)):
        if action.event_id not in session.event_ids:
            raise EventTrailError("INVALID_EVENT", f"Unknown event {action.event_id}")
        event_idx = session.event_ids.index(action.event_id)
    else:
        event_idx = None

    if isinstance(action, ApproveEvent):
        assert event_idx is not None
        return apply_approve(slot, action, event_idx, decoder, started)
    elif isinstance(action, UseFrame):
        assert event_idx is not None
        return apply_use_frame(slot, action, event_idx, decoder, started)
    elif isinstance(action, DeclineCandidate):
        assert event_idx is not None
        return apply_decline(slot, action, event_idx, decoder, started)
    elif isinstance(action, ClearAnchor):
        assert event_idx is not None
        return apply_clear_anchor(slot, action, event_idx, decoder, started)
    elif isinstance(action, RepairEvent):
        assert event_idx is not None
        return apply_repair(slot, action, event_idx, decoder, started)
    elif isinstance(action, SetWindow):
        return apply_set_window(slot, action, decoder, started)
    elif isinstance(action, ClearWindow):
        return apply_clear_window(slot, decoder, started)
    else:
        raise EventTrailError("INVALID_ACTION", f"Unsupported action {action}")
