"""Revisioned EventTrail service managing interactive temporal exploration sessions.

This module coordinates exploration session lifecycle (open, close, get, act)
by delegating state transitions, diffing, and constraints decoding to dedicated subpackages.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from uuid import uuid4

from hcmai.event_trail.config import EventTrailSettings

from hcmai.event_trail.actions.diff import (
    build_trail_view,
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
from hcmai.event_trail.decoding.decoder import (
    ConstraintSnapshot,
    TemporalConstraintDecoder,
)
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.logging import log_trail_event
from hcmai.event_trail.models import (
    ApproveEvent,
    ClearAnchor,
    DeclineCandidate,
    EventTrailSession,
    RepairEvent,
    SetWindow,
    TemporalMode,
    TrailAction,
    TrailTransition,
    TrailView,
    UseFrame,
)
from hcmai.event_trail.storage.session import (
    EventTrailSessionStore,
    SessionSlot,
)
from hcmai.event_trail.storage.snapshot import EvidenceSnapshotStore


class EventTrailService:
    """Orchestrate revisioned, interactive exploration sessions on snapshot evidence."""

    def __init__(
        self,
        snapshot_store: EvidenceSnapshotStore,
        session_store: EventTrailSessionStore,
        decoder: TemporalConstraintDecoder,
        settings: EventTrailSettings | None = None,
    ) -> None:
        self.snapshot_store = snapshot_store
        self.session_store = session_store
        self.decoder = decoder
        self.settings = settings or EventTrailSettings()

    def open(
        self,
        snapshot_id: str,
        result_id: str,
        expected_kis_revision: int,
        search_session_id: str | None = None,
    ) -> TrailView:
        """Open a new EventTrail session initialized from one snapshot result."""
        started = perf_counter()
        snapshot = self.snapshot_store.get(snapshot_id)
        if snapshot.kis_revision != expected_kis_revision:
            raise EventTrailError(
                "KIS_REVISION_MISMATCH",
                f"Expected KIS revision {expected_kis_revision} but snapshot has {snapshot.kis_revision}",
            )
        if result_id not in snapshot.results:
            raise EventTrailError(
                "RESULT_NOT_FOUND",
                f"Result {result_id} not found in snapshot {snapshot_id}",
            )

        snapshot_result = snapshot.results[result_id]
        if snapshot_result.video_id not in snapshot.video_evidence:
            raise RuntimeError("snapshot evidence is inconsistent")
        video = snapshot.video_evidence[snapshot_result.video_id]

        initial_path = materialize_snapshot_path(snapshot_result, video)

        anchors = tuple(None for _ in snapshot.event_ids)
        rejected_cells = tuple(() for _ in snapshot.event_ids)
        constraints = ConstraintSnapshot(
            anchors=anchors,
            rejected_cells=rejected_cells,
            window=None,
        )

        session_id = f"ses_{uuid4().hex}"
        session = EventTrailSession(
            session_id=session_id,
            snapshot_id=snapshot_id,
            result_id=result_id,
            video_id=snapshot_result.video_id,
            kis_revision=snapshot.kis_revision,
            scoring_revision=snapshot.scoring_revision,
            event_ids=snapshot.event_ids,
            video_evidence=video,
            decoder_config=snapshot.decoder_config,
            trail_revision=0,
            constraints=constraints,
            current_path=initial_path,
            last_valid_path=initial_path,
            history=(),
            submission_selection=None,
            status="active",
            search_session_id=search_session_id,
        )
        self.session_store.put(session)
        view = build_trail_view(session)
        total_ms = (perf_counter() - started) * 1000.0

        log_trail_event(
            event_type="trail_open",
            kis_revision=snapshot.kis_revision,
            snapshot_id=snapshot_id,
            result_id=result_id,
            video_id=snapshot_result.video_id,
            trail_session_id=session_id,
            trail_revision=0,
            event_id=None,
            search_session_id=search_session_id,
            payload={
                "initial_path": list(snapshot_result.initial_path),
                "total_ms": round(total_ms, 3),
            },
        )
        return view

    def close(
        self,
        session_id: str,
        expected_trail_revision: int | None = None,
    ) -> None:
        """Close an active exploration session and remove it from store."""
        started = perf_counter()
        with self.session_store.locked(session_id) as slot:
            session = slot.session
            if (
                expected_trail_revision is not None
                and session.trail_revision != expected_trail_revision
            ):
                raise EventTrailError(
                    "TRAIL_REVISION_CONFLICT",
                    f"Expected trail revision {expected_trail_revision} but session is at {session.trail_revision}",
                )
            self.session_store.remove(session_id)
            total_ms = (perf_counter() - started) * 1000.0
            log_trail_event(
                event_type="trail_close",
                kis_revision=session.kis_revision,
                snapshot_id=session.snapshot_id,
                result_id=session.result_id,
                video_id=session.video_id,
                trail_session_id=session.session_id,
                trail_revision=session.trail_revision,
                event_id=None,
                search_session_id=session.search_session_id,
                payload={
                    "total_ms": round(total_ms, 3),
                },
            )

    def get(self, session_id: str) -> TrailView:
        """Fetch the current state projection of an active session."""
        session = self.session_store.get(session_id)
        return build_trail_view(session)

    def alternatives(
        self,
        session_id: str,
        expected_trail_revision: int,
        event_id: str,
    ) -> tuple[TemporalMode, ...]:
        """Return non-mutating complete-path alternatives for a focused event."""
        with self.session_store.locked(session_id) as slot:
            session = slot.session
            if session.trail_revision != expected_trail_revision:
                raise EventTrailError(
                    "TRAIL_REVISION_CONFLICT",
                    f"Expected trail revision {expected_trail_revision} but session is at {session.trail_revision}",
                )
            if event_id not in session.event_ids:
                raise EventTrailError("INVALID_EVENT", f"Unknown event {event_id}")
            event_idx = session.event_ids.index(event_id)

            if session.focused_event_id == event_id and session.alternatives:
                return session.alternatives

            modes = self.decoder.alternatives(
                session.video_evidence,
                session.constraints,
                session.decoder_config,
                event_idx=event_idx,
                event_id=event_id,
                settings=self.settings,
                current_path=session.current_path,
            )
            slot.session = replace(
                session,
                focused_event_id=event_id,
                alternatives=modes,
            )
            return modes

    def act(
        self,
        session_id: str,
        expected_trail_revision: int,
        action: TrailAction,
    ) -> TrailView:
        """Apply a transactional feedback action to mutate a session's constraints."""
        started = perf_counter()
        with self.session_store.locked(session_id) as slot:
            session = slot.session
            if session.trail_revision != expected_trail_revision:
                raise EventTrailError(
                    "TRAIL_REVISION_CONFLICT",
                    f"Expected trail revision {expected_trail_revision} but session is at {session.trail_revision}",
                )
            return dispatch_action(slot, action, self.decoder, started)

    def _build_trail_view(
        self, session: EventTrailSession, transition: TrailTransition | None = None
    ) -> TrailView:
        """Backward-compatibility helper delegating to build_trail_view."""
        return build_trail_view(session, transition)

    def _act_undo(self, slot: SessionSlot, started: float) -> TrailView:
        """Backward-compatibility delegation for undo action."""
        return apply_undo(slot, self.decoder, started)

    def _act_repair(
        self, slot: SessionSlot, action: RepairEvent, event_idx: int, started: float
    ) -> TrailView:
        """Backward-compatibility delegation for repair action."""
        return apply_repair(slot, action, event_idx, self.decoder, started)

    def _act_approve(
        self, slot: SessionSlot, action: ApproveEvent, event_idx: int, started: float
    ) -> TrailView:
        """Backward-compatibility delegation for approve action."""
        return apply_approve(slot, action, event_idx, self.decoder, started)

    def _act_use_frame(
        self, slot: SessionSlot, action: UseFrame, event_idx: int, started: float
    ) -> TrailView:
        """Backward-compatibility delegation for use_frame action."""
        return apply_use_frame(slot, action, event_idx, self.decoder, started)

    def _act_decline(
        self, slot: SessionSlot, action: DeclineCandidate, event_idx: int, started: float
    ) -> TrailView:
        """Backward-compatibility delegation for decline action."""
        return apply_decline(slot, action, event_idx, self.decoder, started)

    def _act_clear_anchor(
        self, slot: SessionSlot, action: ClearAnchor, event_idx: int, started: float
    ) -> TrailView:
        """Backward-compatibility delegation for clear_anchor action."""
        return apply_clear_anchor(slot, action, event_idx, self.decoder, started)

    def _act_set_window(
        self, slot: SessionSlot, action: SetWindow, started: float
    ) -> TrailView:
        """Backward-compatibility delegation for set_window action."""
        return apply_set_window(slot, action, self.decoder, started)

    def _act_clear_window(self, slot: SessionSlot, started: float) -> TrailView:
        """Backward-compatibility delegation for clear_window action."""
        return apply_clear_window(slot, self.decoder, started)


__all__ = [
    "EventTrailService",
    "materialize_snapshot_path",
]
