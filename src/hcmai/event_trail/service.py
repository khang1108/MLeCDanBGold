"""Revisioned EventTrail service managing interactive temporal exploration sessions."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from uuid import uuid4

import numpy as np

from hcmai.event_trail.decoder import (
    ConstraintSnapshot,
    TemporalConstraintDecoder,
)
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import (
    ApproveEvent,
    CandidateDiff,
    ClearAnchor,
    ClearWindow,
    DeclineCandidate,
    EventCandidate,
    EventTrailSession,
    SetWindow,
    SnapshotResult,
    SubmissionSelection,
    TrailAction,
    TrailCheckpoint,
    TrailTransition,
    TrailView,
    Undo,
    UseFrame,
)
from hcmai.event_trail.store import (
    EventTrailSessionStore,
    EvidenceSnapshotStore,
    SessionSlot,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import AlignedPath


def materialize_snapshot_path(
    result: SnapshotResult, video: VideoEventScores
) -> AlignedPath:
    """Materialize an AlignedPath from a SnapshotResult and its frozen video scores.

    Frame arrays are verified to uniquely contain every path frame ID. Mismatches
    or inconsistencies raise RuntimeError as an internal correctness failure.
    """
    frame_idxs: list[int] = []
    timestamps_ms: list[int] = []
    for fid in result.initial_path:
        indices = np.where(video.frame_ids == fid)[0]
        if len(indices) != 1:
            raise RuntimeError("snapshot evidence is inconsistent")
        pos = indices[0]
        frame_idxs.append(int(video.frame_idx[pos]))
        timestamps_ms.append(int(video.timestamps_ms[pos]))

    return AlignedPath(
        video_id=result.video_id,
        score=result.path_score,
        frame_ids=result.initial_path,
        frame_idxs=tuple(frame_idxs),
        timestamps_ms=tuple(timestamps_ms),
    )


class EventTrailService:
    """Orchestrate revisioned, interactive exploration sessions on snapshot evidence."""

    def __init__(
        self,
        snapshot_store: EvidenceSnapshotStore,
        session_store: EventTrailSessionStore,
        decoder: TemporalConstraintDecoder,
    ) -> None:
        self.snapshot_store = snapshot_store
        self.session_store = session_store
        self.decoder = decoder

    def open(
        self,
        snapshot_id: str,
        result_id: str,
        expected_kis_revision: int,
    ) -> TrailView:
        """Open a new EventTrail session initialized from one snapshot result."""
        snapshot = self.snapshot_store.get(snapshot_id)
        if snapshot.kis_revision != expected_kis_revision:
            raise EventTrailError(
                "KIS_REVISION_CONFLICT",
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
        )
        self.session_store.put(session)
        return self._build_trail_view(session)

    def get(self, session_id: str) -> TrailView:
        """Fetch the current state projection of an active session."""
        session = self.session_store.get(session_id)
        return self._build_trail_view(session)

    def act(
        self,
        session_id: str,
        expected_trail_revision: int,
        action: TrailAction,
    ) -> TrailView:
        """Apply a transactional feedback action to mutate a session's constraints."""
        with self.session_store.locked(session_id) as slot:
            session = slot.session
            if session.trail_revision != expected_trail_revision:
                raise EventTrailError(
                    "TRAIL_REVISION_CONFLICT",
                    f"Expected trail revision {expected_trail_revision} but session is at {session.trail_revision}",
                )

            if isinstance(action, Undo):
                return self._act_undo(slot)

            if isinstance(action, (ApproveEvent, UseFrame, DeclineCandidate, ClearAnchor)):
                if action.event_id not in session.event_ids:
                    raise EventTrailError("UNKNOWN_EVENT", f"Unknown event {action.event_id}")
                event_idx = session.event_ids.index(action.event_id)
            else:
                event_idx = None

            if isinstance(action, ApproveEvent):
                assert event_idx is not None
                return self._act_approve(slot, action, event_idx)
            elif isinstance(action, UseFrame):
                assert event_idx is not None
                return self._act_use_frame(slot, action, event_idx)
            elif isinstance(action, DeclineCandidate):
                assert event_idx is not None
                return self._act_decline(slot, action, event_idx)
            elif isinstance(action, ClearAnchor):
                assert event_idx is not None
                return self._act_clear_anchor(slot, action, event_idx)
            elif isinstance(action, SetWindow):
                return self._act_set_window(slot, action)
            elif isinstance(action, ClearWindow):
                return self._act_clear_window(slot)
            else:
                raise EventTrailError("INVALID_ACTION", f"Unsupported action {action}")

    def _act_undo(self, slot: SessionSlot) -> TrailView:
        session = slot.session
        if not session.history:
            raise EventTrailError("CANNOT_UNDO", "No prior checkpoint to undo")

        prev_checkpoint = session.history[-1]
        new_history = session.history[:-1]

        outcome = self.decoder.decode(
            session.video_evidence, prev_checkpoint.constraints, session.decoder_config
        )
        if outcome.status == "ok":
            new_path = outcome.path
            new_status = "active"
            new_last_valid = outcome.path
        else:
            new_path = None
            new_status = "exhausted"
            new_last_valid = session.last_valid_path

        diffs: list[CandidateDiff] = []
        indirect_changed: list[str] = []
        for i, eid in enumerate(session.event_ids):
            old_fid = session.current_path.frame_ids[i] if session.current_path else None
            new_fid = new_path.frame_ids[i] if new_path else None
            old_ts = int(session.current_path.timestamps_ms[i]) if session.current_path else None
            new_ts = int(new_path.timestamps_ms[i]) if new_path else None
            if old_fid != new_fid:
                diffs.append(CandidateDiff(eid, old_fid, new_fid, old_ts, new_ts))
                indirect_changed.append(eid)

        transition = TrailTransition(
            action_event_id=None,
            direct_changed_event_ids=(),
            indirect_changed_event_ids=tuple(indirect_changed),
            candidate_diffs=tuple(diffs),
            latency_ms=outcome.constraint_ms + outcome.dp_ms,
        )

        updated = replace(
            session,
            trail_revision=session.trail_revision + 1,
            constraints=prev_checkpoint.constraints,
            current_path=new_path,
            last_valid_path=new_last_valid,
            history=new_history,
            submission_selection=prev_checkpoint.submission_selection,
            status=new_status,
        )
        slot.session = updated
        return self._build_trail_view(updated, transition)

    def _act_approve(self, slot: SessionSlot, action: ApproveEvent, event_idx: int) -> TrailView:
        session = slot.session
        if session.current_path is None:
            raise EventTrailError("CONSTRAINT_CONFLICT", "Cannot approve when path is exhausted")
        if session.constraints.anchors[event_idx] is not None:
            raise EventTrailError("CONSTRAINT_CONFLICT", f"Event {action.event_id} is already anchored")

        current_fid = session.current_path.frame_ids[event_idx]
        new_anchors = list(session.constraints.anchors)
        new_anchors[event_idx] = current_fid
        new_constraints = replace(session.constraints, anchors=tuple(new_anchors))

        outcome = self.decoder.decode(session.video_evidence, new_constraints, session.decoder_config)
        if outcome.status != "ok":
            raise EventTrailError(
                "CONSTRAINT_CONFLICT", f"Approve contradicts constraints: {outcome.status}"
            )

        diffs: list[CandidateDiff] = []
        indirect_changed: list[str] = []
        for i, eid in enumerate(session.event_ids):
            old_fid = session.current_path.frame_ids[i]
            new_fid = outcome.path.frame_ids[i]  # type: ignore[union-attr]
            old_ts = int(session.current_path.timestamps_ms[i])
            new_ts = int(outcome.path.timestamps_ms[i])  # type: ignore[union-attr]
            if old_fid != new_fid:
                diffs.append(CandidateDiff(eid, old_fid, new_fid, old_ts, new_ts))
                if eid != action.event_id:
                    indirect_changed.append(eid)

        transition = TrailTransition(
            action_event_id=action.event_id,
            direct_changed_event_ids=(action.event_id,),
            indirect_changed_event_ids=tuple(indirect_changed),
            candidate_diffs=tuple(diffs),
            latency_ms=outcome.constraint_ms + outcome.dp_ms,
        )
        checkpoint = TrailCheckpoint(
            constraints=session.constraints,
            submission_selection=session.submission_selection,
        )
        updated = replace(
            session,
            trail_revision=session.trail_revision + 1,
            constraints=new_constraints,
            current_path=outcome.path,
            last_valid_path=outcome.path,
            history=session.history + (checkpoint,),
            status="active",
        )
        slot.session = updated
        return self._build_trail_view(updated, transition)

    def _act_use_frame(self, slot: SessionSlot, action: UseFrame, event_idx: int) -> TrailView:
        session = slot.session
        if session.constraints.anchors[event_idx] is not None:
            raise EventTrailError("CONSTRAINT_CONFLICT", f"Event {action.event_id} is already anchored")

        indices = np.where(session.video_evidence.frame_ids == action.frame_id)[0]
        if len(indices) == 0:
            raise EventTrailError(
                "FRAME_NOT_FOUND",
                f"Frame {action.frame_id} not found in video {session.video_id}",
            )
        pos = indices[0]
        resolved_idx = int(session.video_evidence.frame_idx[pos])
        resolved_ts = int(session.video_evidence.timestamps_ms[pos])
        new_selection = SubmissionSelection(
            event_id=action.event_id,
            frame_id=action.frame_id,
            frame_idx=resolved_idx,
            timestamp_ms=resolved_ts,
        )

        new_anchors = list(session.constraints.anchors)
        new_anchors[event_idx] = action.frame_id
        new_constraints = replace(session.constraints, anchors=tuple(new_anchors))

        outcome = self.decoder.decode(session.video_evidence, new_constraints, session.decoder_config)
        if outcome.status != "ok":
            raise EventTrailError(
                "CONSTRAINT_CONFLICT", f"UseFrame contradicts constraints: {outcome.status}"
            )

        diffs: list[CandidateDiff] = []
        indirect_changed: list[str] = []
        for i, eid in enumerate(session.event_ids):
            old_fid = session.current_path.frame_ids[i] if session.current_path else None
            new_fid = outcome.path.frame_ids[i]  # type: ignore[union-attr]
            old_ts = (
                int(session.current_path.timestamps_ms[i]) if session.current_path else None
            )
            new_ts = int(outcome.path.timestamps_ms[i])  # type: ignore[union-attr]
            if old_fid != new_fid:
                diffs.append(CandidateDiff(eid, old_fid, new_fid, old_ts, new_ts))
                if eid != action.event_id:
                    indirect_changed.append(eid)

        transition = TrailTransition(
            action_event_id=action.event_id,
            direct_changed_event_ids=(action.event_id,),
            indirect_changed_event_ids=tuple(indirect_changed),
            candidate_diffs=tuple(diffs),
            latency_ms=outcome.constraint_ms + outcome.dp_ms,
        )
        checkpoint = TrailCheckpoint(
            constraints=session.constraints,
            submission_selection=session.submission_selection,
        )
        updated = replace(
            session,
            trail_revision=session.trail_revision + 1,
            constraints=new_constraints,
            current_path=outcome.path,
            last_valid_path=outcome.path,
            history=session.history + (checkpoint,),
            submission_selection=new_selection,
            status="active",
        )
        slot.session = updated
        return self._build_trail_view(updated, transition)

    def _act_decline(self, slot: SessionSlot, action: DeclineCandidate, event_idx: int) -> TrailView:
        session = slot.session
        if session.current_path is None:
            raise EventTrailError("CONSTRAINT_CONFLICT", "Cannot decline when path is exhausted")
        if session.constraints.anchors[event_idx] is not None:
            raise EventTrailError("CONSTRAINT_CONFLICT", f"Cannot decline anchored event {action.event_id}")

        current_fid = session.current_path.frame_ids[event_idx]
        cell = self.decoder.rejection_cell(session.video_evidence, current_fid)

        new_rejected = list(session.constraints.rejected_cells)
        new_rejected[event_idx] = session.constraints.rejected_cells[event_idx] + (cell,)
        new_constraints = replace(session.constraints, rejected_cells=tuple(new_rejected))

        outcome = self.decoder.decode(session.video_evidence, new_constraints, session.decoder_config)

        if outcome.status == "ok":
            new_path = outcome.path
            new_status = "active"
            new_last_valid = outcome.path
            diffs: list[CandidateDiff] = []
            indirect_changed: list[str] = []
            for i, eid in enumerate(session.event_ids):
                old_fid = session.current_path.frame_ids[i]
                new_fid = outcome.path.frame_ids[i]  # type: ignore[union-attr]
                old_ts = int(session.current_path.timestamps_ms[i])
                new_ts = int(outcome.path.timestamps_ms[i])  # type: ignore[union-attr]
                if old_fid != new_fid:
                    diffs.append(CandidateDiff(eid, old_fid, new_fid, old_ts, new_ts))
                    if eid != action.event_id:
                        indirect_changed.append(eid)
            transition = TrailTransition(
                action_event_id=action.event_id,
                direct_changed_event_ids=(action.event_id,),
                indirect_changed_event_ids=tuple(indirect_changed),
                candidate_diffs=tuple(diffs),
                latency_ms=outcome.constraint_ms + outcome.dp_ms,
            )
        else:
            new_path = None
            new_status = "exhausted"
            new_last_valid = session.last_valid_path
            old_fid = session.current_path.frame_ids[event_idx]
            old_ts = int(session.current_path.timestamps_ms[event_idx])
            diffs = [
                CandidateDiff(
                    event_id=action.event_id,
                    before_frame_id=old_fid,
                    after_frame_id=None,
                    before_timestamp_ms=old_ts,
                    after_timestamp_ms=None,
                )
            ]
            transition = TrailTransition(
                action_event_id=action.event_id,
                direct_changed_event_ids=(action.event_id,),
                indirect_changed_event_ids=(),
                candidate_diffs=tuple(diffs),
                latency_ms=outcome.constraint_ms + outcome.dp_ms,
            )

        checkpoint = TrailCheckpoint(
            constraints=session.constraints,
            submission_selection=session.submission_selection,
        )
        updated = replace(
            session,
            trail_revision=session.trail_revision + 1,
            constraints=new_constraints,
            current_path=new_path,
            last_valid_path=new_last_valid,
            history=session.history + (checkpoint,),
            status=new_status,
        )
        slot.session = updated
        return self._build_trail_view(updated, transition)

    def _act_clear_anchor(self, slot: SessionSlot, action: ClearAnchor, event_idx: int) -> TrailView:
        session = slot.session
        new_anchors = list(session.constraints.anchors)
        new_anchors[event_idx] = None
        new_constraints = replace(session.constraints, anchors=tuple(new_anchors))

        new_selection = (
            None
            if (
                session.submission_selection is not None
                and session.submission_selection.event_id == action.event_id
            )
            else session.submission_selection
        )

        outcome = self.decoder.decode(session.video_evidence, new_constraints, session.decoder_config)
        if outcome.status == "ok":
            new_path = outcome.path
            new_status = "active"
            new_last_valid = outcome.path
        else:
            new_path = None
            new_status = "exhausted"
            new_last_valid = session.last_valid_path

        diffs: list[CandidateDiff] = []
        indirect_changed: list[str] = []
        for i, eid in enumerate(session.event_ids):
            old_fid = session.current_path.frame_ids[i] if session.current_path else None
            new_fid = new_path.frame_ids[i] if new_path else None
            old_ts = int(session.current_path.timestamps_ms[i]) if session.current_path else None
            new_ts = int(new_path.timestamps_ms[i]) if new_path else None
            if old_fid != new_fid:
                diffs.append(CandidateDiff(eid, old_fid, new_fid, old_ts, new_ts))
                if eid != action.event_id:
                    indirect_changed.append(eid)

        transition = TrailTransition(
            action_event_id=action.event_id,
            direct_changed_event_ids=(action.event_id,),
            indirect_changed_event_ids=tuple(indirect_changed),
            candidate_diffs=tuple(diffs),
            latency_ms=outcome.constraint_ms + outcome.dp_ms,
        )
        checkpoint = TrailCheckpoint(
            constraints=session.constraints,
            submission_selection=session.submission_selection,
        )
        updated = replace(
            session,
            trail_revision=session.trail_revision + 1,
            constraints=new_constraints,
            current_path=new_path,
            last_valid_path=new_last_valid,
            history=session.history + (checkpoint,),
            submission_selection=new_selection,
            status=new_status,
        )
        slot.session = updated
        return self._build_trail_view(updated, transition)

    def _act_set_window(self, slot: SessionSlot, action: SetWindow) -> TrailView:
        session = slot.session
        if action.start_ms < 0 or action.start_ms > action.end_ms:
            raise EventTrailError("INVALID_WINDOW", "Window start must be <= end and >= 0")

        new_constraints = replace(session.constraints, window=(action.start_ms, action.end_ms))
        outcome = self.decoder.decode(session.video_evidence, new_constraints, session.decoder_config)
        if outcome.status != "ok":
            raise EventTrailError(
                "CONSTRAINT_CONFLICT", f"SetWindow contradicts constraints: {outcome.status}"
            )

        diffs: list[CandidateDiff] = []
        indirect_changed: list[str] = []
        for i, eid in enumerate(session.event_ids):
            old_fid = session.current_path.frame_ids[i] if session.current_path else None
            new_fid = outcome.path.frame_ids[i]  # type: ignore[union-attr]
            old_ts = (
                int(session.current_path.timestamps_ms[i]) if session.current_path else None
            )
            new_ts = int(outcome.path.timestamps_ms[i])  # type: ignore[union-attr]
            if old_fid != new_fid:
                diffs.append(CandidateDiff(eid, old_fid, new_fid, old_ts, new_ts))
                indirect_changed.append(eid)

        transition = TrailTransition(
            action_event_id=None,
            direct_changed_event_ids=(),
            indirect_changed_event_ids=tuple(indirect_changed),
            candidate_diffs=tuple(diffs),
            latency_ms=outcome.constraint_ms + outcome.dp_ms,
        )
        checkpoint = TrailCheckpoint(
            constraints=session.constraints,
            submission_selection=session.submission_selection,
        )
        updated = replace(
            session,
            trail_revision=session.trail_revision + 1,
            constraints=new_constraints,
            current_path=outcome.path,
            last_valid_path=outcome.path,
            history=session.history + (checkpoint,),
            status="active",
        )
        slot.session = updated
        return self._build_trail_view(updated, transition)

    def _act_clear_window(self, slot: SessionSlot) -> TrailView:
        session = slot.session
        new_constraints = replace(session.constraints, window=None)
        outcome = self.decoder.decode(session.video_evidence, new_constraints, session.decoder_config)
        if outcome.status == "ok":
            new_path = outcome.path
            new_status = "active"
            new_last_valid = outcome.path
        else:
            new_path = None
            new_status = "exhausted"
            new_last_valid = session.last_valid_path

        diffs: list[CandidateDiff] = []
        indirect_changed: list[str] = []
        for i, eid in enumerate(session.event_ids):
            old_fid = session.current_path.frame_ids[i] if session.current_path else None
            new_fid = new_path.frame_ids[i] if new_path else None
            old_ts = int(session.current_path.timestamps_ms[i]) if session.current_path else None
            new_ts = int(new_path.timestamps_ms[i]) if new_path else None
            if old_fid != new_fid:
                diffs.append(CandidateDiff(eid, old_fid, new_fid, old_ts, new_ts))
                indirect_changed.append(eid)

        transition = TrailTransition(
            action_event_id=None,
            direct_changed_event_ids=(),
            indirect_changed_event_ids=tuple(indirect_changed),
            candidate_diffs=tuple(diffs),
            latency_ms=outcome.constraint_ms + outcome.dp_ms,
        )
        checkpoint = TrailCheckpoint(
            constraints=session.constraints,
            submission_selection=session.submission_selection,
        )
        updated = replace(
            session,
            trail_revision=session.trail_revision + 1,
            constraints=new_constraints,
            current_path=new_path,
            last_valid_path=new_last_valid,
            history=session.history + (checkpoint,),
            status=new_status,
        )
        slot.session = updated
        return self._build_trail_view(updated, transition)

    def _build_trail_view(
        self, session: EventTrailSession, transition: TrailTransition | None = None
    ) -> TrailView:
        path_candidates = None
        if session.current_path is not None:
            path_candidates = tuple(
                EventCandidate(
                    event_id=eid,
                    frame_id=fid,
                    frame_idx=fidx,
                    timestamp_ms=ts,
                )
                for eid, fid, fidx, ts in zip(
                    session.event_ids,
                    session.current_path.frame_ids,
                    session.current_path.frame_idxs,
                    session.current_path.timestamps_ms,
                    strict=True,
                )
            )

        last_valid_candidates = None
        if session.last_valid_path is not None:
            last_valid_candidates = tuple(
                EventCandidate(
                    event_id=eid,
                    frame_id=fid,
                    frame_idx=fidx,
                    timestamp_ms=ts,
                )
                for eid, fid, fidx, ts in zip(
                    session.event_ids,
                    session.last_valid_path.frame_ids,
                    session.last_valid_path.frame_idxs,
                    session.last_valid_path.timestamps_ms,
                    strict=True,
                )
            )

        approved_ids = tuple(
            eid
            for eid, anchor in zip(session.event_ids, session.constraints.anchors, strict=True)
            if anchor is not None
        )

        rejected_counts = {
            eid: len(cells)
            for eid, cells in zip(
                session.event_ids, session.constraints.rejected_cells, strict=True
            )
        }

        return TrailView(
            session_id=session.session_id,
            result_id=session.result_id,
            video_id=session.video_id,
            kis_revision=session.kis_revision,
            trail_revision=session.trail_revision,
            status=session.status,
            path=path_candidates,
            last_valid_path=last_valid_candidates,
            approved_event_ids=approved_ids,
            rejected_counts=rejected_counts,
            window=session.constraints.window,
            submission_selection=session.submission_selection,
            transition=transition,
        )
