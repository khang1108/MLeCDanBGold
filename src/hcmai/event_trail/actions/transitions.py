"""State transition functions for EventTrail user actions.

Each transition function validates domain preconditions, decodes the selected
video under updated constraints, records history checkpoints, emits audit logs,
and updates the session slot.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from hcmai.event_trail.actions.diff import (
    build_trail_view,
    compute_candidate_diffs,
)
from hcmai.event_trail.decoding.decoder import (
    TemporalConstraintDecoder,
    repair_block,
)
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.logging import log_trail_event
from hcmai.event_trail.models import (
    ApproveEvent,
    CandidateDiff,
    ClearAnchor,
    DeclineCandidate,
    RepairEvent,
    SetWindow,
    SubmissionSelection,
    TrailCheckpoint,
    TrailTransition,
    TrailView,
    UseFrame,
)
from hcmai.event_trail.storage.session import SessionSlot


def apply_undo(
    slot: SessionSlot,
    decoder: TemporalConstraintDecoder,
    started: float,
) -> TrailView:
    """Revert the most recent constraint change from session history."""
    session = slot.session
    if not session.history:
        raise EventTrailError("NOTHING_TO_UNDO", "No prior checkpoint to undo")

    prev_checkpoint = session.history[-1]
    new_history = session.history[:-1]

    outcome = decoder.decode(
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

    diffs, indirect_changed, diff_ms = compute_candidate_diffs(
        session.event_ids,
        session.current_path,
        new_path,
        action_event_id=None,
    )

    transition = TrailTransition(
        action_event_id=None,
        direct_changed_event_ids=(),
        indirect_changed_event_ids=indirect_changed,
        candidate_diffs=diffs,
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
    total_ms = (perf_counter() - started) * 1000.0

    log_trail_event(
        event_type="trail_undo",
        kis_revision=session.kis_revision,
        snapshot_id=session.snapshot_id,
        result_id=session.result_id,
        video_id=session.video_id,
        trail_session_id=session.session_id,
        trail_revision=updated.trail_revision,
        event_id=None,
        search_session_id=session.search_session_id,
        payload={
            "path_before": list(session.current_path.frame_ids) if session.current_path else [],
            "path_after": list(new_path.frame_ids) if new_path else None,
            "direct_changed_event_ids": list(transition.direct_changed_event_ids),
            "indirect_changed_event_ids": list(transition.indirect_changed_event_ids),
            "outcome": new_status,
            "constraint_ms": round(outcome.constraint_ms, 3),
            "dp_ms": round(outcome.dp_ms, 3),
            "diff_ms": round(diff_ms, 3),
            "total_ms": round(total_ms, 3),
        },
    )
    if new_status == "exhausted" and session.status != "exhausted":
        log_trail_event(
            event_type="trail_exhausted",
            kis_revision=session.kis_revision,
            snapshot_id=session.snapshot_id,
            result_id=session.result_id,
            video_id=session.video_id,
            trail_session_id=session.session_id,
            trail_revision=updated.trail_revision,
            event_id=None,
            search_session_id=session.search_session_id,
            payload={
                "exhausted_by_event_id": None,
                "last_valid_path": list(new_last_valid.frame_ids) if new_last_valid else [],
                "total_ms": round(total_ms, 3),
            },
        )
    return build_trail_view(updated, transition)


def apply_repair(
    slot: SessionSlot,
    action: RepairEvent,
    event_idx: int,
    decoder: TemporalConstraintDecoder,
    started: float,
) -> TrailView:
    """Perform localized repair bounded by surrounding anchors."""
    session = slot.session
    if session.constraints.anchors[event_idx] is not None:
        raise EventTrailError(
            "INVALID_EVENT",
            f"Event {action.event_id} is still anchored; clear anchor before repair",
        )

    outcome = decoder.repair(
        session.video_evidence,
        session.constraints,
        session.decoder_config,
        session.current_path or session.last_valid_path,
        event_idx,
    )

    checkpoint = TrailCheckpoint(
        constraints=session.constraints,
        submission_selection=session.submission_selection,
    )
    new_history = (*session.history, checkpoint)

    start_idx, end_idx = repair_block(session.constraints.anchors, event_idx)
    diffs: list[CandidateDiff] = []
    indirect_changed: list[str] = []

    if outcome.status == "ok" and outcome.path is not None:
        new_path = outcome.path
        new_status = "active"
        new_last_valid = outcome.path

        ref_path = session.current_path or session.last_valid_path
        if ref_path is not None:
            for i in range(start_idx, end_idx + 1):
                eid = session.event_ids[i]
                old_fid = ref_path.frame_ids[i]
                new_fid = new_path.frame_ids[i]
                old_ts = int(ref_path.timestamps_ms[i])
                new_ts = int(new_path.timestamps_ms[i])
                if old_fid != new_fid:
                    diffs.append(CandidateDiff(eid, old_fid, new_fid, old_ts, new_ts))
                    if eid != action.event_id:
                        indirect_changed.append(eid)
    else:
        new_path = None
        new_status = "exhausted"
        new_last_valid = session.last_valid_path

    transition = TrailTransition(
        action_event_id=action.event_id,
        direct_changed_event_ids=(action.event_id,),
        indirect_changed_event_ids=tuple(indirect_changed),
        candidate_diffs=tuple(diffs),
        latency_ms=outcome.constraint_ms + outcome.dp_ms,
    )

    updated = replace(
        session,
        trail_revision=session.trail_revision + 1,
        current_path=new_path,
        last_valid_path=new_last_valid,
        status=new_status,
        history=new_history,
    )
    slot.session = updated
    return build_trail_view(updated, transition)


def apply_approve(
    slot: SessionSlot,
    action: ApproveEvent,
    event_idx: int,
    decoder: TemporalConstraintDecoder,
    started: float,
) -> TrailView:
    """Approve candidate frame at event_idx and pin it as an anchor."""
    session = slot.session
    if session.current_path is None:
        raise EventTrailError("CONSTRAINT_CONFLICT", "Cannot approve when path is exhausted")
    if session.constraints.anchors[event_idx] is not None:
        raise EventTrailError("CONSTRAINT_CONFLICT", f"Event {action.event_id} is already anchored")

    current_fid = session.current_path.frame_ids[event_idx]
    new_anchors = list(session.constraints.anchors)
    new_anchors[event_idx] = current_fid
    new_constraints = replace(session.constraints, anchors=tuple(new_anchors))

    outcome = decoder.decode(session.video_evidence, new_constraints, session.decoder_config)
    if outcome.status != "ok":
        raise EventTrailError(
            "CONSTRAINT_CONFLICT", f"Approve contradicts constraints: {outcome.status}"
        )

    diffs, indirect_changed, diff_ms = compute_candidate_diffs(
        session.event_ids,
        session.current_path,
        outcome.path,
        action_event_id=action.event_id,
    )

    transition = TrailTransition(
        action_event_id=action.event_id,
        direct_changed_event_ids=(action.event_id,),
        indirect_changed_event_ids=indirect_changed,
        candidate_diffs=diffs,
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
    total_ms = (perf_counter() - started) * 1000.0

    log_trail_event(
        event_type="trail_approve",
        kis_revision=session.kis_revision,
        snapshot_id=session.snapshot_id,
        result_id=session.result_id,
        video_id=session.video_id,
        trail_session_id=session.session_id,
        trail_revision=updated.trail_revision,
        event_id=action.event_id,
        search_session_id=session.search_session_id,
        payload={
            "path_before": list(session.current_path.frame_ids),
            "path_after": list(outcome.path.frame_ids),  # type: ignore[union-attr]
            "direct_changed_event_ids": list(transition.direct_changed_event_ids),
            "indirect_changed_event_ids": list(transition.indirect_changed_event_ids),
            "outcome": "active",
            "constraint_ms": round(outcome.constraint_ms, 3),
            "dp_ms": round(outcome.dp_ms, 3),
            "diff_ms": round(diff_ms, 3),
            "total_ms": round(total_ms, 3),
        },
    )
    return build_trail_view(updated, transition)


def apply_use_frame(
    slot: SessionSlot,
    action: UseFrame,
    event_idx: int,
    decoder: TemporalConstraintDecoder,
    started: float,
) -> TrailView:
    """Set an explicit frame as anchor and select it as submission candidate."""
    session = slot.session
    if session.constraints.anchors[event_idx] is not None:
        raise EventTrailError("CONSTRAINT_CONFLICT", f"Event {action.event_id} is already anchored")

    indices = np.where(session.video_evidence.frame_ids == action.frame_id)[0]
    if len(indices) == 0:
        raise EventTrailError(
            "INVALID_FRAME",
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

    outcome = decoder.decode(session.video_evidence, new_constraints, session.decoder_config)
    if outcome.status != "ok":
        raise EventTrailError(
            "CONSTRAINT_CONFLICT", f"UseFrame contradicts constraints: {outcome.status}"
        )

    diffs, indirect_changed, diff_ms = compute_candidate_diffs(
        session.event_ids,
        session.current_path,
        outcome.path,
        action_event_id=action.event_id,
    )

    transition = TrailTransition(
        action_event_id=action.event_id,
        direct_changed_event_ids=(action.event_id,),
        indirect_changed_event_ids=indirect_changed,
        candidate_diffs=diffs,
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
    total_ms = (perf_counter() - started) * 1000.0

    log_trail_event(
        event_type="trail_use",
        kis_revision=session.kis_revision,
        snapshot_id=session.snapshot_id,
        result_id=session.result_id,
        video_id=session.video_id,
        trail_session_id=session.session_id,
        trail_revision=updated.trail_revision,
        event_id=action.event_id,
        search_session_id=session.search_session_id,
        payload={
            "path_before": list(session.current_path.frame_ids) if session.current_path else [],
            "path_after": list(outcome.path.frame_ids),  # type: ignore[union-attr]
            "direct_changed_event_ids": list(transition.direct_changed_event_ids),
            "indirect_changed_event_ids": list(transition.indirect_changed_event_ids),
            "outcome": "active",
            "constraint_ms": round(outcome.constraint_ms, 3),
            "dp_ms": round(outcome.dp_ms, 3),
            "diff_ms": round(diff_ms, 3),
            "total_ms": round(total_ms, 3),
        },
    )
    log_trail_event(
        event_type="submission_select",
        kis_revision=session.kis_revision,
        snapshot_id=session.snapshot_id,
        result_id=session.result_id,
        video_id=session.video_id,
        trail_session_id=session.session_id,
        trail_revision=updated.trail_revision,
        event_id=action.event_id,
        search_session_id=session.search_session_id,
        payload={
            "event_id": action.event_id,
            "frame_id": action.frame_id,
            "frame_idx": resolved_idx,
            "timestamp_ms": resolved_ts,
        },
    )
    return build_trail_view(updated, transition)


def apply_decline(
    slot: SessionSlot,
    action: DeclineCandidate,
    event_idx: int,
    decoder: TemporalConstraintDecoder,
    started: float,
) -> TrailView:
    """Decline current candidate frame by rejecting its derived midpoint interval."""
    session = slot.session
    if session.current_path is None:
        raise EventTrailError("CONSTRAINT_CONFLICT", "Cannot decline when path is exhausted")
    if session.constraints.anchors[event_idx] is not None:
        raise EventTrailError("CONSTRAINT_CONFLICT", f"Cannot decline anchored event {action.event_id}")

    current_fid = session.current_path.frame_ids[event_idx]
    cell = decoder.rejection_cell(session.video_evidence, current_fid)

    new_rejected = list(session.constraints.rejected_cells)
    new_rejected[event_idx] = session.constraints.rejected_cells[event_idx] + (cell,)
    new_constraints = replace(session.constraints, rejected_cells=tuple(new_rejected))

    outcome = decoder.decode(session.video_evidence, new_constraints, session.decoder_config)

    t_diff = perf_counter()
    if outcome.status == "ok":
        new_path = outcome.path
        new_status = "active"
        new_last_valid = outcome.path
        diffs_tuple, indirect_changed, _ = compute_candidate_diffs(
            session.event_ids,
            session.current_path,
            outcome.path,
            action_event_id=action.event_id,
        )
        transition = TrailTransition(
            action_event_id=action.event_id,
            direct_changed_event_ids=(action.event_id,),
            indirect_changed_event_ids=indirect_changed,
            candidate_diffs=diffs_tuple,
            latency_ms=outcome.constraint_ms + outcome.dp_ms,
        )
    else:
        new_path = None
        new_status = "exhausted"
        new_last_valid = session.last_valid_path
        old_fid = session.current_path.frame_ids[event_idx]
        old_ts = int(session.current_path.timestamps_ms[event_idx])
        single_diff = [
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
            candidate_diffs=tuple(single_diff),
            latency_ms=outcome.constraint_ms + outcome.dp_ms,
        )
    diff_ms = (perf_counter() - t_diff) * 1000.0

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
    total_ms = (perf_counter() - started) * 1000.0

    log_trail_event(
        event_type="trail_decline",
        kis_revision=session.kis_revision,
        snapshot_id=session.snapshot_id,
        result_id=session.result_id,
        video_id=session.video_id,
        trail_session_id=session.session_id,
        trail_revision=updated.trail_revision,
        event_id=action.event_id,
        search_session_id=session.search_session_id,
        payload={
            "path_before": list(session.current_path.frame_ids),
            "path_after": list(new_path.frame_ids) if new_path else None,
            "direct_changed_event_ids": list(transition.direct_changed_event_ids),
            "indirect_changed_event_ids": list(transition.indirect_changed_event_ids),
            "outcome": new_status,
            "constraint_ms": round(outcome.constraint_ms, 3),
            "dp_ms": round(outcome.dp_ms, 3),
            "diff_ms": round(diff_ms, 3),
            "total_ms": round(total_ms, 3),
        },
    )
    if new_status == "exhausted" and session.status != "exhausted":
        log_trail_event(
            event_type="trail_exhausted",
            kis_revision=session.kis_revision,
            snapshot_id=session.snapshot_id,
            result_id=session.result_id,
            video_id=session.video_id,
            trail_session_id=session.session_id,
            trail_revision=updated.trail_revision,
            event_id=action.event_id,
            search_session_id=session.search_session_id,
            payload={
                "exhausted_by_event_id": action.event_id,
                "last_valid_path": list(new_last_valid.frame_ids) if new_last_valid else [],
                "total_ms": round(total_ms, 3),
            },
        )
    return build_trail_view(updated, transition)


def apply_clear_anchor(
    slot: SessionSlot,
    action: ClearAnchor,
    event_idx: int,
    decoder: TemporalConstraintDecoder,
    started: float,
) -> TrailView:
    """Clear pinned anchor at event_idx and re-decode."""
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

    outcome = decoder.decode(session.video_evidence, new_constraints, session.decoder_config)
    if outcome.status == "ok":
        new_path = outcome.path
        new_status = "active"
        new_last_valid = outcome.path
    else:
        new_path = None
        new_status = "exhausted"
        new_last_valid = session.last_valid_path

    diffs, indirect_changed, diff_ms = compute_candidate_diffs(
        session.event_ids,
        session.current_path,
        new_path,
        action_event_id=action.event_id,
    )

    transition = TrailTransition(
        action_event_id=action.event_id,
        direct_changed_event_ids=(action.event_id,),
        indirect_changed_event_ids=indirect_changed,
        candidate_diffs=diffs,
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
    total_ms = (perf_counter() - started) * 1000.0

    log_trail_event(
        event_type="trail_clear_anchor",
        kis_revision=session.kis_revision,
        snapshot_id=session.snapshot_id,
        result_id=session.result_id,
        video_id=session.video_id,
        trail_session_id=session.session_id,
        trail_revision=updated.trail_revision,
        event_id=action.event_id,
        search_session_id=session.search_session_id,
        payload={
            "path_before": list(session.current_path.frame_ids) if session.current_path else [],
            "path_after": list(new_path.frame_ids) if new_path else None,
            "direct_changed_event_ids": list(transition.direct_changed_event_ids),
            "indirect_changed_event_ids": list(transition.indirect_changed_event_ids),
            "outcome": new_status,
            "constraint_ms": round(outcome.constraint_ms, 3),
            "dp_ms": round(outcome.dp_ms, 3),
            "diff_ms": round(diff_ms, 3),
            "total_ms": round(total_ms, 3),
        },
    )
    if new_status == "exhausted" and session.status != "exhausted":
        log_trail_event(
            event_type="trail_exhausted",
            kis_revision=session.kis_revision,
            snapshot_id=session.snapshot_id,
            result_id=session.result_id,
            video_id=session.video_id,
            trail_session_id=session.session_id,
            trail_revision=updated.trail_revision,
            event_id=action.event_id,
            search_session_id=session.search_session_id,
            payload={
                "exhausted_by_event_id": action.event_id,
                "last_valid_path": list(new_last_valid.frame_ids) if new_last_valid else [],
                "total_ms": round(total_ms, 3),
            },
        )
    return build_trail_view(updated, transition)


def apply_set_window(
    slot: SessionSlot,
    action: SetWindow,
    decoder: TemporalConstraintDecoder,
    started: float,
) -> TrailView:
    """Set global temporal interval window [start_ms, end_ms] and re-decode."""
    session = slot.session
    if action.start_ms < 0 or action.start_ms > action.end_ms:
        raise EventTrailError("INVALID_WINDOW", "Window start must be <= end and >= 0")

    new_constraints = replace(session.constraints, window=(action.start_ms, action.end_ms))
    outcome = decoder.decode(session.video_evidence, new_constraints, session.decoder_config)
    if outcome.status != "ok":
        raise EventTrailError(
            "CONSTRAINT_CONFLICT", f"SetWindow contradicts constraints: {outcome.status}"
        )

    diffs, indirect_changed, diff_ms = compute_candidate_diffs(
        session.event_ids,
        session.current_path,
        outcome.path,
        action_event_id=None,
    )

    transition = TrailTransition(
        action_event_id=None,
        direct_changed_event_ids=(),
        indirect_changed_event_ids=indirect_changed,
        candidate_diffs=diffs,
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
    total_ms = (perf_counter() - started) * 1000.0

    log_trail_event(
        event_type="trail_window",
        kis_revision=session.kis_revision,
        snapshot_id=session.snapshot_id,
        result_id=session.result_id,
        video_id=session.video_id,
        trail_session_id=session.session_id,
        trail_revision=updated.trail_revision,
        event_id=None,
        search_session_id=session.search_session_id,
        payload={
            "operation": "set",
            "start_ms": action.start_ms,
            "end_ms": action.end_ms,
            "path_before": list(session.current_path.frame_ids) if session.current_path else [],
            "path_after": list(outcome.path.frame_ids),  # type: ignore[union-attr]
            "direct_changed_event_ids": list(transition.direct_changed_event_ids),
            "indirect_changed_event_ids": list(transition.indirect_changed_event_ids),
            "outcome": "active",
            "constraint_ms": round(outcome.constraint_ms, 3),
            "dp_ms": round(outcome.dp_ms, 3),
            "diff_ms": round(diff_ms, 3),
            "total_ms": round(total_ms, 3),
        },
    )
    return build_trail_view(updated, transition)


def apply_clear_window(
    slot: SessionSlot,
    decoder: TemporalConstraintDecoder,
    started: float,
) -> TrailView:
    """Clear global temporal window constraint and re-decode."""
    session = slot.session
    new_constraints = replace(session.constraints, window=None)
    outcome = decoder.decode(session.video_evidence, new_constraints, session.decoder_config)
    if outcome.status == "ok":
        new_path = outcome.path
        new_status = "active"
        new_last_valid = outcome.path
    else:
        new_path = None
        new_status = "exhausted"
        new_last_valid = session.last_valid_path

    diffs, indirect_changed, diff_ms = compute_candidate_diffs(
        session.event_ids,
        session.current_path,
        new_path,
        action_event_id=None,
    )

    transition = TrailTransition(
        action_event_id=None,
        direct_changed_event_ids=(),
        indirect_changed_event_ids=indirect_changed,
        candidate_diffs=diffs,
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
    total_ms = (perf_counter() - started) * 1000.0

    log_trail_event(
        event_type="trail_window",
        kis_revision=session.kis_revision,
        snapshot_id=session.snapshot_id,
        result_id=session.result_id,
        video_id=session.video_id,
        trail_session_id=session.session_id,
        trail_revision=updated.trail_revision,
        event_id=None,
        search_session_id=session.search_session_id,
        payload={
            "operation": "clear",
            "path_before": list(session.current_path.frame_ids) if session.current_path else [],
            "path_after": list(new_path.frame_ids) if new_path else None,
            "direct_changed_event_ids": list(transition.direct_changed_event_ids),
            "indirect_changed_event_ids": list(transition.indirect_changed_event_ids),
            "outcome": new_status,
            "constraint_ms": round(outcome.constraint_ms, 3),
            "dp_ms": round(outcome.dp_ms, 3),
            "diff_ms": round(diff_ms, 3),
            "total_ms": round(total_ms, 3),
        },
    )
    if new_status == "exhausted" and session.status != "exhausted":
        log_trail_event(
            event_type="trail_exhausted",
            kis_revision=session.kis_revision,
            snapshot_id=session.snapshot_id,
            result_id=session.result_id,
            video_id=session.video_id,
            trail_session_id=session.session_id,
            trail_revision=updated.trail_revision,
            event_id=None,
            search_session_id=session.search_session_id,
            payload={
                "exhausted_by_event_id": None,
                "last_valid_path": list(new_last_valid.frame_ids) if new_last_valid else [],
                "total_ms": round(total_ms, 3),
            },
        )
    return build_trail_view(updated, transition)
