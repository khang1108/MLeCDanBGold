"""Candidate diff computation and state projection for EventTrail.

This module provides pure transformations to compute candidate differences
between path states, materialize snapshot paths, and project session state into TrailView.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np

from hcmai.event_trail.models import (
    CandidateDiff,
    EventCandidate,
    EventTrailSession,
    SnapshotResult,
    TrailTransition,
    TrailView,
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


def compute_candidate_diffs(
    event_ids: Sequence[str],
    old_path: AlignedPath | None,
    new_path: AlignedPath | None,
    action_event_id: str | None = None,
) -> tuple[tuple[CandidateDiff, ...], tuple[str, ...], float]:
    """Compute candidate differences and indirect changed event IDs between two paths.

    Returns:
        (diffs, indirect_changed_event_ids, diff_computation_ms)
    """
    t_start = perf_counter()
    diffs: list[CandidateDiff] = []
    indirect_changed: list[str] = []

    for i, eid in enumerate(event_ids):
        old_fid = old_path.frame_ids[i] if old_path is not None else None
        new_fid = new_path.frame_ids[i] if new_path is not None else None
        old_ts = int(old_path.timestamps_ms[i]) if old_path is not None else None
        new_ts = int(new_path.timestamps_ms[i]) if new_path is not None else None

        if old_fid != new_fid:
            diffs.append(CandidateDiff(eid, old_fid, new_fid, old_ts, new_ts))
            if action_event_id is None or eid != action_event_id:
                indirect_changed.append(eid)

    diff_ms = (perf_counter() - t_start) * 1000.0
    return tuple(diffs), tuple(indirect_changed), diff_ms


def build_trail_view(
    session: EventTrailSession,
    transition: TrailTransition | None = None,
) -> TrailView:
    """Project current session state and optional transition into a client-facing TrailView."""
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
