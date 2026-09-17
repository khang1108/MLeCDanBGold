"""Tests verifying search_session_id correlation in EventTrail structured logs."""

from datetime import datetime, timezone
import json
import logging
from unittest.mock import Mock
import numpy as np
import pytest

from hcmai.event_trail.decoder import TemporalConstraintDecoder
from hcmai.event_trail.models import (
    ApproveEvent,
    DeclineCandidate,
    EvidenceSnapshot,
    SnapshotResult,
    Undo,
    UseFrame,
    freeze_video_scores,
)
from hcmai.event_trail.service import EventTrailService
from hcmai.event_trail.store import (
    EventTrailSessionStore,
    EvidenceSnapshotStore,
)
from hcmai.orchestration.workflows.temporal_search import DecoderConfigSnapshot
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import AlignedPath


class FakeTemporalSearchService:
    def decode_video(self, video: VideoEventScores, *, allowed: np.ndarray, decoder_config=None):
        e1_idx = 0
        e2_idx = 2 if len(video.frame_ids) > 2 and not allowed[1, 1] else 1
        return (
            AlignedPath(
                video_id=video.video_id,
                score=1.0,
                frame_ids=(str(video.frame_ids[e1_idx]), str(video.frame_ids[e2_idx])),
                frame_idxs=(int(video.frame_idx[e1_idx]), int(video.frame_idx[e2_idx])),
                timestamps_ms=(int(video.timestamps_ms[e1_idx]), int(video.timestamps_ms[e2_idx])),
            ),
        )


def test_search_session_logging_correlation(caplog) -> None:
    caplog.set_level(logging.INFO, logger="hcmai.event_trail.interactions")

    snapshots = EvidenceSnapshotStore(ttl_seconds=900, max_entries=16)
    sessions = EventTrailSessionStore(ttl_seconds=1800, max_entries=16)
    decoder = TemporalConstraintDecoder(FakeTemporalSearchService())  # type: ignore[arg-type]
    service = EventTrailService(snapshots, sessions, decoder)

    video = VideoEventScores(
        video_id="V01",
        frame_ids=np.array(["f1", "f2"]),
        frame_idx=np.array([10, 20]),
        timestamps_ms=np.array([1000, 2000]),
        scores=np.ones((2, 2)),
    )
    snapshot = EvidenceSnapshot(
        snapshot_id="snap_1",
        kis_revision=1,
        scoring_revision="rev_1",
        event_ids=("E1", "E2"),
        decoder_config=DecoderConfigSnapshot(
            cluster_delta=0.0,
            event_power=1.0,
            lambda_gap=1e-5,
            path_min_separation_ms=0,
        ),
        results={
            "r_1": SnapshotResult(
                result_id="r_1",
                video_id="V01",
                initial_path=("f1", "f2"),
                path_score=1.0,
            ),
        },
        video_evidence={"V01": freeze_video_scores(video)},
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc),
    )
    snapshots.put(snapshot)

    # 1. Open with correlation ID
    view = service.open("snap_1", "r_1", expected_kis_revision=1, search_session_id="query_123")
    session_id = view.session_id

    # 2. Action (Approve E1)
    service.act(session_id, expected_trail_revision=0, action=ApproveEvent(event_id="E1"))

    # 3. Close
    service.close(session_id, expected_trail_revision=1)

    records = [
        json.loads(r.message)
        for r in caplog.records
        if r.name == "hcmai.event_trail.interactions"
    ]
    assert len(records) >= 3

    event_types = [rec["type"] for rec in records]
    assert "trail_open" in event_types
    assert "trail_approve" in event_types
    assert "trail_close" in event_types

    for rec in records:
        assert rec["search_session_id"] == "query_123"
        assert rec["snapshot_id"] == "snap_1"
        assert rec["result_id"] == "r_1"
        assert rec["trail_session_id"] == session_id


def test_end_to_end_correlation_lifecycle(caplog) -> None:
    caplog.set_level(logging.INFO, logger="hcmai.event_trail.interactions")

    snapshots = EvidenceSnapshotStore(ttl_seconds=900, max_entries=16)
    sessions = EventTrailSessionStore(ttl_seconds=1800, max_entries=16)
    decoder = TemporalConstraintDecoder(FakeTemporalSearchService())  # type: ignore[arg-type]
    service = EventTrailService(snapshots, sessions, decoder)

    video = VideoEventScores(
        video_id="V01",
        frame_ids=np.array(["f1", "f2", "f3"]),
        frame_idx=np.array([10, 20, 30]),
        timestamps_ms=np.array([1000, 2000, 3000]),
        scores=np.ones((3, 2)),
    )
    snapshot = EvidenceSnapshot(
        snapshot_id="snap_1",
        kis_revision=1,
        scoring_revision="rev_1",
        event_ids=("E1", "E2"),
        decoder_config=DecoderConfigSnapshot(
            cluster_delta=0.0,
            event_power=1.0,
            lambda_gap=1e-5,
            path_min_separation_ms=0,
        ),
        results={
            "r_1": SnapshotResult(
                result_id="r_1",
                video_id="V01",
                initial_path=("f1", "f2"),
                path_score=1.0,
            ),
        },
        video_evidence={"V01": freeze_video_scores(video)},
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc),
    )
    snapshots.put(snapshot)

    # 1. Open -> trail_open (rev 0)
    view = service.open("snap_1", "r_1", expected_kis_revision=1, search_session_id="query_123")
    session_id = view.session_id
    assert view.trail_revision == 0

    # 2. Approve -> trail_approve (rev 1)
    view = service.act(session_id, expected_trail_revision=0, action=ApproveEvent(event_id="E1"))
    assert view.trail_revision == 1

    # 3. Decline -> trail_decline (rev 2)
    view = service.act(session_id, expected_trail_revision=1, action=DeclineCandidate(event_id="E2"))
    assert view.trail_revision == 2

    # 4. Use -> trail_use + submission_select (rev 3)
    view = service.act(session_id, expected_trail_revision=2, action=UseFrame(event_id="E2", frame_id="f3"))
    assert view.trail_revision == 3

    # 5. Undo -> trail_undo (rev 4)
    view = service.act(session_id, expected_trail_revision=3, action=Undo())
    assert view.trail_revision == 4

    # 6. Close -> trail_close (rev 4)
    service.close(session_id, expected_trail_revision=4)

    records = [
        json.loads(r.message)
        for r in caplog.records
        if r.name == "hcmai.event_trail.interactions"
    ]
    expected_types = [
        "trail_open",
        "trail_approve",
        "trail_decline",
        "trail_use",
        "submission_select",
        "trail_undo",
        "trail_close",
    ]
    actual_types = [r["type"] for r in records]
    assert actual_types == expected_types

    for r in records:
        assert r["search_session_id"] == "query_123"
        assert r["snapshot_id"] == "snap_1"
        assert r["result_id"] == "r_1"
        assert r["trail_session_id"] == session_id

    revisions = [r["trail_revision"] for r in records]
    assert revisions == [0, 1, 2, 3, 3, 4, 4]
    for i in range(len(revisions) - 1):
        assert revisions[i] <= revisions[i + 1]

