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
    EvidenceSnapshot,
    SnapshotResult,
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
        return (
            AlignedPath(
                video_id=video.video_id,
                score=1.0,
                frame_ids=("f1", "f2"),
                frame_idxs=(10, 20),
                timestamps_ms=(1000, 2000),
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
