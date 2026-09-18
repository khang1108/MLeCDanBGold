"""Tests for explicit EventTrail session close lifecycle and HTTP DELETE route."""

from datetime import datetime, timezone
from unittest.mock import Mock
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hcmai.app import create_app
from hcmai.event_trail.decoder import TemporalConstraintDecoder
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import (
    EvidenceSnapshot,
    SnapshotResult,
    freeze_video_scores,
)
from hcmai.event_trail.service import EventTrailService
from hcmai.event_trail.store import (
    EventTrailSessionStore,
    EvidenceSnapshotStore,
)
from hcmai.orchestration.workflows.search.temporal import DecoderConfigSnapshot
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import AlignedPath


class FakeTemporalSearchService:
    def decode_video(self, video: VideoEventScores, *, allowed: np.ndarray, decoder_config=None):
        return (
            AlignedPath(
                video_id=video.video_id,
                score=1.0,
                frame_ids=tuple(str(video.frame_ids[i]) for i in range(2)),
                frame_idxs=(10, 20),
                timestamps_ms=(1000, 2000),
            ),
        )


def _seed_snapshot(snapshots: EvidenceSnapshotStore, snapshot_id: str = "snap_1") -> None:
    video = VideoEventScores(
        video_id="V01",
        frame_ids=np.array(["f1", "f2"]),
        frame_idx=np.array([10, 20]),
        timestamps_ms=np.array([1000, 2000]),
        scores=np.ones((2, 2)),
    )
    snapshot = EvidenceSnapshot(
        snapshot_id=snapshot_id,
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


@pytest.fixture
def service_and_client():
    snapshots = EvidenceSnapshotStore(ttl_seconds=900, max_entries=16)
    sessions = EventTrailSessionStore(ttl_seconds=1800, max_entries=16)
    decoder = TemporalConstraintDecoder(FakeTemporalSearchService())  # type: ignore[arg-type]
    service = EventTrailService(snapshots, sessions, decoder)

    mock_search_service = Mock(event_trail=service)
    app = create_app(search_service=mock_search_service)
    client = TestClient(app)
    return service, snapshots, client


def test_service_close_removes_session(service_and_client) -> None:
    service, snapshots, _ = service_and_client
    _seed_snapshot(snapshots)
    view = service.open("snap_1", "r_1", expected_kis_revision=1)
    session_id = view.session_id

    # Session exists
    assert service.get(session_id).session_id == session_id

    # Close session at revision 0
    service.close(session_id, expected_trail_revision=0)

    # Session is now gone
    with pytest.raises(EventTrailError) as exc:
        service.get(session_id)
    assert exc.value.code == "TRAIL_SESSION_NOT_FOUND"


def test_service_close_conflict_preserves_session(service_and_client) -> None:
    service, snapshots, _ = service_and_client
    _seed_snapshot(snapshots)
    view = service.open("snap_1", "r_1", expected_kis_revision=1)
    session_id = view.session_id

    # Stale expected revision 99
    with pytest.raises(EventTrailError) as exc:
        service.close(session_id, expected_trail_revision=99)
    assert exc.value.code == "TRAIL_REVISION_CONFLICT"

    # Session still exists and has original revision
    current = service.get(session_id)
    assert current.session_id == session_id
    assert current.trail_revision == 0


def test_delete_endpoint_closes_session(service_and_client) -> None:
    service, snapshots, client = service_and_client
    _seed_snapshot(snapshots)
    view = service.open("snap_1", "r_1", expected_kis_revision=1)
    session_id = view.session_id

    # DELETE route returns 204
    resp = client.delete(f"/api/v1/event-trail/{session_id}?expected_trail_revision=0")
    assert resp.status_code == 204

    # Subsequent GET returns 404
    get_resp = client.get(f"/api/v1/event-trail/{session_id}")
    assert get_resp.status_code == 404
    assert get_resp.json()["detail"]["code"] == "TRAIL_SESSION_NOT_FOUND"


def test_delete_endpoint_revision_conflict(service_and_client) -> None:
    service, snapshots, client = service_and_client
    _seed_snapshot(snapshots)
    view = service.open("snap_1", "r_1", expected_kis_revision=1)
    session_id = view.session_id

    resp = client.delete(f"/api/v1/event-trail/{session_id}?expected_trail_revision=5")
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "TRAIL_REVISION_CONFLICT"
