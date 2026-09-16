"""Tests for EventTrail HTTP contracts and API endpoints."""

from datetime import datetime, timezone
from unittest.mock import Mock
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hcmai.api.contracts.event_trail import (
    ApproveAction,
    ClearAnchorAction,
    ClearWindowAction,
    DeclineAction,
    EventTrailActionRequest,
    EventTrailOpenRequest,
    EventTrailStateResponse,
    SetWindowAction,
    UndoAction,
    UseFrameAction,
)
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
from hcmai.orchestration.workflows.temporal_search import DecoderConfigSnapshot
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import AlignedPath


# ---- Step 1: Contract validation tests ----

def test_contract_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        EventTrailOpenRequest.model_validate({
            "snapshot_id": "snap_1",
            "result_id": "r_1",
            "expected_kis_revision": 1,
            "extra_field": 123,
        })


def test_contract_set_window_validation() -> None:
    with pytest.raises(ValidationError):
        SetWindowAction(start_ms=2000, end_ms=1000)

    valid = SetWindowAction(start_ms=1000, end_ms=2000)
    assert valid.start_ms == 1000
    assert valid.end_ms == 2000


def test_contract_action_discriminator() -> None:
    req = EventTrailActionRequest.model_validate({
        "expected_trail_revision": 0,
        "action": {"type": "approve", "event_id": "E1"},
    })
    assert isinstance(req.action, ApproveAction)
    assert req.action.event_id == "E1"

    req_use = EventTrailActionRequest.model_validate({
        "expected_trail_revision": 0,
        "action": {"type": "use_frame", "event_id": "E1", "frame_id": "f1"},
    })
    assert isinstance(req_use.action, UseFrameAction)
    assert req_use.action.frame_id == "f1"


def test_contract_use_frame_requires_both_fields() -> None:
    with pytest.raises(ValidationError):
        EventTrailActionRequest.model_validate({
            "expected_trail_revision": 0,
            "action": {"type": "use_frame", "event_id": "E1"},
        })


# ---- Step 2: Route lifecycle & error mapping tests ----

class FakeTemporalSearchService:
    def decode_video(self, video: VideoEventScores, *, allowed: np.ndarray, decoder_config=None):
        n_events, n_frames = allowed.shape
        chosen = []
        last_t = -1
        for e in range(n_events):
            picked = None
            for f in range(n_frames):
                if allowed[e, f] and video.timestamps_ms[f] > last_t:
                    picked = f
                    break
            if picked is None:
                return ()
            chosen.append(picked)
            last_t = video.timestamps_ms[picked]

        return (
            AlignedPath(
                video_id=video.video_id,
                score=1.0,
                frame_ids=tuple(str(video.frame_ids[i]) for i in chosen),
                frame_idxs=tuple(int(video.frame_idx[i]) for i in chosen),
                timestamps_ms=tuple(int(video.timestamps_ms[i]) for i in chosen),
            ),
        )


@pytest.fixture
def mock_search_service():
    service = Mock()
    snapshots = EvidenceSnapshotStore(ttl_seconds=900, max_entries=16)
    sessions = EventTrailSessionStore(ttl_seconds=1800, max_entries=16)
    decoder = TemporalConstraintDecoder(FakeTemporalSearchService())  # type: ignore[arg-type]
    event_trail = EventTrailService(snapshots, sessions, decoder)

    video = VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["f1", "f2", "f3", "f4"]),
        frame_idx=np.array([1, 2, 3, 4]),
        timestamps_ms=np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        scores=np.array([
            [0.9, 0.4, 0.1, 0.1],
            [0.1, 0.8, 0.5, 0.2],
            [0.1, 0.2, 0.8, 0.9],
        ], dtype=np.float32),
    )
    now = datetime.now(timezone.utc)
    snap = EvidenceSnapshot(
        snapshot_id="snap_test",
        kis_revision=1,
        scoring_revision="score_rev_1",
        event_ids=("E1", "E2", "E3"),
        decoder_config=DecoderConfigSnapshot(0.5, 1.0, 0.0, 0),
        results={
            "r_1": SnapshotResult("r_1", "v1", ("f1", "f2", "f4"), 2.6)
        },
        video_evidence={"v1": freeze_video_scores(video)},
        created_at=now,
        expires_at=now,
    )
    snapshots.put(snap)

    service.event_trail = event_trail
    return service


@pytest.fixture
def client(mock_search_service) -> TestClient:
    app = create_app(search_service=mock_search_service)
    return TestClient(app)


def test_event_trail_route_lifecycle(client: TestClient) -> None:
    # 1. Open
    open_res = client.post(
        "/api/v1/event-trail/open",
        json={"snapshot_id": "snap_test", "result_id": "r_1", "expected_kis_revision": 1},
    )
    assert open_res.status_code == 200, open_res.text
    data = open_res.json()
    session_id = data["session_id"]
    assert data["trail_revision"] == 0
    assert len(data["path"]) == 3

    # 2. GET state
    get_res = client.get(f"/api/v1/event-trail/{session_id}")
    assert get_res.status_code == 200
    assert get_res.json()["trail_revision"] == 0

    # 3. Approve E1
    appr_res = client.post(
        f"/api/v1/event-trail/{session_id}/actions",
        json={"expected_trail_revision": 0, "action": {"type": "approve", "event_id": "E1"}},
    )
    assert appr_res.status_code == 200
    assert appr_res.json()["trail_revision"] == 1
    assert "E1" in appr_res.json()["approved_event_ids"]

    # 4. Decline E2
    dec_res = client.post(
        f"/api/v1/event-trail/{session_id}/actions",
        json={"expected_trail_revision": 1, "action": {"type": "decline", "event_id": "E2"}},
    )
    assert dec_res.status_code == 200
    assert dec_res.json()["trail_revision"] == 2

    # 5. Stale revision conflict (expecting 1 instead of 2)
    stale_res = client.post(
        f"/api/v1/event-trail/{session_id}/actions",
        json={"expected_trail_revision": 1, "action": {"type": "approve", "event_id": "E3"}},
    )
    assert stale_res.status_code == 409
    assert stale_res.json()["detail"]["code"] == "TRAIL_REVISION_CONFLICT"


@pytest.mark.parametrize(
    "code,expected_status",
    [
        ("SNAPSHOT_NOT_FOUND", 404),
        ("RESULT_NOT_FOUND", 404),
        ("TRAIL_SESSION_NOT_FOUND", 404),
        ("KIS_REVISION_MISMATCH", 409),
        ("TRAIL_REVISION_CONFLICT", 409),
        ("CONSTRAINT_CONFLICT", 409),
        ("SNAPSHOT_EXPIRED", 410),
        ("TRAIL_SESSION_EXPIRED", 410),
        ("INVALID_EVENT", 422),
        ("INVALID_FRAME", 422),
        ("INVALID_WINDOW", 422),
        ("NOTHING_TO_UNDO", 422),
        ("EVENT_TRAIL_UNAVAILABLE", 503),
    ],
)
def test_error_status_mapping(mock_search_service, client: TestClient, code: str, expected_status: int) -> None:
    # Force service method to raise EventTrailError with this code
    def raise_err(*args, **kwargs):
        raise EventTrailError(code, f"Simulated error: {code}")

    mock_search_service.event_trail.open = raise_err

    res = client.post(
        "/api/v1/event-trail/open",
        json={"snapshot_id": "snap_1", "result_id": "r_1", "expected_kis_revision": 1},
    )
    assert res.status_code == expected_status
    assert res.json()["detail"]["code"] == code


def test_missing_event_trail_service_returns_503(mock_search_service, client: TestClient) -> None:
    mock_search_service.event_trail = None
    res = client.post(
        "/api/v1/event-trail/open",
        json={"snapshot_id": "snap_1", "result_id": "r_1", "expected_kis_revision": 1},
    )
    assert res.status_code == 503
    assert res.json()["detail"]["message"] == "EventTrail is unavailable"
