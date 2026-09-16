"""Tests for EventTrailService and session action semantics."""

from datetime import datetime, timezone
import json
import logging
import numpy as np
import pytest

from hcmai.event_trail.config import EventTrailSettings
from hcmai.event_trail.decoder import TemporalConstraintDecoder
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import (
    ApproveEvent,
    ClearAnchor,
    ClearWindow,
    DeclineCandidate,
    EvidenceSnapshot,
    SetWindow,
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


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self._current = start

    def __call__(self) -> float:
        return self._current

    def advance(self, seconds: float) -> None:
        self._current += seconds


class Stores:
    def __init__(self, clock: FakeClock) -> None:
        self.snapshots = EvidenceSnapshotStore(ttl_seconds=900, max_entries=4, clock=clock)
        self.sessions = EventTrailSessionStore(ttl_seconds=1800, max_entries=4, clock=clock)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def stores(clock: FakeClock) -> Stores:
    return Stores(clock)


@pytest.fixture
def sample_video() -> VideoEventScores:
    return VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["f1", "f2", "f3", "f4"]),
        frame_idx=np.array([1, 2, 3, 4]),
        timestamps_ms=np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        scores=np.array([
            [0.9, 0.4, 0.1, 0.1],  # E1 scores
            [0.1, 0.8, 0.5, 0.2],  # E2 scores
            [0.1, 0.2, 0.8, 0.9],  # E3 scores
        ], dtype=np.float32),
    )


@pytest.fixture
def snapshot(sample_video: VideoEventScores) -> EvidenceSnapshot:
    frozen = freeze_video_scores(sample_video)
    now = datetime.now(timezone.utc)
    return EvidenceSnapshot(
        snapshot_id="snap_1",
        kis_revision=1,
        scoring_revision="score_gen_1",
        event_ids=("E1", "E2", "E3"),
        decoder_config=DecoderConfigSnapshot(
            lambda_gap=0.5,
            event_power=1.0,
            cluster_delta=0.0,
            path_min_separation_ms=0,
        ),
        results={
            "r_1": SnapshotResult(
                result_id="r_1",
                video_id="v1",
                initial_path=("f1", "f2", "f4"),
                path_score=2.6,
            )
        },
        video_evidence={"v1": frozen},
        created_at=now,
        expires_at=now,
    )


class FakeTemporalSearchService:
    """Minimal fake temporal service for decoding."""

    def decode_video(self, video: VideoEventScores, *, allowed: np.ndarray, decoder_config=None):
        # Find first allowed frame for each event strictly monotonically
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

    def score_plan(self, *args, **kwargs):
        raise NotImplementedError("score_plan should not be called")

    def search_plan_artifact(self, *args, **kwargs):
        raise NotImplementedError("search_plan_artifact should not be called")


@pytest.fixture
def service(stores: Stores) -> EventTrailService:
    decoder = TemporalConstraintDecoder(FakeTemporalSearchService())  # type: ignore[arg-type]
    return EventTrailService(
        snapshot_store=stores.snapshots,
        session_store=stores.sessions,
        decoder=decoder,
    )


def test_open_uses_exact_ranked_path_without_scoring_or_redecode(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    assert [row.frame_id for row in view.path] == list(snapshot.results["r_1"].initial_path)
    assert view.trail_revision == 0
    assert view.status == "active"
    assert view.approved_event_ids == ()
    assert view.window is None
    assert view.submission_selection is None


def test_open_session_survives_snapshot_eviction(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    stores.snapshots.remove(snapshot.snapshot_id)
    assert service.get(view.session_id).path == view.path


def test_session_expiration_is_fixed_not_sliding(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot, clock: FakeClock
) -> None:
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)

    clock.advance(1000)
    assert service.get(view.session_id).trail_revision == 0

    clock.advance(801)
    with pytest.raises(EventTrailError) as exc:
        service.get(view.session_id)
    assert exc.value.code == "TRAIL_SESSION_EXPIRED"


def test_capacity_eviction_returns_not_found(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    v1 = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    v2 = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    v3 = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    v4 = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    # 5th session evicts v1 by capacity
    v5 = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)

    with pytest.raises(EventTrailError) as exc:
        service.get(v1.session_id)
    assert exc.value.code == "TRAIL_SESSION_NOT_FOUND"


def test_approve_event_anchors_candidate(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    updated = service.act(view.session_id, expected_trail_revision=0, action=ApproveEvent("E2"))
    assert updated.trail_revision == 1
    assert "E2" in updated.approved_event_ids
    assert updated.path[1].frame_id == "f2"
    assert updated.transition is not None
    assert updated.transition.action_event_id == "E2"
    assert updated.transition.direct_changed_event_ids == ("E2",)


def test_use_frame_anchors_and_sets_submission_selection(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    updated = service.act(
        view.session_id, expected_trail_revision=0, action=UseFrame(event_id="E2", frame_id="f2")
    )
    assert updated.trail_revision == 1
    assert "E2" in updated.approved_event_ids
    assert updated.submission_selection is not None
    assert updated.submission_selection.event_id == "E2"
    assert updated.submission_selection.frame_id == "f2"
    assert updated.submission_selection.timestamp_ms == 2000


def test_decline_candidate_re_decodes(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    # Decline E2's current candidate (f2)
    updated = service.act(view.session_id, expected_trail_revision=0, action=DeclineCandidate("E2"))
    assert updated.trail_revision == 1
    assert updated.rejected_counts["E2"] == 1
    # E2 cannot be f2 now
    assert updated.path[1].frame_id != "f2"


def test_decline_anchored_event_raises_conflict(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    service.act(view.session_id, expected_trail_revision=0, action=ApproveEvent("E2"))

    with pytest.raises(EventTrailError) as exc:
        service.act(view.session_id, expected_trail_revision=1, action=DeclineCandidate("E2"))
    assert exc.value.code == "CONSTRAINT_CONFLICT"


def test_undo_restores_previous_state(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view0 = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    view1 = service.act(view0.session_id, expected_trail_revision=0, action=ApproveEvent("E2"))
    assert "E2" in view1.approved_event_ids

    view2 = service.act(view1.session_id, expected_trail_revision=1, action=Undo())
    assert view2.trail_revision == 2
    assert "E2" not in view2.approved_event_ids


def test_stale_revision_conflict(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    with pytest.raises(EventTrailError) as exc:
        service.act(view.session_id, expected_trail_revision=99, action=ApproveEvent("E1"))
    assert exc.value.code == "TRAIL_REVISION_CONFLICT"


def test_cannot_undo_empty_history(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    with pytest.raises(EventTrailError) as exc:
        service.act(view.session_id, expected_trail_revision=0, action=Undo())
    assert exc.value.code == "CANNOT_UNDO"


def test_clear_anchor_relaxes_anchor(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view0 = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    view1 = service.act(view0.session_id, expected_trail_revision=0, action=ApproveEvent("E2"))
    assert "E2" in view1.approved_event_ids

    view2 = service.act(view1.session_id, expected_trail_revision=1, action=ClearAnchor("E2"))
    assert view2.trail_revision == 2
    assert "E2" not in view2.approved_event_ids


def test_set_and_clear_window(
    service: EventTrailService, stores: Stores, snapshot: EvidenceSnapshot
) -> None:
    stores.snapshots.put(snapshot)
    view0 = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)
    view1 = service.act(view0.session_id, expected_trail_revision=0, action=SetWindow(1000, 4000))
    assert view1.window == (1000, 4000)

    view2 = service.act(view1.session_id, expected_trail_revision=1, action=ClearWindow())
    assert view2.window is None


def test_decline_and_exhausting_decline_logging(
    service: EventTrailService,
    stores: Stores,
    snapshot: EvidenceSnapshot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="hcmai.event_trail.interactions")
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)

    # 1. Normal Decline
    updated = service.act(view.session_id, expected_trail_revision=0, action=DeclineCandidate("E2"))
    assert updated.status == "active"

    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "hcmai.event_trail.interactions"
    ]
    # Check open record
    open_rec = [r for r in records if r["type"] == "trail_open"][0]
    assert open_rec["trail_session_id"] == view.session_id
    assert open_rec["trail_revision"] == 0
    assert open_rec["payload"]["initial_path"]
    assert open_rec["payload"]["total_ms"] >= 0

    # Check decline record
    decline_rec = [r for r in records if r["type"] == "trail_decline"][0]
    assert decline_rec["type"] == "trail_decline"
    assert decline_rec["event_id"] == "E2"
    assert decline_rec["payload"]["path_before"]
    assert "direct_changed_event_ids" in decline_rec["payload"]
    assert "indirect_changed_event_ids" in decline_rec["payload"]
    assert decline_rec["payload"]["total_ms"] >= 0
    assert decline_rec["payload"]["outcome"] == "active"
    assert decline_rec["payload"]["path_after"] is not None

    # 2. Exhausting decline: decline until exhausted
    # Decline E1 twice more so it pushes past available frames for subsequent events
    rev = updated.trail_revision
    exhausted_view = None
    for _ in range(5):
        try:
            res = service.act(view.session_id, expected_trail_revision=rev, action=DeclineCandidate("E1"))
            rev = res.trail_revision
            if res.status == "exhausted":
                exhausted_view = res
                break
        except EventTrailError:
            break

    assert exhausted_view is not None
    assert exhausted_view.status == "exhausted"

    records_after = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "hcmai.event_trail.interactions"
    ]
    exhaust_declines = [
        r for r in records_after
        if r["type"] == "trail_decline" and r["payload"]["outcome"] == "exhausted"
    ]
    assert len(exhaust_declines) >= 1
    exhaust_rec = exhaust_declines[0]
    assert exhaust_rec["payload"]["path_after"] is None
    assert exhaust_rec["payload"]["total_ms"] >= 0

    exhausted_events = [r for r in records_after if r["type"] == "trail_exhausted"]
    assert len(exhausted_events) >= 1
    assert exhausted_events[0]["payload"]["exhausted_by_event_id"] == "E1"
    assert len(exhausted_events[0]["payload"]["last_valid_path"]) > 0


def test_no_heavy_retrieval_dependency_in_action_loop(
    service: EventTrailService,
    stores: Stores,
    snapshot: EvidenceSnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)

    def fail_rescore(*args, **kwargs):
        raise AssertionError("EventTrail action reran full-corpus scoring")

    monkeypatch.setattr(service.decoder.temporal, "score_plan", fail_rescore)
    service.act(view.session_id, view.trail_revision, DeclineCandidate(event_id="E2"))


def test_use_frame_logs_submission_select(
    service: EventTrailService,
    stores: Stores,
    snapshot: EvidenceSnapshot,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="hcmai.event_trail.interactions")
    stores.snapshots.put(snapshot)
    view = service.open(snapshot.snapshot_id, "r_1", snapshot.kis_revision)

    service.act(
        view.session_id,
        expected_trail_revision=0,
        action=UseFrame(event_id="E2", frame_id="f2"),
    )

    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "hcmai.event_trail.interactions"
    ]
    use_rec = [r for r in records if r["type"] == "trail_use"][0]
    assert use_rec["event_id"] == "E2"
    assert use_rec["payload"]["outcome"] == "active"

    select_rec = [r for r in records if r["type"] == "submission_select"][0]
    assert select_rec["event_id"] == "E2"
    assert select_rec["payload"]["frame_id"] == "f2"
    assert select_rec["payload"]["frame_idx"] == 2
    assert select_rec["payload"]["timestamp_ms"] == 2000

