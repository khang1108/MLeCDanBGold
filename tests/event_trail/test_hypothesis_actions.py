"""Tests for Keep, UseAlternative, and RejectMode actions in EventTrail."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from hcmai.common.config import AlignmentConfig
from hcmai.corpus.models import Frame
from hcmai.event_trail.config import EventTrailSettings
from hcmai.event_trail.decoding import TemporalConstraintDecoder
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import (
    EvidenceSnapshot,
    KeepOccurrence,
    RejectMode,
    SnapshotResult,
    TemporalMode,
    TrailView,
    UseAlternative,
    freeze_video_scores,
)
from hcmai.event_trail.service import EventTrailService
from hcmai.event_trail.storage import (
    EventTrailSessionStore,
    EvidenceSnapshotStore,
)
from hcmai.orchestration.workflows.search.temporal import (
    DecoderConfigSnapshot,
    TemporalSearchService,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class FakeCorpus:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = {f.frame_id: f for f in frames}

    def frame(self, frame_id: str) -> Frame:
        return self._frames[frame_id]


@pytest.fixture
def service_and_snapshot() -> tuple[EventTrailService, EvidenceSnapshot]:
    frame_ids = np.array(["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"])
    frame_idx = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    timestamps_ms = np.array(
        [1_000, 2_000, 18_000, 19_000, 41_000, 42_000, 60_000, 61_000],
        dtype=np.int64,
    )
    scores = np.array(
        [
            [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # E1
            [0.1, 0.1, 0.9, 0.85, 0.88, 0.82, 0.1, 0.1],  # E2
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.85],  # E3
        ],
        dtype=np.float32,
    )

    video = VideoEventScores(
        video_id="v1",
        frame_ids=frame_ids,
        frame_idx=frame_idx,
        timestamps_ms=timestamps_ms,
        scores=scores,
    )

    frames = [
        Frame(
            video_id="v1",
            frame_id=fid,
            frame_idx=idx,
            timestamp_ms=t,
            image_path="/tmp/f.jpg",
        )
        for fid, idx, t in zip(
            frame_ids, frame_idx, timestamps_ms, strict=True
        )
    ]
    corpus = FakeCorpus(frames)
    config = AlignmentConfig(
        lambda_gap=0.0,
        event_power=1.0,
        cluster_delta=0.0,
        path_min_separation_ms=0,
    )
    temporal_service = TemporalSearchService(
        corpus=corpus,  # type: ignore[arg-type]
        evidence=None,  # type: ignore[arg-type]
        config=config,
    )
    decoder = TemporalConstraintDecoder(temporal=temporal_service)
    snapshot_store = EvidenceSnapshotStore(ttl_seconds=900, max_entries=8)
    session_store = EventTrailSessionStore(ttl_seconds=1800, max_entries=8)
    settings = EventTrailSettings(
        alternative_count=4,
        mode_min_separation_ms=5_000,
        mode_max_radius_ms=8_000,
    )
    service = EventTrailService(
        snapshot_store=snapshot_store,
        session_store=session_store,
        decoder=decoder,
        settings=settings,
    )

    frozen = freeze_video_scores(video)
    now = datetime.now(timezone.utc)
    snapshot = EvidenceSnapshot(
        snapshot_id="snap_1",
        kis_revision=1,
        scoring_revision="score_gen_1",
        event_ids=("E1", "E2", "E3"),
        decoder_config=DecoderConfigSnapshot(
            lambda_gap=0.0,
            event_power=1.0,
            cluster_delta=0.0,
            path_min_separation_ms=0,
        ),
        results={
            "r_1": SnapshotResult(
                result_id="r_1",
                video_id="v1",
                initial_path=("f1", "f3", "f7"),
                path_score=2.7,
            )
        },
        video_evidence={"v1": frozen},
        created_at=now,
        expires_at=now,
    )
    snapshot_store.put(snapshot)
    return service, snapshot


@pytest.fixture
def service(service_and_snapshot) -> EventTrailService:
    return service_and_snapshot[0]


@pytest.fixture
def opened_session(service_and_snapshot) -> TrailView:
    svc, snap = service_and_snapshot
    return svc.open(snap.snapshot_id, "r_1", expected_kis_revision=1)


@pytest.fixture
def session_with_modes(service, opened_session) -> tuple[TrailView, TemporalMode]:
    modes = service.alternatives(
        opened_session.session_id, opened_session.trail_revision, "E2"
    )
    # Pick the first alternative mode that is not the current active mode
    non_current = [m for m in modes if not m.is_current]
    chosen = non_current[0] if non_current else modes[0]
    return opened_session, chosen


def test_fetching_alternatives_is_non_mutating(service, opened_session):
    before = service.get(opened_session.session_id)
    modes = service.alternatives(
        opened_session.session_id, before.trail_revision, "E2"
    )
    after = service.get(opened_session.session_id)
    assert modes
    assert after.trail_revision == before.trail_revision
    assert after.constraints == before.constraints


def test_reject_mode_excludes_entire_mode_interval(service, session_with_modes):
    state, mode = session_with_modes
    updated = service.act(
        state.session_id,
        state.trail_revision,
        RejectMode(event_id="E2", mode_id=mode.mode_id),
    )
    assert updated.trail_revision == state.trail_revision + 1
    stored = service.get(state.session_id)
    # Event E2 is index 1
    # Check that mode interval is in rejected cells for E2
    assert any(
        cell == mode.interval
        for cell in stored.constraints.rejected_cells[1]
    )


def test_use_alternative_commits_anchor_not_preview_path(
    service, session_with_modes
):
    state, mode = session_with_modes
    service.act(
        state.session_id,
        state.trail_revision,
        UseAlternative(event_id="E2", alternative_id=mode.mode_id),
    )
    stored = service.get(state.session_id)
    assert stored.constraints.anchors[1] == mode.representative_frame_id


def test_keep_occurrence_commits_anchor(service, opened_session):
    before = service.get(opened_session.session_id)
    assert before.path is not None
    current_fid = before.path[1].frame_id
    updated = service.act(
        opened_session.session_id,
        before.trail_revision,
        KeepOccurrence(event_id="E2"),
    )
    assert updated.trail_revision == before.trail_revision + 1
    stored = service.get(opened_session.session_id)
    assert stored.constraints.anchors[1] == current_fid


def test_use_alternative_fails_if_alternative_is_stale(
    service, session_with_modes
):
    state, mode = session_with_modes
    # First commit an action that advances revision and clears alternatives
    service.act(
        state.session_id,
        state.trail_revision,
        KeepOccurrence(event_id="E1"),
    )
    latest = service.get(state.session_id)
    with pytest.raises(EventTrailError) as exc_info:
        service.act(
            state.session_id,
            latest.trail_revision,
            UseAlternative(event_id="E2", alternative_id=mode.mode_id),
        )
    assert exc_info.value.code == "STALE_ALTERNATIVE"


def test_result_action_logs_mode_and_revision(caplog, service, session_with_modes):
    import logging
    caplog.set_level(logging.INFO)
    state, mode = session_with_modes
    service.act(
        state.session_id,
        state.trail_revision,
        UseAlternative(event_id="E2", alternative_id=mode.mode_id),
    )
    records = [r for r in caplog.records if "trail_use_alternative" in r.message]
    assert len(records) > 0
    msg = records[0].message
    assert '"event_id":"E2"' in msg
    assert '"trail_revision":1' in msg
    assert f'"{mode.mode_id}"' in msg


def test_alternatives_fetching_is_logged(caplog, service, opened_session):
    import logging
    caplog.set_level(logging.INFO)
    modes = service.alternatives(
        opened_session.session_id, opened_session.trail_revision, "E2"
    )
    records = [r for r in caplog.records if "trail_alternatives" in r.message]
    assert len(records) > 0
    msg = records[0].message
    assert '"event_id":"E2"' in msg
    assert f'"mode_count":{len(modes)}' in msg
