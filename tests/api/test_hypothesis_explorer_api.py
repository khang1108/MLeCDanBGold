"""API tests for Result Hypothesis Explorer / EventTrail alternatives and actions."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hcmai.app import create_app
from hcmai.common.config import AlignmentConfig
from hcmai.corpus.models import Frame
from hcmai.event_trail.config import EventTrailSettings
from hcmai.event_trail.decoding import TemporalConstraintDecoder
from hcmai.event_trail.models import (
    EvidenceSnapshot,
    SnapshotResult,
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
def api_test_setup():
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
    event_trail = EventTrailService(
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

    mock_search = Mock()
    mock_search.event_trail = event_trail
    app = create_app(search_service=mock_search)
    client = TestClient(app)
    return client, snapshot


@pytest.fixture
def client(api_test_setup) -> TestClient:
    return api_test_setup[0]


@pytest.fixture
def opened_trail(api_test_setup) -> dict:
    client, snapshot = api_test_setup
    res = client.post(
        "/api/v1/event-trail/open",
        json={
            "snapshot_id": snapshot.snapshot_id,
            "result_id": "r_1",
            "expected_kis_revision": 1,
        },
    )
    assert res.status_code == 200, res.text
    return res.json()


def test_old_alternative_is_rejected_after_commit(
    client: TestClient, opened_trail: dict
) -> None:
    modes_res = client.get(
        f"/api/v1/event-trail/{opened_trail['session_id']}/alternatives",
        params={
            "event_id": "E2",
            "expected_trail_revision": opened_trail["trail_revision"],
        },
    )
    assert modes_res.status_code == 200, modes_res.text
    modes = modes_res.json()["alternatives"]
    assert len(modes) > 0
    first = modes[0]

    # Commit a keep action on E1 to advance trail_revision
    committed = client.post(
        f"/api/v1/event-trail/{opened_trail['session_id']}/actions",
        json={
            "expected_trail_revision": opened_trail["trail_revision"],
            "action": {"type": "keep", "event_id": "E1"},
        },
    ).json()
    assert committed["trail_revision"] == opened_trail["trail_revision"] + 1

    # Attempting to use the old alternative from before the commit must fail with 409 STALE_ALTERNATIVE
    stale = client.post(
        f"/api/v1/event-trail/{opened_trail['session_id']}/actions",
        json={
            "expected_trail_revision": committed["trail_revision"],
            "action": {
                "type": "use_alternative",
                "event_id": "E2",
                "alternative_id": first["alternative_id"],
            },
        },
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "STALE_ALTERNATIVE"
