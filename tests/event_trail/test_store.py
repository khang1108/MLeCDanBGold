"""Tests for EvidenceSnapshotStore TTL, LRU, and immutability."""

from datetime import datetime, timezone
import numpy as np
import pytest

from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import (
    EvidenceSnapshot,
    SnapshotResult,
    freeze_video_scores,
)
from hcmai.event_trail.storage import EvidenceSnapshotStore
from hcmai.orchestration.workflows.search.temporal import DecoderConfigSnapshot
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self._current = start

    def __call__(self) -> float:
        return self._current

    def advance(self, seconds: float) -> None:
        self._current += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sample_video() -> VideoEventScores:
    return VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["v1_f1", "v1_f2"]),
        frame_idx=np.array([10, 20]),
        timestamps_ms=np.array([1000, 2000], dtype=np.int64),
        scores=np.array([[0.9, 0.8]], dtype=np.float32),
    )


@pytest.fixture
def snapshot(sample_video: VideoEventScores) -> EvidenceSnapshot:
    frozen = freeze_video_scores(sample_video)
    now = datetime.now(timezone.utc)
    return EvidenceSnapshot(
        snapshot_id="s_123",
        kis_revision=1,
        scoring_revision="score_gen_1",
        event_ids=("E1",),
        decoder_config=DecoderConfigSnapshot(
            lambda_gap=0.5,
            event_power=1.0,
            cluster_delta=2.0,
            path_min_separation_ms=1000,
        ),
        results={
            "r_1": SnapshotResult(
                result_id="r_1",
                video_id="v1",
                initial_path=("v1_f1",),
                path_score=0.9,
            )
        },
        video_evidence={"v1": frozen},
        created_at=now,
        expires_at=now,
    )


@pytest.fixture
def snapshot_store(clock: FakeClock) -> EvidenceSnapshotStore:
    return EvidenceSnapshotStore(ttl_seconds=900, max_entries=64, clock=clock)


def test_snapshot_expiration_is_fixed_not_sliding(
    snapshot_store: EvidenceSnapshotStore, clock: FakeClock, snapshot: EvidenceSnapshot
) -> None:
    snapshot_store.put(snapshot)
    clock.advance(500)
    assert snapshot_store.get(snapshot.snapshot_id) is snapshot
    clock.advance(401)
    with pytest.raises(EventTrailError) as exc:
        snapshot_store.get(snapshot.snapshot_id)
    assert exc.value.code == "SNAPSHOT_EXPIRED"


def test_stored_arrays_are_read_only(snapshot: EvidenceSnapshot) -> None:
    evidence = snapshot.video_evidence["v1"]
    assert not evidence.scores.flags.writeable
    assert not evidence.frame_ids.flags.writeable
    assert not evidence.frame_idx.flags.writeable
    assert not evidence.timestamps_ms.flags.writeable


def test_capacity_eviction_is_not_tombstoned(
    clock: FakeClock, sample_video: VideoEventScores
) -> None:
    store = EvidenceSnapshotStore(ttl_seconds=900, max_entries=2, clock=clock)
    frozen = freeze_video_scores(sample_video)
    now = datetime.now(timezone.utc)

    def make_snap(sid: str) -> EvidenceSnapshot:
        return EvidenceSnapshot(
            snapshot_id=sid,
            kis_revision=1,
            scoring_revision="score_gen_1",
            event_ids=("E1",),
            decoder_config=DecoderConfigSnapshot(0.5, 1.0, 2.0, 1000),
            results={},
            video_evidence={"v1": frozen},
            created_at=now,
            expires_at=now,
        )

    s1 = make_snap("s_1")
    s2 = make_snap("s_2")
    s3 = make_snap("s_3")

    store.put(s1)
    store.put(s2)
    store.put(s3)  # evicts s1 by capacity

    with pytest.raises(EventTrailError) as exc:
        store.get("s_1")
    assert exc.value.code == "SNAPSHOT_NOT_FOUND"

    assert store.get("s_2") is s2
    assert store.get("s_3") is s3
