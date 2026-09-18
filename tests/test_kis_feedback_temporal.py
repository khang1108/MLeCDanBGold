"""Tests for local temporal repair, bounding blocks, and anchor-conditioned DP decoding."""

from unittest.mock import Mock

import numpy as np
import pytest

from hcmai.common.config import AlignmentConfig
from hcmai.corpus.models import Frame
from hcmai.event_trail.decoder import (
    ConstraintSnapshot,
    DecodeOutcome,
    TemporalConstraintDecoder,
    repair_block,
)
from hcmai.orchestration.workflows.search.temporal import (
    DecoderConfigSnapshot,
    TemporalSearchService,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import AlignedPath


def _make_aligned_path(video_id: str, frame_ids: tuple[str, ...], timestamps_ms: tuple[int, ...], score: float = 1.0) -> AlignedPath:
    return AlignedPath(
        video_id=video_id,
        score=score,
        frame_ids=frame_ids,
        frame_idxs=tuple(range(1, len(frame_ids) + 1)),
        timestamps_ms=timestamps_ms,
    )


class FakeCorpus:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = {f.frame_id: f for f in frames}

    def frame(self, frame_id: str) -> Frame:
        return self._frames[frame_id]


@pytest.fixture
def synthetic_5frames_video() -> VideoEventScores:
    # Timestamps at 10, 20, 30, 40, 50 seconds in ms
    return VideoEventScores(
        video_id="v_synth",
        frame_ids=np.array(["f10", "f20", "f30", "f40", "f50"]),
        frame_idx=np.array([1, 2, 3, 4, 5]),
        timestamps_ms=np.array([10000, 20000, 30000, 40000, 50000], dtype=np.int64),
        scores=np.array([
            [0.9, 0.2, 0.1, 0.1, 0.1],  # E1
            [0.1, 0.7, 0.9, 0.6, 0.1],  # E2
            [0.1, 0.1, 0.1, 0.2, 0.95], # E3
        ], dtype=np.float32),
    )


@pytest.fixture
def decoder(synthetic_5frames_video: VideoEventScores) -> TemporalConstraintDecoder:
    frames = [
        Frame(
            video_id="v_synth",
            frame_id=fid,
            frame_idx=idx,
            timestamp_ms=t,
            image_path="/tmp/f.jpg",
        )
        for fid, idx, t in zip(
            synthetic_5frames_video.frame_ids,
            synthetic_5frames_video.frame_idx,
            synthetic_5frames_video.timestamps_ms,
            strict=True,
        )
    ]
    corpus = FakeCorpus(frames)
    config = AlignmentConfig(lambda_gap=0.5, event_power=1.0, cluster_delta=0.0, path_min_separation_ms=0)
    service = TemporalSearchService(corpus=corpus, evidence=None, config=config)  # type: ignore[arg-type]
    return TemporalConstraintDecoder(temporal=service)


@pytest.fixture
def decoder_config() -> DecoderConfigSnapshot:
    return DecoderConfigSnapshot(
        lambda_gap=0.5,
        event_power=1.0,
        cluster_delta=0.0,
        path_min_separation_ms=0,
    )


def test_repair_block_calculation():
    """repair_block returns inclusive unconfirmed block bounded by nearest anchors."""
    # 3 events, anchor at 0 and 2
    anchors = ("f10", None, "f50")
    assert repair_block(anchors, 1) == (1, 1)

    # 4 events, anchor at 0 and 3
    anchors_4 = ("f1", None, None, "f4")
    assert repair_block(anchors_4, 1) == (1, 2)
    assert repair_block(anchors_4, 2) == (1, 2)

    # First event with right anchor
    anchors_first = (None, "f2", None)
    assert repair_block(anchors_first, 0) == (0, 0)

    # Last event with left anchor
    anchors_last = (None, "f2", None)
    assert repair_block(anchors_last, 2) == (2, 2)

    # No anchors at all
    anchors_none = (None, None, None)
    assert repair_block(anchors_none, 1) == (0, 2)


def test_repair_strictly_inside_anchors(
    decoder: TemporalConstraintDecoder,
    synthetic_5frames_video: VideoEventScores,
    decoder_config: DecoderConfigSnapshot,
):
    """E2 repair must stay strictly between anchor E1@10s and anchor E3@50s; outside block unchanged."""
    constraints = ConstraintSnapshot(
        anchors=("f10", None, "f50"),
        rejected_cells=(),
        window=None,
    )
    current_path = _make_aligned_path(
        video_id="v_synth",
        frame_ids=("f10", "f20", "f50"),
        timestamps_ms=(10000, 20000, 50000),
        score=2.5,
    )

    outcome = decoder.repair(
        video=synthetic_5frames_video,
        constraints=constraints,
        decoder_config=decoder_config,
        current_path=current_path,
        target_event_index=1,
    )

    assert outcome.status == "ok"
    assert outcome.path is not None
    # E1 pinned to f10, E3 pinned to f50
    assert outcome.path.frame_ids[0] == "f10"
    assert outcome.path.frame_ids[2] == "f50"
    # E2 must be strictly inside (10000, 50000), e.g. f30 has highest score (0.9)
    assert outcome.path.frame_ids[1] == "f30"
    assert outcome.path.timestamps_ms[1] == 30000
    assert 10000 < outcome.path.timestamps_ms[1] < 50000


def test_repair_target_is_anchored_raises():
    """Attempting to repair an event that is still anchored must be rejected."""
    decoder = TemporalConstraintDecoder(temporal=Mock())
    constraints = ConstraintSnapshot(
        anchors=("f10", "f20", "f50"),
        rejected_cells=(),
        window=None,
    )
    with pytest.raises(ValueError, match="anchored"):
        decoder.repair(
            video=Mock(),
            constraints=constraints,
            decoder_config=Mock(),
            current_path=Mock(),
            target_event_index=1,
        )


def test_repair_contradictory_anchors(
    decoder: TemporalConstraintDecoder,
    synthetic_5frames_video: VideoEventScores,
    decoder_config: DecoderConfigSnapshot,
):
    """When left anchor timestamp is >= right anchor timestamp, outcome is contradictory_conditions."""
    # E1 at 50s, E3 at 10s (contradictory)
    constraints = ConstraintSnapshot(
        anchors=("f50", None, "f10"),
        rejected_cells=(),
        window=None,
    )
    current_path = _make_aligned_path(
        video_id="v_synth",
        frame_ids=("f50", "f20", "f10"),
        timestamps_ms=(50000, 20000, 10000),
        score=1.0,
    )
    outcome = decoder.repair(
        video=synthetic_5frames_video,
        constraints=constraints,
        decoder_config=decoder_config,
        current_path=current_path,
        target_event_index=1,
    )
    assert outcome.status == "contradictory_conditions"
    assert outcome.path is None


def test_repair_empty_domain_returns_no_valid_path(
    decoder: TemporalConstraintDecoder,
    synthetic_5frames_video: VideoEventScores,
    decoder_config: DecoderConfigSnapshot,
):
    """When no indexed frames exist strictly between anchors, returns no_valid_path."""
    # Anchors at f10 (10s) and f20 (20s) -> no frame exists strictly between 10s and 20s
    constraints = ConstraintSnapshot(
        anchors=("f10", None, "f20"),
        rejected_cells=(),
        window=None,
    )
    current_path = _make_aligned_path(
        video_id="v_synth",
        frame_ids=("f10", "f20", "f20"),
        timestamps_ms=(10000, 20000, 20000),
        score=1.0,
    )
    outcome = decoder.repair(
        video=synthetic_5frames_video,
        constraints=constraints,
        decoder_config=decoder_config,
        current_path=current_path,
        target_event_index=1,
    )
    assert outcome.status in ("no_valid_path", "no_indexed_frames")
    assert outcome.path is None


def test_repair_duplicate_timestamp_frame_identity(
    decoder_config: DecoderConfigSnapshot,
):
    """Pins outside-block frames by exact frame_id even when multiple frames share the same timestamp."""
    video = VideoEventScores(
        video_id="v_dup",
        frame_ids=np.array(["f1", "f2a", "f2b", "f3"]),
        frame_idx=np.array([1, 2, 3, 4]),
        timestamps_ms=np.array([1000, 3000, 3000, 7000], dtype=np.int64),
        scores=np.array([
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.8, 0.1],
            [0.1, 0.1, 0.1, 0.9],
        ], dtype=np.float32),
    )
    frames = [
        Frame(video_id="v_dup", frame_id=fid, frame_idx=idx, timestamp_ms=t, image_path="/tmp/f.jpg")
        for fid, idx, t in zip(video.frame_ids, video.frame_idx, video.timestamps_ms, strict=True)
    ]
    temporal_service = TemporalSearchService(
        corpus=FakeCorpus(frames),  # type: ignore[arg-type]
        evidence=None,  # type: ignore[arg-type]
        config=AlignmentConfig(lambda_gap=0.5, event_power=1.0, cluster_delta=0.0, path_min_separation_ms=0),
    )
    decoder = TemporalConstraintDecoder(temporal=temporal_service)

    # E1 anchored to f1, E2 anchored to f2b, repairing E3
    constraints = ConstraintSnapshot(
        anchors=("f1", "f2b", None),
        rejected_cells=((), (), ()),
        window=None,
    )
    current_path = _make_aligned_path(
        video_id="v_dup",
        frame_ids=("f1", "f2b", "f3"),
        timestamps_ms=(1000, 3000, 7000),
        score=2.6,
    )

    outcome = decoder.repair(
        video=video,
        constraints=constraints,
        decoder_config=decoder_config,
        current_path=current_path,
        target_event_index=2,  # Repairing E3
    )

    assert outcome.status == "ok"
    assert outcome.path is not None
    # E2 is outside the repair block for E3 (since E2 was in current_path), must remain f2b (not f2a)
    assert outcome.path.frame_ids[1] == "f2b"
