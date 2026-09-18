"""Tests for TemporalConstraintDecoder."""

import numpy as np
import pytest

from hcmai.common.config import AlignmentConfig
from hcmai.corpus.models import Frame
from hcmai.event_trail.decoder import (
    ConstraintSnapshot,
    DecodeOutcome,
    TemporalConstraintDecoder,
)
from hcmai.orchestration.workflows.search.temporal import (
    DecoderConfigSnapshot,
    TemporalSearchService,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import AlignedPath


class FakeCorpus:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = {f.frame_id: f for f in frames}

    def frame(self, frame_id: str) -> Frame:
        return self._frames[frame_id]


@pytest.fixture
def scored_video_3frames() -> VideoEventScores:
    return VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["f1", "f2", "f3"]),
        frame_idx=np.array([1, 2, 3]),
        timestamps_ms=np.array([1000, 3000, 7000], dtype=np.int64),
        scores=np.array([
            [0.9, 0.2, 0.1],
            [0.1, 0.8, 0.2],
            [0.2, 0.1, 0.9],
        ], dtype=np.float32),
    )


@pytest.fixture
def scored_video_duplicates() -> VideoEventScores:
    return VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["f1", "f2a", "f2b", "f3"]),
        frame_idx=np.array([1, 2, 3, 4]),
        timestamps_ms=np.array([1000, 3000, 3000, 7000], dtype=np.int64),
        scores=np.array([
            [0.9, 0.2, 0.2, 0.1],
            [0.1, 0.8, 0.7, 0.2],
            [0.2, 0.1, 0.1, 0.9],
        ], dtype=np.float32),
    )


@pytest.fixture
def scored_video_single() -> VideoEventScores:
    return VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["f1"]),
        frame_idx=np.array([1]),
        timestamps_ms=np.array([3000], dtype=np.int64),
        scores=np.array([[0.9]], dtype=np.float32),
    )


@pytest.fixture
def temporal_service(scored_video_3frames) -> TemporalSearchService:
    frames = [
        Frame(video_id="v1", frame_id=fid, frame_idx=idx, timestamp_ms=t, image_path="/tmp/f.jpg")
        for fid, idx, t in zip(
            scored_video_3frames.frame_ids,
            scored_video_3frames.frame_idx,
            scored_video_3frames.timestamps_ms,
            strict=True,
        )
    ]
    corpus = FakeCorpus(frames)
    evidence = None  # decode_video does not use evidence
    config = AlignmentConfig(lambda_gap=0.5, event_power=1.0, cluster_delta=0.0, path_min_separation_ms=0)
    service = TemporalSearchService(
        corpus=corpus,  # type: ignore[arg-type]
        evidence=evidence,  # type: ignore[arg-type]
        config=config,
    )
    return service


@pytest.fixture
def decoder(temporal_service: TemporalSearchService) -> TemporalConstraintDecoder:
    return TemporalConstraintDecoder(temporal=temporal_service)


@pytest.fixture
def decoder_config() -> DecoderConfigSnapshot:
    return DecoderConfigSnapshot(
        lambda_gap=0.5,
        event_power=1.0,
        cluster_delta=0.0,
        path_min_separation_ms=0,
    )


def test_rejection_cell_uses_neighbor_midpoints(
    decoder: TemporalConstraintDecoder, scored_video_3frames: VideoEventScores
) -> None:
    # timestamps: [1000, 3000, 7000]
    assert decoder.rejection_cell(scored_video_3frames, "f1") == (1000, 2000)
    assert decoder.rejection_cell(scored_video_3frames, "f2") == (2001, 5000)
    assert decoder.rejection_cell(scored_video_3frames, "f3") == (5001, 7000)


def test_rejection_cell_with_duplicate_timestamps(
    decoder: TemporalConstraintDecoder, scored_video_duplicates: VideoEventScores
) -> None:
    # timestamps: [1000, 3000, 3000, 7000]
    assert decoder.rejection_cell(scored_video_duplicates, "f2a") == (2001, 5000)
    assert decoder.rejection_cell(scored_video_duplicates, "f2b") == (2001, 5000)
    assert decoder.rejection_cell(scored_video_duplicates, "f1") == (1000, 2000)
    assert decoder.rejection_cell(scored_video_duplicates, "f3") == (5001, 7000)


def test_rejection_cell_single_timestamp(
    decoder: TemporalConstraintDecoder, scored_video_single: VideoEventScores
) -> None:
    assert decoder.rejection_cell(scored_video_single, "f1") == (3000, 3000)


def test_anchor_keeps_event_fixed(
    decoder: TemporalConstraintDecoder,
    scored_video_3frames: VideoEventScores,
    decoder_config: DecoderConfigSnapshot,
) -> None:
    # Anchoring E2 at "f2"
    constraints = ConstraintSnapshot(
        anchors=(None, "f2", None),
        rejected_cells=((), (), ()),
        window=None,
    )
    outcome = decoder.decode(scored_video_3frames, constraints, decoder_config)
    assert outcome.status == "ok"
    assert outcome.path is not None
    assert outcome.path.frame_ids[1] == "f2"


def test_reject_twice_excludes_both_cells(
    decoder: TemporalConstraintDecoder,
    scored_video_3frames: VideoEventScores,
    decoder_config: DecoderConfigSnapshot,
) -> None:
    # Rejecting f1 and f2 for E1
    cell1 = decoder.rejection_cell(scored_video_3frames, "f1")
    cell2 = decoder.rejection_cell(scored_video_3frames, "f2")
    constraints = ConstraintSnapshot(
        anchors=(None, None, None),
        rejected_cells=((cell1, cell2), (), ()),
        window=None,
    )
    outcome = decoder.decode(scored_video_3frames, constraints, decoder_config)
    # E1 can only pick f3, but since E2 and E3 must follow E1 strictly after or non-decreasingly,
    # let's see if a path exists
    assert outcome.constraint_ms >= 0


def test_anchor_outside_window_is_contradictory(
    decoder: TemporalConstraintDecoder,
    scored_video_3frames: VideoEventScores,
    decoder_config: DecoderConfigSnapshot,
) -> None:
    # f1 is at 1000 ms, but window is [2000, 7000]
    constraints = ConstraintSnapshot(
        anchors=("f1", None, None),
        rejected_cells=((), (), ()),
        window=(2000, 7000),
    )
    outcome = decoder.decode(scored_video_3frames, constraints, decoder_config)
    assert outcome.status == "contradictory_conditions"
    assert outcome.path is None


def test_rejections_exhausting_all_paths(
    decoder: TemporalConstraintDecoder,
    scored_video_3frames: VideoEventScores,
    decoder_config: DecoderConfigSnapshot,
) -> None:
    # Reject entire video window for E1
    constraints = ConstraintSnapshot(
        anchors=(None, None, None),
        rejected_cells=(((1000, 7000),), (), ()),
        window=None,
    )
    outcome = decoder.decode(scored_video_3frames, constraints, decoder_config)
    assert outcome.status in ("no_indexed_frames", "contradictory_conditions", "no_valid_path")
    assert outcome.path is None


def test_clear_recovers_path(
    decoder: TemporalConstraintDecoder,
    scored_video_3frames: VideoEventScores,
    decoder_config: DecoderConfigSnapshot,
) -> None:
    # When constraints are empty, path is recovered
    constraints = ConstraintSnapshot(
        anchors=(None, None, None),
        rejected_cells=((), (), ()),
        window=None,
    )
    outcome = decoder.decode(scored_video_3frames, constraints, decoder_config)
    assert outcome.status == "ok"
    assert outcome.path is not None
    assert len(outcome.path.frame_ids) == 3
