"""Unit tests verifying exact frame_id pinning in TemporalConstraintDecoder.

Ensures that when multiple canonical frames share the exact same timestamp,
pinning an anchor to frame_id pins that exact frame rather than allowing
DP to select a different frame with a higher score at the same timestamp.
"""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.common.config import AlignmentConfig
from hcmai.corpus.models import Frame
from hcmai.event_trail.config import EventTrailSettings
from hcmai.event_trail.decoding import (
    ConstraintSnapshot,
    TemporalConstraintDecoder,
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
def scored_video_duplicate_timestamps() -> VideoEventScores:
    """Video where f2a and f2b share timestamp 3000ms, but f2a has a higher score than f2b."""
    return VideoEventScores(
        video_id="v_dups",
        frame_ids=np.array(["f1", "f2a", "f2b", "f3"]),
        frame_idx=np.array([1, 2, 3, 4]),
        timestamps_ms=np.array([1000, 3000, 3000, 7000], dtype=np.int64),
        scores=np.array([
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.95, 0.60, 0.1],  # f2a has 0.95, f2b has 0.60
            [0.1, 0.1, 0.1, 0.9],
        ], dtype=np.float32),
    )


@pytest.fixture
def decoder(scored_video_duplicate_timestamps: VideoEventScores) -> TemporalConstraintDecoder:
    frames = [
        Frame(video_id="v_dups", frame_id=fid, frame_idx=idx, timestamp_ms=t, image_path="/tmp/f.jpg")
        for fid, idx, t in zip(
            scored_video_duplicate_timestamps.frame_ids,
            scored_video_duplicate_timestamps.frame_idx,
            scored_video_duplicate_timestamps.timestamps_ms,
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


def test_decode_pins_exact_frame_id_when_timestamps_identical(
    decoder: TemporalConstraintDecoder,
    scored_video_duplicate_timestamps: VideoEventScores,
    decoder_config: DecoderConfigSnapshot,
) -> None:
    # Pin E1 (index 1) to f2b (the one with lower score 0.60 vs f2a's 0.95)
    constraints = ConstraintSnapshot(
        anchors=(None, "f2b", None),
        rejected_cells=((), (), ()),
        window=None,
    )

    outcome = decoder.decode(scored_video_duplicate_timestamps, constraints, decoder_config)
    assert outcome.status == "ok"
    assert outcome.path is not None
    # Must pick f2b, not f2a!
    assert outcome.path.frame_ids[1] == "f2b"


def test_alternatives_pins_exact_frame_id_when_timestamps_identical(
    decoder: TemporalConstraintDecoder,
    scored_video_duplicate_timestamps: VideoEventScores,
    decoder_config: DecoderConfigSnapshot,
) -> None:
    settings = EventTrailSettings()

    # Pin E1 (index 1) to f2b, and focus on E0 (index 0) to get alternatives
    constraints = ConstraintSnapshot(
        anchors=(None, "f2b", None),
        rejected_cells=((), (), ()),
        window=None,
    )

    modes = decoder.alternatives(
        scored_video_duplicate_timestamps,
        constraints,
        decoder_config,
        event_idx=0,
        event_id="E1",
        settings=settings,
    )
    # Every returned mode's conditioned path must respect the anchor on E1 as f2b
    assert len(modes) > 0
    for mode in modes:
        if mode.path is not None:
            assert mode.path.frame_ids[1] == "f2b"
