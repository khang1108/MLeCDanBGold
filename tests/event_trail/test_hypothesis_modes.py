"""Tests for temporal hypothesis modes and alternatives decoding."""

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
from hcmai.event_trail.decoding.decoder import derive_mode_intervals
from hcmai.event_trail.models import TemporalMode
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
def mode_builder():
    def _builder(
        peaks: list[tuple[int, float]],
        max_radius_ms: int = 8_000,
        domain: tuple[int, int] | None = None,
    ) -> list[TemporalMode]:
        timestamps = [p[0] for p in peaks]
        intervals = derive_mode_intervals(
            timestamps, max_radius_ms=max_radius_ms, domain=domain
        )
        modes = []
        for i, ((ts, score), interval) in enumerate(zip(peaks, intervals, strict=True)):
            dummy_path = AlignedPath(
                video_id="v1",
                score=score,
                frame_ids=(f"f_{i}",),
                frame_idxs=(i,),
                timestamps_ms=(ts,),
            )
            modes.append(
                TemporalMode(
                    mode_id=f"mode_{i}",
                    event_id="E1",
                    representative_frame_id=f"f_{i}",
                    representative_frame_idx=i,
                    representative_timestamp_ms=ts,
                    interval=interval,
                    score=score,
                    path=dummy_path,
                )
            )
        return modes

    return _builder


class DecoderHarness:
    def __init__(
        self,
        decoder: TemporalConstraintDecoder,
        video: VideoEventScores,
        constraints: ConstraintSnapshot,
        decoder_config: DecoderConfigSnapshot,
        settings: EventTrailSettings,
    ) -> None:
        self.decoder = decoder
        self.video = video
        self.constraints = constraints
        self.decoder_config = decoder_config
        self.settings = settings

    def alternatives(
        self,
        event_idx: int,
        event_id: str | None = None,
        current_path: AlignedPath | None = None,
    ) -> tuple[TemporalMode, ...]:
        eid = event_id or f"E{event_idx + 1}"
        return self.decoder.alternatives(
            self.video,
            self.constraints,
            self.decoder_config,
            event_idx=event_idx,
            event_id=eid,
            settings=self.settings,
            current_path=current_path,
        )


@pytest.fixture
def decoder_fixture() -> DecoderHarness:
    # 8 frames spanning 60 seconds with multiple occurrences
    frame_ids = np.array(["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"])
    frame_idx = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    timestamps_ms = np.array(
        [1_000, 2_000, 18_000, 19_000, 41_000, 42_000, 60_000, 61_000],
        dtype=np.int64,
    )
    # 3 events: E1, E2, E3
    # E2 has high scores at f3 (18s), f4 (19s), f5 (41s), f6 (42s)
    scores = np.array(
        [
            [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # E1: peak at f1
            [0.1, 0.1, 0.9, 0.85, 0.88, 0.82, 0.1, 0.1],  # E2: clusters at ~18s and ~41s
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.85],  # E3: peak at f7 (~60s)
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
    constraints = ConstraintSnapshot(
        anchors=(None, None, None),
        rejected_cells=((), (), ()),
        window=None,
    )
    decoder_config = DecoderConfigSnapshot(
        lambda_gap=0.0,
        event_power=1.0,
        cluster_delta=0.0,
        path_min_separation_ms=0,
    )
    settings = EventTrailSettings(
        alternative_count=4,
        mode_min_separation_ms=5_000,
        mode_max_radius_ms=8_000,
    )
    return DecoderHarness(
        decoder=decoder,
        video=video,
        constraints=constraints,
        decoder_config=decoder_config,
        settings=settings,
    )


def test_mode_regions_are_bounded_by_nearest_competitor_and_max_radius(mode_builder):
    modes = mode_builder(
        peaks=[(18_000, 1.0), (41_000, 0.9), (70_000, 0.8)],
        max_radius_ms=8_000,
    )
    assert modes[0].interval == (10_000, 26_000)
    assert modes[1].interval == (33_000, 49_000)
    assert modes[2].interval == (62_000, 78_000)


def test_current_occurrence_is_not_exposed_as_multiple_nearby_modes(decoder_fixture):
    modes = decoder_fixture.alternatives(event_idx=1)
    timestamps = [mode.representative_timestamp_ms for mode in modes]
    assert all(
        abs(a - b) >= decoder_fixture.settings.mode_min_separation_ms
        for i, a in enumerate(timestamps)
        for b in timestamps[i + 1 :]
    )


def test_single_peak_uses_max_radius(mode_builder):
    modes = mode_builder(peaks=[(20_000, 1.0)], max_radius_ms=8_000)
    assert len(modes) == 1
    assert modes[0].interval == (12_000, 28_000)


def test_close_peaks_scale_radius_to_half_distance(mode_builder):
    modes = mode_builder(
        peaks=[(20_000, 1.0), (26_000, 0.9)],
        max_radius_ms=8_000,
    )
    # Distance is 6_000. Half-distance is 3_000 < 8_000.
    assert modes[0].interval == (17_000, 23_000)
    assert modes[1].interval == (23_000, 29_000)


def test_active_occurrence_is_marked_as_current(decoder_fixture):
    # First decode to get the active path
    outcome = decoder_fixture.decoder.decode(
        decoder_fixture.video,
        decoder_fixture.constraints,
        decoder_fixture.decoder_config,
    )
    assert outcome.path is not None
    modes = decoder_fixture.alternatives(event_idx=1, current_path=outcome.path)
    current_modes = [m for m in modes if m.is_current]
    assert len(current_modes) == 1
    assert (
        current_modes[0].interval[0]
        <= outcome.path.timestamps_ms[1]
        <= current_modes[0].interval[1]
    )
