"""Plan 04 contracts, facade routing, and TRAKE compatibility parity."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    FrameRecord,
    OrderedPathCandidate,
    QueryUnit,
    RetrievalTrace,
    TaskType,
    TemporalAlignmentMode,
    TemporalConstraint,
    TemporalQueryPlan,
    TemporalRelation,
)
from hcmai.data.pipeline import DataService
from hcmai.pipelines.trake import TRAKESettings, rank_paths
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal import TemporalEvidenceCore
from hcmai.temporal.aligners.monotonic import MonotonicOrderedPathAligner
from hcmai.temporal.ports import ProgressiveAcquisition


def _frame(
    frame_id: str,
    *,
    video_id: str = "v1",
    frame_idx: int = 0,
    timestamp_ms: int = 0,
) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        video_id=video_id,
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        image_path=f"{frame_id}.jpg",
        width=10,
        height=10,
    )


def _units() -> tuple[QueryUnit, QueryUnit]:
    return (
        QueryUnit(unit_id="e0", text="first", order=0),
        QueryUnit(unit_id="e1", text="second", order=1),
    )


def _ordered_plan() -> TemporalQueryPlan:
    return TemporalEvidenceCore.ordered_plan(["first", "second"])


@pytest.mark.parametrize(
    ("task", "mode"),
    [
        (TaskType.KIS, TemporalAlignmentMode.ORDERED_PATH),
        (TaskType.TRAKE, TemporalAlignmentMode.PROGRESSIVE_SCENE),
    ],
)
def test_query_plan_rejects_task_mode_mismatches(task, mode) -> None:
    with pytest.raises(ValidationError, match="requires"):
        TemporalQueryPlan(task_type=task, units=_units(), alignment_mode=mode)


def test_query_plan_validates_unit_identity_order_and_constraints() -> None:
    with pytest.raises(ValidationError, match="unique"):
        TemporalQueryPlan(
            task_type=TaskType.KIS,
            units=(
                QueryUnit(unit_id="h0", text="one", order=0),
                QueryUnit(unit_id="h0", text="two", order=1),
            ),
            alignment_mode=TemporalAlignmentMode.PROGRESSIVE_SCENE,
        )
    with pytest.raises(ValidationError, match="consecutive"):
        TemporalQueryPlan(
            task_type=TaskType.KIS,
            units=(QueryUnit(unit_id="h0", text="one", order=3),),
            alignment_mode=TemporalAlignmentMode.PROGRESSIVE_SCENE,
        )
    with pytest.raises(ValidationError, match="only BEFORE"):
        TemporalQueryPlan(
            task_type=TaskType.TRAKE,
            units=_units(),
            constraints=(TemporalConstraint(
                relation=TemporalRelation.OVERLAP,
                subject_unit_id="e0",
                object_unit_id="e1",
                reason="invalid_ordered_relation",
            ),),
            alignment_mode=TemporalAlignmentMode.ORDERED_PATH,
        )


def test_ordered_path_rejects_mixed_video_and_nonchronological_frames() -> None:
    with pytest.raises(ValidationError, match="path.video_id"):
        OrderedPathCandidate(
            path_id="p1",
            video_id="v1",
            frames=(_frame("f1"), _frame("f2", video_id="v2")),
            query_unit_ids=("e0", "e1"),
            score=1.0,
        )
    with pytest.raises(ValidationError, match="chronological"):
        OrderedPathCandidate(
            path_id="p2",
            video_id="v1",
            frames=(
                _frame("f2", timestamp_ms=2_000),
                _frame("f1", timestamp_ms=1_000),
            ),
            query_unit_ids=("e0", "e1"),
            score=1.0,
        )


class Data:
    def __init__(self, frames: list[FrameRecord]) -> None:
        self.frames = {frame.frame_id: frame for frame in frames}

    def get_frame(self, frame_id: str) -> FrameRecord:
        return self.frames[frame_id]


def _scores() -> VideoEventScores:
    return VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["f0", "f1", "f2", "f3"], dtype=object),
        frame_idx=np.array([10, 20, 30, 40]),
        timestamps_ms=np.array([0.0, 1_000.0, 2_000.0, 3_000.0]),
        scores=np.array(
            [[0.1, 0.9, 0.2, 0.2], [0.0, 0.8, 0.5, 0.3]],
            dtype=np.float32,
        ),
    )


def test_shared_monotonic_adapter_matches_existing_paths_and_scores() -> None:
    video = _scores()
    data = Data([
        _frame(
            str(frame_id),
            frame_idx=int(video.frame_idx[index]),
            timestamp_ms=int(video.timestamps_ms[index]),
        )
        for index, frame_id in enumerate(video.frame_ids)
    ])
    settings = TRAKESettings(lambda_gap=0.0)
    expected = rank_paths((video,), 0.0, 2)

    actual = MonotonicOrderedPathAligner(
        cast(DataService, data), settings
    ).align(_ordered_plan(), (video,), max_paths=2)

    assert [tuple(frame.frame_id for frame in path.frames) for path in actual] == [
        row.frame_ids for row in expected
    ]
    assert [path.score for path in actual] == pytest.approx(
        [row.score for row in expected]
    )


def test_dense_identity_conflict_fails_before_path_materialization() -> None:
    video = _scores()
    frames = [
        _frame(
            str(frame_id),
            frame_idx=int(video.frame_idx[index]),
            timestamp_ms=int(video.timestamps_ms[index]),
        )
        for index, frame_id in enumerate(video.frame_ids)
    ]
    frames[1] = frames[1].model_copy(update={"video_id": "other"})
    aligner = MonotonicOrderedPathAligner(
        cast(DataService, Data(frames)), TRAKESettings()
    )

    with pytest.raises(ValueError, match="mixed canonical video"):
        aligner.align(_ordered_plan(), (video,), max_paths=1)


def test_monotonic_alignment_with_fewer_frames_than_units_returns_no_path() -> None:
    video = VideoEventScores(
        video_id="v1",
        frame_ids=np.array(["f0"], dtype=object),
        frame_idx=np.array([10]),
        timestamps_ms=np.array([0.0]),
        scores=np.array([[0.9], [0.8]], dtype=np.float32),
    )
    aligner = MonotonicOrderedPathAligner(
        cast(DataService, Data([_frame("f0", frame_idx=10)])),
        TRAKESettings(),
    )

    assert aligner.align(_ordered_plan(), (video,), max_paths=1) == ()


class OrderedProvider:
    def __init__(self, scores: tuple[VideoEventScores, ...]) -> None:
        self.scores = scores
        self.plans: list[TemporalQueryPlan] = []

    def acquire(self, plan: TemporalQueryPlan) -> tuple[VideoEventScores, ...]:
        self.plans.append(plan)
        return self.scores


def test_facade_ordered_operation_reports_shared_diagnostics() -> None:
    video = _scores()
    data = Data([
        _frame(
            str(frame_id),
            frame_idx=int(video.frame_idx[index]),
            timestamp_ms=int(video.timestamps_ms[index]),
        )
        for index, frame_id in enumerate(video.frame_ids)
    ])
    provider = OrderedProvider((video,))
    core = TemporalEvidenceCore(
        cast(DataService, data),
        cast(RetrievalService, object()),
        SearchConfig(),
        ordered_provider=provider,
        ordered_aligner=MonotonicOrderedPathAligner(
            cast(DataService, data), TRAKESettings(lambda_gap=0.0)
        ),
    )

    result = core.align_ordered(core.ordered_plan(["first", "second"]), max_paths=1)

    assert provider.plans == [result.plan]
    assert result.diagnostics == {
        "alignment_mode": "ordered_path",
        "query_unit_count": 2,
        "candidate_video_count": 1,
        "path_count": 1,
    }


class FailingOrderedProvider:
    def acquire(self, plan: TemporalQueryPlan):
        del plan
        raise RuntimeError("dense provider unavailable")


class BombOrderedAligner:
    def align(self, *args: Any, **kwargs: Any):
        raise AssertionError("ordered failure must not use another aligner")


def test_dense_failure_does_not_fall_back_to_scene_alignment() -> None:
    core = TemporalEvidenceCore(
        cast(DataService, object()),
        cast(RetrievalService, object()),
        SearchConfig(),
        ordered_provider=FailingOrderedProvider(),
        ordered_aligner=BombOrderedAligner(),
    )

    with pytest.raises(RuntimeError, match="dense provider unavailable"):
        core.align_ordered(_ordered_plan(), max_paths=1)


class CapturingProgressiveProvider:
    def acquire(self, state, unit, filters):
        del unit, filters
        return ProgressiveAcquisition(
            evidence=state.evidence,
            candidate_video_ids=(),
            warnings=(),
            trace=RetrievalTrace(),
        )


class CapturingSceneAligner:
    def __init__(self) -> None:
        self.plans: list[TemporalQueryPlan] = []

    def align(self, plan, evidence):
        del evidence
        self.plans.append(plan)
        return ()


def test_kis_snapshot_is_preserved_in_shared_localization_plan() -> None:
    aligner = CapturingSceneAligner()
    core = TemporalEvidenceCore(
        cast(DataService, object()),
        cast(RetrievalService, object()),
        SearchConfig(),
        progressive_provider=CapturingProgressiveProvider(),
        scene_aligner=aligner,
    )

    core.localize(
        "person standing beside red car",
        search_id=None,
        filters=None,
        task_type=TaskType.KIS,
    )

    assert [unit.text for unit in aligner.plans[0].units] == [
        "person standing beside red car"
    ]
