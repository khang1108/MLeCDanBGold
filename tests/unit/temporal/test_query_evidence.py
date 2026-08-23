from __future__ import annotations

import pytest

from hcmai.common.schemas import FrameEvidence, FrameRecord, SceneCandidate
from hcmai.temporal.state.evidence import EvaluationState, ProgressiveEvidenceState
from hcmai.temporal.query import SnapshotDiffMode, diff_snapshot


@pytest.mark.parametrize(
    ("previous", "current", "mode", "delta"),
    [
        (None, "Gợi ý đầu tiên", SnapshotDiffMode.FIRST, "Gợi ý đầu tiên"),
        ("A", "A B", SnapshotDiffMode.APPEND, "B"),
        ("A", "A\n   B", SnapshotDiffMode.APPEND, "B"),
        ("A  B", "A B", SnapshotDiffMode.NO_CHANGE, None),
        ("A", "A.", SnapshotDiffMode.NO_CHANGE, None),
        ("A B", "Nội dung viết lại", SnapshotDiffMode.REPLACEMENT, None),
    ],
)
def test_snapshot_diff_contract(previous, current, mode, delta):
    result = diff_snapshot(previous, current)
    assert result.mode is mode
    assert result.delta_text == delta


def test_ten_progressive_snapshots_create_only_ten_deltas():
    previous = None
    snapshot = ""
    deltas = []
    for index in range(10):
        snapshot = f"{snapshot} Gợi ý {index}".strip()
        result = diff_snapshot(previous, snapshot)
        deltas.append(result.delta_text)
        previous = result.normalized_current
    assert deltas == ["Gợi ý 0", *[f"Gợi ý {index}" for index in range(1, 10)]]


def test_empty_snapshot_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        diff_snapshot(None, "  \n")


def test_evaluated_empty_survives_round_trip():
    state = ProgressiveEvidenceState()
    state.mark_evaluated("h0", "video-a")
    restored = ProgressiveEvidenceState.from_dict(state.to_dict())
    assert restored.evaluation_state("h0", "video-a") is EvaluationState.EVALUATED_NO_MATCH
    assert restored.evaluation_state("h1", "video-a") is EvaluationState.UNKNOWN


def test_matched_evidence_marks_pair_evaluated():
    frame = FrameRecord(
        frame_id="f1", video_id="v1", frame_idx=1, timestamp_ms=100,
        image_path="f1.jpg", width=10, height=10,
    )
    item = FrameEvidence(frame=frame, unit_scores={"h0": 0.8}, score=0.8)
    state = ProgressiveEvidenceState()
    state.mark_evaluated("h0", "v1", (item,))
    assert state.evaluation_state("h0", "v1") is EvaluationState.MATCHED
    assert state.is_evaluated("h0", "v1")


def test_orphaned_matched_evidence_fails_invariant():
    frame = FrameRecord(
        frame_id="f1", video_id="v1", frame_idx=1, timestamp_ms=100,
        image_path="f1.jpg", width=10, height=10,
    )
    state = ProgressiveEvidenceState(evidence={
        ("h0", "v1"): (FrameEvidence(frame=frame, score=0.8),),
    })
    with pytest.raises(ValueError, match="must be evaluated"):
        state.validate()


def test_evidence_state_can_be_bounded_to_active_videos():
    first = FrameRecord(
        frame_id="f1", video_id="v1", frame_idx=1, timestamp_ms=100,
        image_path="f1.jpg", width=10, height=10,
    )
    second = first.model_copy(update={"frame_id": "f2", "video_id": "v2"})
    state = ProgressiveEvidenceState()
    state.mark_evaluated("h0", "v1", (FrameEvidence(frame=first, score=0.8),))
    state.mark_evaluated("h0", "v2", (FrameEvidence(frame=second, score=0.7),))
    state.retain_videos({"v1"})
    assert state.is_evaluated("h0", "v1")
    assert not state.is_evaluated("h0", "v2")
    assert all(video_id == "v1" for _, video_id in state.evidence)


def test_scene_candidate_rejects_cross_video_and_out_of_range_evidence():
    frame = FrameRecord(
        frame_id="f1", video_id="v1", frame_idx=1, timestamp_ms=1_000,
        image_path="f1.jpg", width=10, height=10,
    )
    item = FrameEvidence(frame=frame, score=0.8)
    with pytest.raises(ValueError, match="scene.video_id"):
        SceneCandidate(
            scene_id="bad-video", video_id="v2", start_ms=0, end_ms=2_000,
            evidence=(item,),
        )
    with pytest.raises(ValueError, match="inside scene range"):
        SceneCandidate(
            scene_id="bad-time", video_id="v1", start_ms=0, end_ms=500,
            evidence=(item,),
        )
    with pytest.raises(ValueError):
        SceneCandidate(
            scene_id="empty", video_id="v1", start_ms=0, end_ms=500,
            evidence=(),
        )
