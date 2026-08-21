from __future__ import annotations

from hcmai.common.schemas import (
    FrameRecord,
    SearchResult,
    SearchScores,
    TRAKESubmission,
    VQARetrievalEvidence,
    VQASubmission,
)
from hcmai.common.utils.video import derive_fps, format_video_id, official_frame_idx


def test_format_video_id_standard_paths() -> None:
    assert (
        format_video_id("Videos_L26_b/videos/L26_V196/001.mp4")
        == "L26_b.L26_V196.001"
    )
    assert (
        format_video_id("data/Videos_L26_b/videos/L26_V196/001.mp4")
        == "L26_b.L26_V196.001"
    )
    assert (
        format_video_id("Keyframes_L26_b/keyframes/L26_V196/001/0001.jpg")
        == "L26_b.L26_V196.001"
    )
    assert (
        format_video_id("Videos_L26_b.L26_V196.001")
        == "L26_b.L26_V196.001"
    )
    assert (
        format_video_id("L26_b.L26_V196.001")
        == "L26_b.L26_V196.001"
    )
    assert (
        format_video_id("Videos_L26_b/videos/L26_V196/001")
        == "L26_b.L26_V196.001"
    )


def test_format_video_id_fallback() -> None:
    assert (
        format_video_id("", fallback_path="Videos_L26_b/videos/L26_V196/001.mp4")
        == "L26_b.L26_V196.001"
    )
    assert format_video_id("TEST_V001") == "TEST_V001"


def test_derive_fps() -> None:
    assert derive_fps(None) == 25.0
    frame_default = FrameRecord(
        frame_id="f1",
        video_id="v1",
        frame_idx=0,
        timestamp_ms=0,
        image_path="1.jpg",
        width=100,
        height=100,
    )
    assert derive_fps(frame_default) == 25.0

    frame_with_fps = FrameRecord(
        frame_id="f1",
        video_id="v1",
        frame_idx=0,
        timestamp_ms=0,
        image_path="1.jpg",
        width=100,
        height=100,
        fps=29.97,
    )
    assert derive_fps(frame_with_fps) == 29.97


def test_official_frame_idx_uses_btc_coordinate_without_recomputing_from_time() -> None:
    frame = FrameRecord(
        frame_id="f1",
        video_id="v1",
        frame_idx=7,
        timestamp_ms=33,
        image_path="1.jpg",
        width=100,
        height=100,
        fps=30.0,
    )

    assert official_frame_idx(frame) == 7


def test_search_result_streaming_fields() -> None:
    result = SearchResult(
        rank=1,
        frame_id="L26_V196_001_keyframe_000001",
        frame_ids=["L26_V196_001_keyframe_000001"],
        video_id="L26_b.L26_V196.001",
        frame_idx=150,
        fps=25.0,
        timestamp_ms=6000,
        scores=SearchScores(final=0.9),
    )
    assert result.video_id == "L26_b.L26_V196.001"
    assert result.fps == 25.0
    assert result.frame_idx == 150
    assert result.frame_ids == ["L26_V196_001_keyframe_000001"]


def test_trake_submission_streaming_fields() -> None:
    submission = TRAKESubmission(
        rank=1,
        video_id="L26_b.L26_V196.001",
        frame_ids=["f1", "f2"],
        frame_idxs=[10, 20],
        timestamps_ms=[400, 800],
        fps=25.0,
    )
    assert submission.video_id == "L26_b.L26_V196.001"
    assert submission.fps == 25.0
    assert submission.frame_idxs == [10, 20]
    assert submission.frame_ids == ["f1", "f2"]


def test_vqa_submission_streaming_fields() -> None:
    submission = VQASubmission(
        rank=1,
        video_id="L26_b.L26_V196.001",
        frame_id="f1",
        frame_ids=["f1"],
        frame_idx=25,
        fps=25.0,
        answer="apple",
        retrieval_score=0.9,
        grounding_score=0.9,
        answer_score=0.9,
        joint_score=0.9,
    )
    assert submission.video_id == "L26_b.L26_V196.001"
    assert submission.fps == 25.0
    assert submission.frame_idx == 25
    assert submission.frame_ids == ["f1"]


def test_vqa_retrieval_evidence_streaming_fields() -> None:
    evidence = VQARetrievalEvidence(
        rank=1,
        video_id="L26_b.L26_V196.001",
        frame_id="f1",
        frame_idx=25,
        fps=25.0,
        timestamp_ms=1000,
        retrieval_score=0.9,
    )
    assert evidence.video_id == "L26_b.L26_V196.001"
    assert evidence.fps == 25.0
    assert evidence.frame_idx == 25
    assert evidence.frame_id == "f1"
    assert evidence.frame_ids == ["f1"]
