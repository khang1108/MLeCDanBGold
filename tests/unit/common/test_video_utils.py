from __future__ import annotations

from hcmai.common.schemas import (
    FrameRecord,
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
