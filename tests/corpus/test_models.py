"""Tests for the minimal frozen runtime corpus dataclasses."""

from dataclasses import FrozenInstanceError, fields

import pytest

from hcmai.corpus import Frame, TranscriptSegment, VideoMetadata


def test_frame_has_the_runtime_contract() -> None:
    """Frame exposes identity, timing, image paths, and optional FPS."""
    assert [field.name for field in fields(Frame)] == [
        "frame_id",
        "video_id",
        "frame_idx",
        "timestamp_ms",
        "image_path",
        "thumbnail_path",
        "fps",
    ]


def test_runtime_models_are_slotted() -> None:
    """Runtime models do not carry an instance dictionary."""
    assert hasattr(Frame, "__slots__")
    assert hasattr(TranscriptSegment, "__slots__")
    assert hasattr(VideoMetadata, "__slots__")
    assert "__dict__" not in Frame.__slots__
    assert "__dict__" not in TranscriptSegment.__slots__
    assert "__dict__" not in VideoMetadata.__slots__


def test_frame_is_immutable() -> None:
    """Frame fields cannot be changed after construction."""
    frame = Frame("f1", "v1", 1, 1000, "keyframes/f1.jpg")

    with pytest.raises(FrozenInstanceError):
        frame.timestamp_ms = 2000


@pytest.mark.parametrize(
    "model, instance",
    [
        (TranscriptSegment, TranscriptSegment("s1", "v1", 0, 0, 1000, "hello")),
        (VideoMetadata, VideoMetadata("v1")),
    ],
)
def test_other_runtime_models_are_immutable(model: type, instance: object) -> None:
    """Transcript and metadata models use the same frozen contract."""
    field_name = fields(model)[0].name

    with pytest.raises(FrozenInstanceError):
        setattr(instance, field_name, "changed")
