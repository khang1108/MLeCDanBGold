"""Runtime corpus domain models.

These frozen, slotted dataclasses represent the minimal runtime reads used by
the corpus layer.  They intentionally do not replace the Pydantic models used
to validate offline artifacts or carry artifact-generation provenance.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Frame:
    """Canonical runtime frame identity and its image paths.

    ``frame_idx`` is the competition-facing frame coordinate; it must remain
    distinct from keyframe order and the internal ``frame_id``.
    ``fps`` is optional because legacy frame artifacts did not retain it.
    """

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    image_path: str
    thumbnail_path: str | None = None
    fps: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """Timestamped runtime transcript segment belonging to a video."""

    segment_id: str
    video_id: str
    segment_index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Minimal runtime metadata identifying a source video."""

    video_id: str
    title: str | None = None
    video_url: str | None = None


__all__ = ["Frame", "TranscriptSegment", "VideoMetadata"]
