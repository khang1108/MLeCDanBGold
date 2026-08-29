"""Public catalog projections assembled from canonical frame evidence.

The catalog is a read-only API view. It preserves canonical frame identity and
keeps missing specialist evidence distinct from an evaluated empty value.
"""

from __future__ import annotations

from pydantic import Field

from .base import ContractModel, NonEmptyString
from .transcript import TranscriptSegment


class FrameCatalogEntry(ContractModel):
    """One keyframe and its lightweight, inspectable evidence projection.

    ``objects=None`` means object evidence was not loaded or available, while
    an empty mapping means object detection completed with no retained labels.
    ASR remains segment-native, so every segment containing the frame timestamp
    is retained instead of choosing an arbitrary overlapping segment.
    """

    video_id: NonEmptyString
    frame_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    caption: str | None = None
    ocr: str | None = None
    objects: dict[NonEmptyString, int] | None = None
    title: str | None = None
    asr_segments: list[TranscriptSegment] = Field(default_factory=list)
    video_url: str | None = None
