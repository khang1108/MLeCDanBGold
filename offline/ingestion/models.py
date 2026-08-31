"""Offline validation model for canonical frame artifacts.

``FrameArtifact`` preserves the historical Parquet columns and validation used
by ingestion. Runtime code projects those artifacts to ``hcmai.corpus.Frame``
and must not expose this provenance-heavy model.
"""

from __future__ import annotations

from pydantic import Field

from offline.contracts import ContractModel, NonEmptyString


class FrameArtifact(ContractModel):
    """Canonical metadata row written to the frame Parquet artifact."""

    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    keyframe_order: int | None = Field(default=None, ge=1)
    timestamp_ms: int = Field(ge=0)
    fps: float | None = Field(default=None, gt=0)
    image_path: NonEmptyString
    thumbnail_path: NonEmptyString | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    shot_id: NonEmptyString | None = None
    event_id: NonEmptyString | None = None
    is_anchor: bool = True
    pts: int | None = None
    time_base: NonEmptyString | None = None
    motion_score: float = Field(default=0.0, ge=0)
    shot_score: float = Field(default=0.0, ge=0, le=1)
    event_score: float = Field(default=0.0, ge=0, le=1)
    selection_reasons: tuple[NonEmptyString, ...] = ()


__all__ = ["FrameArtifact"]
