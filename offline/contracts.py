"""Pydantic primitives and canonical models shared by offline artifact stages.

These contracts deliberately stay outside runtime packages so offline stages
can validate published artifacts without importing runtime-private modules.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ContractModel(BaseModel):
    """Reject unknown artifact fields and normalize surrounding whitespace."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


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


__all__ = ["ContractModel", "FrameArtifact", "NonEmptyString"]
