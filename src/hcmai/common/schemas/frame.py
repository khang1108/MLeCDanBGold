from __future__ import annotations

from hcmai.common.schemas import ContractModel, NonEmptyString, ProcessingStatus
from pydantic import Field, field_validator

class FrameRecord(ContractModel):
    """Canonical metadata for one searchable frame."""

    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    keyframe_order: int | None = Field(default=None, ge=1)
    timestamp_ms: int = Field(ge=0)
    image_path: NonEmptyString
    thumbnail_path: NonEmptyString | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    shot_id: NonEmptyString | None = None
    is_anchor: bool = True


class FrameEnrichment(ContractModel):
    """Offline caption, OCR, ASR, and object metadata for a frame."""

    frame_id: NonEmptyString
    caption: NonEmptyString | None = None
    detailed_caption: NonEmptyString | None = None
    ocr_text: NonEmptyString | None = None
    asr_text: NonEmptyString | None = None
    enrichment_version: NonEmptyString | None = None
    objects: list[NonEmptyString] = Field(default_factory=list)
    model_name: NonEmptyString
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_message: NonEmptyString | None = None

    @field_validator("objects")
    @classmethod
    def deduplicate_objects(cls, objects: list[str]) -> list[str]:
        """Remove duplicate labels while preserving their original order."""

        return list(dict.fromkeys(objects))
