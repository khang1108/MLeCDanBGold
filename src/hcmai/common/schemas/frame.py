from __future__ import annotations

from pydantic import Field, field_validator

from .base import ContractModel, NonEmptyString
from .enum import ProcessingStatus


class FrameRecord(ContractModel):
    """Canonical metadata for one searchable frame."""

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


class FrameEnrichment(ContractModel):
    """Offline caption, OCR, ASR, and object metadata for a frame."""

    frame_id: NonEmptyString
    frame_store_id: NonEmptyString | None = None
    caption: NonEmptyString | None = None
    detailed_caption: NonEmptyString | None = None
    ocr_text: NonEmptyString | None = None
    asr_text: NonEmptyString | None = None
    source_segment_ids: list[NonEmptyString] = Field(default_factory=list)
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


def validate_frame_enrichment(
    enrichments: dict[str, FrameEnrichment],
    canonical_order: list[str],
    frame_store_id: str | None = None,
) -> None:
    """Ensure strict integrity for enrichment artifacts."""
    if len(enrichments) != len(canonical_order):
        raise ValueError(
            f"Enrichment count ({len(enrichments)}) does not match canonical frame count ({len(canonical_order)})."
        )
    for frame_id in canonical_order:
        if frame_id not in enrichments:
            raise ValueError(f"Missing enrichment for canonical frame: {frame_id}")
        enrichment = enrichments[frame_id]
        if frame_store_id is not None and enrichment.frame_store_id != frame_store_id:
            raise ValueError(
                f"Lineage mismatch for {frame_id}: expected {frame_store_id}, got {enrichment.frame_store_id}"
            )

