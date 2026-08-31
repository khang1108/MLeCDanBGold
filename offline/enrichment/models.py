"""Shared offline enrichment artifact contracts.

This module owns processing state and the deprecated frame-aligned enrichment
projection. Specialist Caption, OCR, Object, Context, and transcript artifacts
remain in their respective owner modules.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from offline.contracts import ContractModel, NonEmptyString


class ProcessingStatus(str, Enum):
    """Status of an offline processing operation."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FrameEnrichment(ContractModel):
    """Deprecated frame-aligned compatibility artifact."""

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


def validate_frame_enrichment(
    enrichments: dict[str, FrameEnrichment],
    canonical_order: list[str],
    frame_store_id: str | None = None,
) -> None:
    """Ensure strict identity and lineage integrity for enrichment artifacts."""

    if len(enrichments) != len(canonical_order):
        raise ValueError(
            f"Enrichment count ({len(enrichments)}) does not match canonical "
            f"frame count ({len(canonical_order)})."
        )
    for frame_id in canonical_order:
        if frame_id not in enrichments:
            raise ValueError(f"Missing enrichment for canonical frame: {frame_id}")
        enrichment = enrichments[frame_id]
        if frame_store_id is not None and enrichment.frame_store_id != frame_store_id:
            raise ValueError(
                f"Lineage mismatch for {frame_id}: expected {frame_store_id}, "
                f"got {enrichment.frame_store_id}"
            )


__all__ = [
    "FrameEnrichment",
    "ProcessingStatus",
    "validate_frame_enrichment",
]
