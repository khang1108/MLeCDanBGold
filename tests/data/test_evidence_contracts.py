"""Tests for shared specialist-evidence contracts and legacy compatibility."""

import pytest
from pydantic import ValidationError

from offline.enrichment.caption.models import CaptionEvidence
from offline.enrichment.caption.models import usable_completed_text as usable_caption_text
from offline.enrichment.context.models import FrameContext
from offline.enrichment.models import FrameEnrichment, ProcessingStatus
from offline.enrichment.objects.models import ObjectDetection, ObjectEvidence
from offline.enrichment.ocr.models import OCREvidence, OCRRegion
from offline.enrichment.ocr.models import usable_completed_text as usable_ocr_text


def usable_completed_text(row: CaptionEvidence | OCREvidence) -> str | None:
    """Dispatch to the specialist owner's completed-text helper."""

    if isinstance(row, CaptionEvidence):
        return usable_caption_text(row)
    return usable_ocr_text(row)


@pytest.mark.parametrize(
    ("contract", "values"),
    [
        (
            CaptionEvidence,
            {
                "artifact_version": "caption-v1",
                "model_name": "captioner",
            },
        ),
        (
            OCREvidence,
            {"artifact_version": "ocr-v1", "model_name": "ocr"},
        ),
        (
            ObjectEvidence,
            {"artifact_version": "object-v1"},
        ),
        (
            FrameContext,
            {
                "context_version": "frame-context-v1",
                "caption_version": "caption-v1",
                "ocr_version": "ocr-v1",
                "object_version": "object-v1",
            },
        ),
    ],
)
def test_frame_aligned_contracts_preserve_full_canonical_identity(
    contract: type[CaptionEvidence | OCREvidence | ObjectEvidence | FrameContext],
    values: dict[str, object],
) -> None:
    """Keep the organizer timestamp alongside every frame-aligned identity."""

    row = contract(
        frame_id="f1",
        video_id="v1",
        frame_idx=10,
        timestamp_ms=1_234,
        **values,
    )

    assert row.timestamp_ms == 1_234


@pytest.mark.parametrize(
    "contract",
    [
        CaptionEvidence,
        OCRRegion,
        OCREvidence,
        ObjectDetection,
        ObjectEvidence,
        FrameContext,
    ],
)
def test_public_evidence_contracts_explain_their_semantics(
    contract: type[
        CaptionEvidence
        | OCRRegion
        | OCREvidence
        | ObjectDetection
        | ObjectEvidence
        | FrameContext
    ],
) -> None:
    """Keep public schema intent discoverable at the contract boundary."""

    assert contract.__doc__ is not None
    assert len(contract.__doc__.split()) >= 8


def test_ocr_region_preserves_full_parent_frame_identity() -> None:
    """Keep video and timestamp identity on every authoritative OCR region."""

    region = OCRRegion(
        frame_id="f1",
        video_id="v1",
        frame_idx=10,
        timestamp_ms=1_234,
        region_id="f1:0",
        region_order=0,
        text="HCMAI",
        x_min=0.1,
        y_min=0.2,
        x_max=0.8,
        y_max=0.9,
    )

    assert (region.video_id, region.timestamp_ms) == ("v1", 1_234)


def test_object_evidence_preserves_repeated_instances() -> None:
    """Keep raw repeated detections while accepting thresholded counts."""

    detections = [
        ObjectDetection(
            label="person",
            confidence=0.9,
            x_min=0.1,
            y_min=0.1,
            x_max=0.2,
            y_max=0.4,
        ),
        ObjectDetection(
            label="person",
            confidence=0.4,
            x_min=0.3,
            y_min=0.1,
            x_max=0.4,
            y_max=0.4,
        ),
    ]
    row = ObjectEvidence(
        frame_id="L01_V001:0000",
        video_id="L01_V001",
        frame_idx=10,
        timestamp_ms=1_000,
        detections=detections,
        counts={"person": 1},
        summary="person x1",
        detection_count=2,
        artifact_version="object-v1",
        status=ProcessingStatus.COMPLETED,
    )

    assert row.counts == {"person": 1}
    assert [detection.label for detection in row.detections] == ["person", "person"]


def test_object_counts_cannot_exceed_raw_detection_multiplicity() -> None:
    """Reject derived counts that invent detections absent from raw evidence."""

    detection = ObjectDetection(
        label="person",
        confidence=0.9,
        x_min=0.1,
        y_min=0.1,
        x_max=0.2,
        y_max=0.4,
    )

    with pytest.raises(ValidationError, match="raw detection multiplicity"):
        ObjectEvidence(
            frame_id="f1",
            video_id="v1",
            frame_idx=1,
            timestamp_ms=1_000,
            detections=[detection],
            counts={"person": 2},
            detection_count=1,
            artifact_version="object-v1",
        )


def test_failed_evidence_requires_diagnostics() -> None:
    """Require an error code and message whenever a specialist row fails."""

    with pytest.raises(ValidationError, match="failed evidence requires"):
        CaptionEvidence(
            frame_id="f1",
            video_id="v1",
            frame_idx=1,
            timestamp_ms=1_000,
            artifact_version="caption-v1",
            model_name="captioner",
            status=ProcessingStatus.FAILED,
        )


def test_completed_empty_evidence_is_valid_but_not_usable_text() -> None:
    """Represent evaluated-empty evidence without turning it into a match."""

    caption = CaptionEvidence(
        frame_id="f1",
        video_id="v1",
        frame_idx=1,
        timestamp_ms=1_000,
        text="  ",
        artifact_version="caption-v1",
        model_name="captioner",
    )
    ocr = OCREvidence(
        frame_id="f1",
        video_id="v1",
        frame_idx=1,
        timestamp_ms=1_000,
        normalized_text="",
        artifact_version="ocr-v1",
        model_name="ocr",
    )

    assert usable_completed_text(caption) is None
    assert usable_completed_text(ocr) is None


def test_region_bounds_preserve_raw_geometry() -> None:
    """Accept canonical OCR geometry and reject reversed coordinates."""

    region = OCRRegion(
        frame_id="f1",
        video_id="v1",
        frame_idx=1,
        timestamp_ms=1_000,
        region_id="f1:r0",
        region_order=0,
        text="HCMAI",
        x_min=0.1,
        y_min=0.2,
        x_max=0.8,
        y_max=0.9,
    )

    assert region.text == "HCMAI"
    with pytest.raises(ValidationError, match="maximum coordinates"):
        OCRRegion(
            frame_id="f1",
            video_id="v1",
            frame_idx=1,
            timestamp_ms=1_000,
            region_id="f1:r1",
            region_order=1,
            text="invalid",
            x_min=0.8,
            y_min=0.2,
            x_max=0.1,
            y_max=0.9,
        )


def test_frame_context_records_upstream_versions() -> None:
    """Keep deterministic context lineage separate for every specialist."""

    row = FrameContext(
        frame_id="f1",
        video_id="v1",
        frame_idx=1,
        timestamp_ms=1_000,
        context_text="[CAPTION]\nA person runs.",
        caption_text="A person runs.",
        ocr_text=None,
        object_summary=None,
        context_version="frame-context-v1",
        caption_version="caption-v1",
        ocr_version="ocr-v1",
        object_version="object-v1",
        frame_store_id="btc-v1",
    )

    assert row.model_dump()["context_version"] == "frame-context-v1"
    assert row.model_dump()["caption_version"] == "caption-v1"
    assert row.model_dump()["ocr_version"] == "ocr-v1"
    assert row.model_dump()["object_version"] == "object-v1"


def test_legacy_projection_preserves_repeated_object_labels() -> None:
    """Keep multiplicity visible to retrieval during compatibility migration."""

    row = FrameEnrichment(
        frame_id="f1",
        objects=["person", "person", "car"],
        model_name="compatibility-projection",
    )

    assert row.objects == ["person", "person", "car"]
