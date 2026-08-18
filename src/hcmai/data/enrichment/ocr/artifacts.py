"""Build and atomically persist structured OCR evidence.

Raw backend regions remain unchanged. Normalized text and quality are derived
views only; this module does not run OCR inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import unicodedata

import pandas as pd

from hcmai.common.schemas import (
    FrameEnrichment,
    OCREvidence,
    OCRRegion,
    ProcessingStatus,
)
from hcmai.common.utils.io import atomic_write, write_json, write_parquet

from .config import OCRConfig
from .models.entities import Evidence, FailureDetail, FrameRow, OCRRegionResult, OCRResult


@dataclass(frozen=True)
class NormalizedRegions:
    """Deterministic derived OCR text and its inspectable quality inputs."""

    text: str | None
    usable_region_count: int
    quality_score: float


def _normalized_line(text: str) -> str:
    """Apply NFC before collapsing and trimming whitespace."""

    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_regions(
    regions: tuple[OCRRegionResult, ...] | list[OCRRegionResult],
    *,
    min_confidence: float,
) -> NormalizedRegions:
    """Derive ordered context text without modifying source OCR regions."""

    retained: list[str] = []
    seen: set[str] = set()
    confidences: list[float] = []

    for region in regions:
        if region.confidence is not None:
            confidence = float(region.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("OCR confidence must be finite and in [0, 1]")
            confidences.append(confidence)

        line = _normalized_line(region.text)
        if region.confidence is not None and region.confidence < min_confidence:
            continue
        if not any(character.isalnum() for character in line):
            continue

        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        retained.append(line)

    text = "\n".join(retained) or None
    if text is None:
        return NormalizedRegions(None, 0, 0.0)

    usable_ratio = len(retained) / max(1, len(regions))
    mean_confidence = sum(confidences) / len(confidences) if confidences else 1.0
    return NormalizedRegions(
        text=text,
        usable_region_count=len(retained),
        quality_score=min(1.0, usable_ratio * mean_confidence),
    )


def valid_ocr(
    data: FrameRow,
    config: OCRConfig,
    *,
    frame_store_id: str | None,
    model_revision: str | None,
) -> OCREvidence | None:
    """Return a completed frame row only when all reusable lineage matches."""

    try:
        values = {
            key: None if isinstance(value, float) and math.isnan(value) else value
            for key, value in dict(data).items()
        }
        row = OCREvidence.model_validate(values)
    except Exception:
        return None

    valid = (
        row.status == ProcessingStatus.COMPLETED
        and row.error_code is None
        and row.error_message is None
        and row.artifact_version == config.artifact_version
        and row.model_name == config.model_name
        and row.model_revision == model_revision
        and row.frame_store_id == frame_store_id
    )
    return row if valid else None


def failure_row(
    frame: FrameRow,
    config: OCRConfig,
    stage: str,
    error: Exception,
    *,
    frame_store_id: str | None,
    model_revision: str | None,
) -> tuple[OCREvidence, FailureDetail]:
    """Build one bounded failed OCR row and machine-readable diagnostic."""

    message = " ".join(str(error).split())[:300] or type(error).__name__
    code = type(error).__name__
    frame_id = str(frame["frame_id"])
    row = OCREvidence(
        frame_id=frame_id,
        video_id=str(frame["video_id"]),
        frame_idx=int(frame["frame_idx"]),
        frame_store_id=frame_store_id,
        artifact_version=config.artifact_version,
        model_name=config.model_name,
        model_revision=model_revision,
        status=ProcessingStatus.FAILED,
        error_code=code,
        error_message=message,
    )
    return row, {
        "frame_id": frame_id,
        "artifact_version": config.artifact_version,
        "processing_stage": stage,
        "exception_category": code,
        "error_message": message,
    }


def parsed_row(
    frame: FrameRow,
    result: object,
    config: OCRConfig,
    *,
    frame_store_id: str | None,
    model_revision: str | None,
) -> tuple[OCREvidence, list[OCRRegion], Evidence]:
    """Validate a structured backend result and preserve every raw region."""

    if not isinstance(result, OCRResult) or not isinstance(result.text, str):
        raise TypeError("OCR backend returned a malformed result")
    if not isinstance(result.regions, tuple) or any(
        not isinstance(region, OCRRegionResult) for region in result.regions
    ):
        raise TypeError("OCR backend returned malformed regions")

    frame_id = str(frame["frame_id"])
    frame_idx = int(frame["frame_idx"])
    normalized = normalize_regions(
        result.regions, min_confidence=config.min_region_confidence
    )
    region_rows = [
        OCRRegion(
            frame_id=frame_id,
            frame_idx=frame_idx,
            region_id=f"{frame_id}:{order}",
            region_order=order,
            text=region.text,
            confidence=region.confidence,
            x_min=region.x_min,
            y_min=region.y_min,
            x_max=region.x_max,
            y_max=region.y_max,
        )
        for order, region in enumerate(result.regions)
    ]
    raw_text = "\n".join(
        region.text for region in result.regions if region.text != ""
    ) or None
    row = OCREvidence(
        frame_id=frame_id,
        video_id=str(frame["video_id"]),
        frame_idx=frame_idx,
        raw_text=raw_text,
        normalized_text=normalized.text,
        quality_score=normalized.quality_score,
        region_count=len(region_rows),
        frame_store_id=frame_store_id,
        artifact_version=config.artifact_version,
        model_name=config.model_name,
        model_revision=model_revision,
    )
    evidence: Evidence = {
        "frame_id": frame_id,
        "raw_output": result.raw_output,
        "usable_region_count": normalized.usable_region_count,
    }
    return row, region_rows, evidence


def _legacy_projection(row: OCREvidence, config: OCRConfig) -> FrameEnrichment:
    """Build the temporary flat OCR view required by existing consumers."""

    return FrameEnrichment(
        frame_id=row.frame_id,
        frame_store_id=row.frame_store_id,
        ocr_text=row.normalized_text,
        enrichment_version=config.enrichment_version,
        model_name=row.model_name,
        status=row.status,
        error_message=row.error_message,
    )


def write_ocr_artifacts(
    output: Path,
    order: list[str],
    rows: dict[str, OCREvidence],
    regions: dict[str, list[OCRRegion]],
    failures: dict[str, FailureDetail],
    config: OCRConfig,
) -> None:
    """Atomically write canonical frame order and backend region order."""

    frame_table = pd.DataFrame(
        [rows[key].model_dump(mode="json") for key in order if key in rows],
        columns=list(OCREvidence.model_fields),
    )
    region_table = pd.DataFrame(
        [
            region.model_dump(mode="json")
            for frame_id in order
            for region in regions.get(frame_id, [])
        ],
        columns=list(OCRRegion.model_fields),
    )
    projection = pd.DataFrame(
        [
            _legacy_projection(rows[key], config).model_dump(mode="json")
            for key in order
            if key in rows
        ],
        columns=list(FrameEnrichment.model_fields),
    )

    atomic_write(output / "frames.parquet", lambda path: write_parquet(frame_table, path, index=False))
    atomic_write(output / "regions.parquet", lambda path: write_parquet(region_table, path, index=False))
    atomic_write(output / "frame_enrichment.parquet", lambda path: write_parquet(projection, path, index=False))
    atomic_write(
        output / "failures.json",
        lambda path: write_json([failures[key] for key in order if key in failures], path),
    )
