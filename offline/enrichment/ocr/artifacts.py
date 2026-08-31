"""Build and atomically persist structured OCR evidence.

Raw backend regions remain unchanged. Normalized text and quality are derived
views only; this module does not run OCR inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
import unicodedata

import pandas as pd

from hcmai.common.utils.io import (
    atomic_write,
    read_json,
    write_json,
    write_parquet,
)
from offline.enrichment.bundle import publish_staged_bundle
from offline.enrichment.models import FrameEnrichment, ProcessingStatus
from offline.enrichment.ocr.models import OCREvidence, OCRRegion

from .config import OCRConfig
from .models.entities import (
    Evidence,
    FailureDetail,
    FrameRow,
    OCRRegionResult,
    OCRResult,
    json_safe_ocr_raw,
)


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
        if (
            not isinstance(data.get("frame_id"), str)
            or not data["frame_id"]
            or data["frame_id"].strip() != data["frame_id"]
            or not isinstance(data.get("video_id"), str)
            or not data["video_id"]
            or data["video_id"].strip() != data["video_id"]
            or isinstance(data.get("frame_idx"), bool)
            or not isinstance(data.get("frame_idx"), Integral)
            or isinstance(data.get("timestamp_ms"), bool)
            or not isinstance(data.get("timestamp_ms"), Integral)
        ):
            return None
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
        timestamp_ms=int(frame["timestamp_ms"]),
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
    video_id = str(frame["video_id"])
    frame_idx = int(frame["frame_idx"])
    timestamp_ms = int(frame["timestamp_ms"])
    normalized = normalize_regions(
        result.regions, min_confidence=config.min_region_confidence
    )
    region_rows = [
        OCRRegion(
            frame_id=frame_id,
            video_id=video_id,
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
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
        timestamp_ms=timestamp_ms,
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
        "raw_output": json_safe_ocr_raw(result.raw_output),
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
    report: dict[str, object],
    manifest: dict[str, object],
) -> None:
    """Stage, validate, and publish the complete structured OCR bundle."""

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

    failure_rows = [failures[key] for key in order if key in failures]
    output.mkdir(parents=True, exist_ok=True)
    published = (
        output / "frames.parquet",
        output / "regions.parquet",
        output / "failures.json",
        output / "frame_enrichment.parquet",
        output / "ocr_report.json",
        output / "manifest.json",
    )
    staged = tuple(
        path.with_name(f".{path.name}.staged") for path in published
    )
    try:
        atomic_write(
            staged[0],
            lambda path: write_parquet(frame_table, path, index=False),
        )
        atomic_write(
            staged[1],
            lambda path: write_parquet(region_table, path, index=False),
        )
        atomic_write(staged[2], lambda path: write_json(failure_rows, path))
        atomic_write(
            staged[3],
            lambda path: write_parquet(projection, path, index=False),
        )
        atomic_write(staged[4], lambda path: write_json(report, path))
        atomic_write(staged[5], lambda path: write_json(manifest, path))

        staged_frames = pd.read_parquet(staged[0])
        staged_regions = pd.read_parquet(staged[1])
        staged_projection = pd.read_parquet(staged[3])
        if staged_frames.columns.tolist() != list(OCREvidence.model_fields):
            raise ValueError("staged OCR frames have an invalid schema")
        if staged_regions.columns.tolist() != list(OCRRegion.model_fields):
            raise ValueError("staged OCR regions have an invalid schema")
        if staged_projection.columns.tolist() != list(FrameEnrichment.model_fields):
            raise ValueError("staged OCR projection has an invalid schema")

        expected_order = [frame_id for frame_id in order if frame_id in rows]
        if staged_frames["frame_id"].tolist() != expected_order:
            raise ValueError("staged OCR frames changed canonical order")
        if staged_projection["frame_id"].tolist() != expected_order:
            raise ValueError("staged OCR projection changed canonical order")

        parsed_frames: dict[str, OCREvidence] = {}
        for data in staged_frames.astype(object).where(
            staged_frames.notna(), None
        ).to_dict(orient="records"):
            row = OCREvidence.model_validate(data)
            parsed_frames[row.frame_id] = row
        parsed_regions: dict[str, list[OCRRegion]] = {}
        for data in staged_regions.astype(object).where(
            staged_regions.notna(), None
        ).to_dict(orient="records"):
            region = OCRRegion.model_validate(data)
            parsed_regions.setdefault(region.frame_id, []).append(region)
        for frame_id, row in parsed_frames.items():
            frame_regions = parsed_regions.get(frame_id, [])
            if len(frame_regions) != row.region_count:
                raise ValueError(
                    f"staged OCR region_count mismatch for {frame_id}"
                )
            for region_order, region in enumerate(frame_regions):
                if (
                    region.region_order != region_order
                    or region.region_id != f"{frame_id}:{region_order}"
                    or region.video_id != row.video_id
                    or region.frame_idx != row.frame_idx
                    or region.timestamp_ms != row.timestamp_ms
                ):
                    raise ValueError(
                        f"staged OCR region identity mismatch for {frame_id}"
                    )
        if set(parsed_regions).difference(parsed_frames):
            raise ValueError("staged OCR regions reference an unknown frame")

        for data in staged_projection.astype(object).where(
            staged_projection.notna(), None
        ).to_dict(orient="records"):
            objects = data.get("objects")
            to_list = getattr(objects, "tolist", None)
            if callable(to_list):
                data["objects"] = to_list()
            FrameEnrichment.model_validate(data)
        if read_json(staged[2]) != failure_rows:
            raise ValueError("staged OCR failures failed validation")
        if read_json(staged[4]) != report:
            raise ValueError("staged OCR report failed validation")
        if read_json(staged[5]) != manifest:
            raise ValueError("staged OCR manifest failed validation")

        versions = {row.artifact_version for row in rows.values()}
        lineages = {row.frame_store_id for row in rows.values()}
        if len(versions) > 1 or len(lineages) > 1:
            raise ValueError("OCR bundle has mixed version or lineage")
        if versions and manifest.get("artifact_version") not in versions:
            raise ValueError("OCR manifest artifact_version mismatch")
        if lineages and manifest.get("frame_store_id") not in lineages:
            raise ValueError("OCR manifest frame_store_id mismatch")

        publish_staged_bundle(staged, published)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
