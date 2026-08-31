"""Build reproducible OCR coverage and quality summaries.

Coverage metrics describe artifact contents; they are never presented as OCR
accuracy because no ground-truth transcription is available here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from offline.enrichment.models import ProcessingStatus
from offline.enrichment.ocr.models import OCREvidence, OCRRegion

from .artifacts import normalize_regions
from .config import OCRConfig
from .models.entities import Evidence, FailureDetail, OCRRegionResult


def build_ocr_report(
    config: OCRConfig,
    path: Path,
    root: Path,
    rows: dict[str, OCREvidence],
    regions: dict[str, list[OCRRegion]],
    evidence: dict[str, Evidence],
    failures: dict[str, FailureDetail],
    old: dict[str, Any],
    started: datetime,
    elapsed: float,
    input_count: int,
    processed: int,
    skipped: int,
    retried: int,
    revision: str | None,
    disabled: int,
    *,
    frame_store_id: str | None,
) -> dict[str, Any]:
    """Summarize raw, normalized, and region OCR evidence independently."""

    complete = sum(row.status == ProcessingStatus.COMPLETED for row in rows.values())
    raw_text_count = sum(row.raw_text is not None for row in rows.values())
    normalized_text_count = sum(row.normalized_text is not None for row in rows.values())
    frames_with_regions = sum(row.region_count > 0 for row in rows.values())
    raw_region_count = sum(row.region_count for row in rows.values())
    usable_region_count = sum(
        normalize_regions(
            tuple(
                OCRRegionResult(
                    text=region.text,
                    confidence=region.confidence,
                    x_min=region.x_min,
                    y_min=region.y_min,
                    x_max=region.x_max,
                    y_max=region.y_max,
                )
                for region in regions.get(frame_id, [])
            ),
            min_confidence=config.min_region_confidence,
        ).usable_region_count
        for frame_id in rows
    )
    quality_scores = [
        row.quality_score
        for row in rows.values()
        if row.status == ProcessingStatus.COMPLETED
    ]
    ratio = lambda count: count / input_count if input_count else 0.0

    return {
        "report_version": "ocr_report.v2",
        "artifact_version": config.artifact_version,
        "enrichment_version": config.enrichment_version,
        "dataset_version": config.dataset_version,
        "frame_store_id": frame_store_id,
        "input_parquet_path": str(path),
        "dataset_root": str(root),
        "backend": config.backend,
        "checkpoint": config.checkpoint,
        "resolved_revision": revision,
        "enabled": config.enabled,
        "device": config.device,
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "runtime_settings": asdict(config),
        "total_frames": input_count,
        "processed_frames": processed,
        "completed_frames": complete,
        "failed_frames": len(rows) - complete,
        "skipped_frames": skipped,
        "retried_frames": retried,
        "disabled_frames": disabled,
        "frames_with_raw_text": raw_text_count,
        "frames_with_normalized_text": normalized_text_count,
        "frames_with_regions": frames_with_regions,
        "raw_region_count": raw_region_count,
        "usable_region_count": usable_region_count,
        "mean_quality_score": (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        ),
        "raw_text_coverage_rate": ratio(raw_text_count),
        "normalized_text_coverage_rate": ratio(normalized_text_count),
        "region_coverage_rate": ratio(frames_with_regions),
        "failure_rate": ratio(len(rows) - complete),
        "error_counts": dict(
            Counter(item["exception_category"] for item in failures.values())
        ),
        "raw_output_available": any(
            item.get("raw_output") is not None for item in evidence.values()
        ),
        "raw_evidence": [evidence[key] for key in rows if key in evidence],
        "normalization_policy": (
            "Unicode NFC; collapse whitespace; confidence filter; require Unicode "
            "alphanumeric; case-insensitive ordered deduplication; newline join."
        ),
        "start_time": started.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "elapsed_time_sec": elapsed,
        "manual_review": old.get(
            "manual_review",
            {"sample_count": 0, "status": "pending", "summary": "Human review pending."},
        ),
        "known_limitations": [
            "Coverage and quality heuristics are not OCR accuracy.",
            "Florence-2 has no calibrated OCR confidence.",
        ],
    }
