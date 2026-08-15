"""Tạo báo cáo cho quá trình OCR (Trích xuất chữ trong ảnh).

Tổng hợp thông tin về tiến trình xử lý, các ảnh đã nhận diện và thống kê lỗi (nếu có).

Các tính năng chính:
1. Tracking tiến trình: Đếm số khung hình thành công và thất bại trong quá trình OCR.
2. Đánh giá chất lượng: Thống kê trung bình lượng văn bản tìm thấy trên mỗi frame.
3. Xuất báo cáo (Export): Đẩy các chỉ số (metrics) ra hệ thống monitor hoặc lưu dưới dạng văn bản."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hcmai.common.schemas import FrameEnrichment, ProcessingStatus

from .config import OCRConfig
from .models.entities import Evidence, FailureDetail


def build_ocr_report(
    config: OCRConfig,
    path: Path,
    root: Path,
    rows: dict[str, FrameEnrichment],
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
) -> dict[str, Any]:
    """Build reproducible OCR coverage and failure evidence."""
    complete = sum(
        row.status == ProcessingStatus.COMPLETED for row in rows.values()
    )
    text_count = sum(row.ocr_text is not None for row in rows.values())
    confidence = [
        float(value)
        for item in evidence.values()
        if isinstance((value := item.get("confidence")), (int, float))
    ]
    ratio = lambda count: count / input_count if input_count else 0.0
    summary = (
        {
            "min": min(confidence),
            "max": max(confidence),
            "mean": sum(confidence) / len(confidence),
        }
        if confidence
        else None
    )
    return {
        "report_version": "ocr_report.v1",
        "artifact_version": "frame_enrichment.v1",
        "enrichment_version": config.enrichment_version,
        "dataset_version": config.dataset_version,
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
        "frames_with_text": text_count,
        "empty_text_frames": complete - text_count,
        "failed_frames": len(rows) - complete,
        "skipped_frames": skipped,
        "retried_frames": retried,
        "disabled_frames": disabled,
        "text_coverage_rate": ratio(text_count),
        "empty_text_rate": ratio(complete - text_count),
        "failure_rate": ratio(len(rows) - complete),
        "error_counts": dict(
            Counter(item["exception_category"] for item in failures.values())
        ),
        "confidence_available": bool(confidence),
        "confidence_summary": summary,
        "raw_output_available": any(
            item.get("raw_output") is not None for item in evidence.values()
        ),
        "raw_evidence": [evidence[key] for key in rows if key in evidence],
        "normalization_policy": (
            "Unicode NFC; collapse whitespace; preserve case, diacritics, "
            "numbers, punctuation."
        ),
        "start_time": started.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "elapsed_time_sec": elapsed,
        "manual_review": old.get(
            "manual_review",
            {
                "sample_count": 0,
                "status": "pending",
                "summary": "Human review pending.",
            },
        ),
        "known_limitations": [
            "Coverage is not OCR accuracy.",
            "Florence-2 has no calibrated OCR confidence.",
        ],
    }
