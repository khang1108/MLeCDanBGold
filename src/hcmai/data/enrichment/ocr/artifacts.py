"""Quản lý Artifacts (Dữ liệu đầu ra) của OCR.

Kiểm tra, xác thực và lưu trữ kết quả nhận diện văn bản (OCR artifacts).

Các tính năng chính:
1. Định dạng lưu trữ: Ghi kết quả OCR (tọa độ hộp, nội dung chữ) ra định dạng JSON hoặc Parquet.
2. Đảm bảo toàn vẹn: Kiểm tra checksum và tính hợp lệ của file sau khi quá trình ghi hoàn tất.
3. Khôi phục dữ liệu: Cung cấp tiện ích load lại danh sách kết quả OCR để hợp nhất (fusion)."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path

import pandas as pd

from hcmai.common.schemas import FrameEnrichment, ProcessingStatus
from hcmai.common.utils.io import atomic_write, write_json, write_parquet

from .config import OCRConfig
from .models.entities import Evidence, FailureDetail, FrameRow, OCRResult


def _null_scalar(value: object) -> bool:
    return value is None or isinstance(value, float) and math.isnan(value)


def normalize_text(text: str) -> str:
    """Normalize text without discarding Vietnamese characters."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def valid_ocr(data: FrameRow, config: OCRConfig) -> FrameEnrichment | None:
    """Return a completed row that can be safely resumed."""
    try:
        values, objects = dict(data), data.get("objects")
        to_list = getattr(objects, "tolist", None)
        values["objects"] = to_list() if callable(to_list) else objects or []
        nullable = ("caption", "detailed_caption", "ocr_text", "asr_text", "error_message")
        values.update(
            {key: None for key in nullable if _null_scalar(values.get(key))}
        )
        row = FrameEnrichment.model_validate(values)
    except Exception:
        return None
    valid = (
        row.enrichment_version == config.enrichment_version
        and row.model_name == config.model_name
        and row.status == ProcessingStatus.COMPLETED
        and row.error_message is None
        and row.caption is None
        and row.detailed_caption is None
        and row.asr_text is None
    )
    return row if valid else None


def failure_row(
    frame_id: str, config: OCRConfig, stage: str, error: Exception
) -> tuple[FrameEnrichment, FailureDetail]:
    """Build one bounded failed row and its diagnostic evidence."""
    message = " ".join(str(error).split())[:300] or type(error).__name__
    row = FrameEnrichment.model_validate(
        {
            "frame_id": frame_id,
            "model_name": config.model_name,
            "enrichment_version": config.enrichment_version,
            "status": ProcessingStatus.FAILED,
            "error_message": message,
        }
    )
    return row, {
        "frame_id": frame_id,
        "enrichment_version": config.enrichment_version,
        "processing_stage": stage,
        "exception_category": type(error).__name__,
        "error_message": message,
    }


def parsed_row(
    frame_id: str, result: object, config: OCRConfig
) -> tuple[FrameEnrichment, Evidence]:
    """Validate one backend response and materialize its shared row."""
    if not isinstance(result, OCRResult) or not isinstance(result.text, str):
        raise TypeError("OCR backend returned a malformed result")
    confidence = result.confidence
    if confidence is not None and (
        not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
    ):
        raise ValueError("OCR confidence must be finite")
    row = FrameEnrichment.model_validate(
        {
            "frame_id": frame_id,
            "ocr_text": normalize_text(result.text) or None,
            "model_name": config.model_name,
            "enrichment_version": config.enrichment_version,
        }
    )
    evidence: Evidence = {
        "frame_id": frame_id,
        "raw_output": (
            str(result.raw_output)[:500]
            if result.raw_output is not None
            else None
        ),
        "confidence": float(confidence) if confidence is not None else None,
    }
    return row, evidence


def write_ocr_artifacts(
    output: Path,
    order: list[str],
    rows: dict[str, FrameEnrichment],
    failures: dict[str, FailureDetail],
) -> None:
    """Atomically write ordered OCR rows and failures."""
    table = pd.DataFrame(
        [rows[key].model_dump(mode="json") for key in order if key in rows],
        columns=list(FrameEnrichment.model_fields),
    )
    atomic_write(
        output / "frame_enrichment.parquet",
        lambda path: write_parquet(table, path, index=False),
    )
    atomic_write(
        output / "failures.json",
        lambda path: write_json(
            [failures[key] for key in order if key in failures], path
        ),
    )
