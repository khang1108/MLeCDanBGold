"""Định nghĩa các Thực thể (Entities) cho dữ liệu OCR.

Chứa các dataclass hoặc cấu trúc dữ liệu mô tả kết quả OCR (như tọa độ bounding box, văn bản nhận diện được)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FrameRow = dict[str, Any]
Evidence = dict[str, Any]
FailureDetail = dict[str, str]


@dataclass(frozen=True)
class OCRRegionResult:
    """One immutable OCR region in backend-provided reading order."""

    text: str
    confidence: float | None
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class OCRResult:
    """One immutable backend OCR response with lossless regions."""

    text: str
    regions: tuple[OCRRegionResult, ...] = ()
    raw_output: object | None = None
