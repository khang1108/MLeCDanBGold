"""Định nghĩa các Thực thể (Entities) cho dữ liệu OCR.

Chứa các dataclass hoặc cấu trúc dữ liệu mô tả kết quả OCR (như tọa độ bounding box, văn bản nhận diện được)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FrameRow = dict[str, Any]
Evidence = dict[str, Any]
FailureDetail = dict[str, str]


@dataclass(frozen=True)
class OCRResult:
    """One ordered backend OCR response."""

    text: str
    raw_output: object | None = None
    confidence: float | None = None
