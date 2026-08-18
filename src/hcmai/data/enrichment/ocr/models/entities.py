"""Định nghĩa các Thực thể (Entities) cho dữ liệu OCR.

Chứa các dataclass hoặc cấu trúc dữ liệu mô tả kết quả OCR (như tọa độ bounding box, văn bản nhận diện được)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

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


def json_safe_ocr_raw(
    value: object, *, max_depth: int = 8, max_items: int = 1_000
) -> Any:
    """Return a bounded deterministic JSON-safe OCR diagnostic value.

    Unsupported objects, cycles, excessive nesting, and non-finite numbers are
    represented as ``None``. The source ``OCRResult`` remains unchanged.
    """

    seen: set[int] = set()

    def sanitize(item: object, depth: int) -> Any:
        if depth > max_depth:
            return None
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, np.ndarray):
            return sanitize(item.tolist(), depth + 1)
        if isinstance(item, np.generic):
            return sanitize(item.item(), depth + 1)
        if isinstance(item, dict):
            identity = id(item)
            if identity in seen:
                return None
            seen.add(identity)
            result: dict[str, Any] = {}
            for key, nested in list(item.items())[:max_items]:
                if isinstance(key, str):
                    normalized_key = key
                elif isinstance(key, (bool, int)):
                    normalized_key = str(key)
                elif isinstance(key, float) and math.isfinite(key):
                    normalized_key = str(key)
                else:
                    continue
                result[normalized_key] = sanitize(nested, depth + 1)
            seen.remove(identity)
            return result
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                return None
            seen.add(identity)
            list_result = [
                sanitize(nested, depth + 1) for nested in item[:max_items]
            ]
            seen.remove(identity)
            return list_result
        return None

    return sanitize(value, 0)
