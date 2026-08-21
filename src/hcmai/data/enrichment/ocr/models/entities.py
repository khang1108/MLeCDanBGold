"""Định nghĩa các Thực thể (Entities) cho dữ liệu OCR.

Chứa các dataclass hoặc cấu trúc dữ liệu mô tả kết quả OCR (như tọa độ bounding box, văn bản nhận diện được)."""

from __future__ import annotations

from dataclasses import dataclass
import json
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
    value: object,
    *,
    max_depth: int = 8,
    max_items: int = 1_000,
    max_nodes: int | None = None,
    max_bytes: int = 65_536,
    max_string_bytes: int = 4_096,
) -> Any:
    """Return a bounded deterministic JSON-safe OCR diagnostic value.

    ``max_items`` remains as the compatibility name for the shared node budget;
    ``max_nodes`` overrides it when supplied. Unsupported objects, cycles,
    excessive nesting, and non-finite numbers become ``None``. Aggregate JSON
    bytes and every string/key are bounded without modifying the source result.
    """

    node_limit = max_items if max_nodes is None else max_nodes
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if node_limit < 1:
        raise ValueError("max_nodes must be positive")
    if max_bytes < 4:
        raise ValueError("max_bytes must be at least four")
    if max_string_bytes < 0:
        raise ValueError("max_string_bytes must be non-negative")

    omitted = object()
    seen: set[int] = set()
    remaining_nodes = node_limit

    def bounded_string(text: str) -> str:
        """Truncate one string on a valid UTF-8 boundary."""

        encoded = text.encode("utf-8")
        if len(encoded) <= max_string_bytes:
            return text
        return encoded[:max_string_bytes].decode("utf-8", errors="ignore")

    def sanitize(item: object, depth: int) -> Any | object:
        nonlocal remaining_nodes
        if remaining_nodes == 0:
            return omitted
        remaining_nodes -= 1
        if depth > max_depth:
            return None
        if isinstance(item, np.ndarray):
            item = item.tolist()
        elif isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, str):
            return bounded_string(item)
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, dict):
            identity = id(item)
            if identity in seen:
                return None
            seen.add(identity)
            result: dict[str, Any] = {}
            for key, nested in item.items():
                if isinstance(key, str):
                    normalized_key = bounded_string(key)
                elif isinstance(key, (bool, int)):
                    normalized_key = bounded_string(str(key))
                elif isinstance(key, float) and math.isfinite(key):
                    normalized_key = bounded_string(str(key))
                else:
                    continue
                sanitized = sanitize(nested, depth + 1)
                if sanitized is omitted:
                    break
                if normalized_key not in result:
                    result[normalized_key] = sanitized
            seen.remove(identity)
            return result
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                return None
            seen.add(identity)
            list_result: list[Any] = []
            for nested in item:
                sanitized = sanitize(nested, depth + 1)
                if sanitized is omitted:
                    break
                list_result.append(sanitized)
            seen.remove(identity)
            return list_result
        return None

    sanitized = sanitize(value, 0)
    if sanitized is omitted:
        sanitized = None

    def encoded_size(item: Any) -> int:
        return len(
            json.dumps(
                item,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    def fit_bytes(item: Any) -> Any:
        """Deterministically retain the longest prefix fitting the byte cap."""

        if encoded_size(item) <= max_bytes:
            return item
        if isinstance(item, str):
            encoded = item.encode("utf-8")
            lower, upper = 0, len(encoded)
            while lower < upper:
                middle = (lower + upper + 1) // 2
                candidate = encoded[:middle].decode("utf-8", errors="ignore")
                if encoded_size(candidate) <= max_bytes:
                    lower = middle
                else:
                    upper = middle - 1
            return encoded[:lower].decode("utf-8", errors="ignore")
        if isinstance(item, list):
            list_result: list[Any] = []
            for nested in item:
                candidate = [*list_result, nested]
                if encoded_size(candidate) > max_bytes:
                    break
                list_result.append(nested)
            return list_result
        if isinstance(item, dict):
            dict_result: dict[str, Any] = {}
            for key, nested in item.items():
                candidate = {**dict_result, key: nested}
                if encoded_size(candidate) > max_bytes:
                    break
                dict_result[key] = nested
            return dict_result
        return None

    return fit_bytes(sanitized)
