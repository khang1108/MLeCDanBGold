"""OCR data models, schemas, and configurations.

This module consolidates OCR configuration, domain entities, raw/region result models,
and artifact contracts for offline OCR enrichment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral
from typing import Any, Protocol, Sequence
import unicodedata

import numpy as np
from PIL import Image
from pydantic import Field, model_validator
from typing_extensions import Self

from offline.contracts import ContractModel, NonEmptyString
from offline.enrichment.models import ProcessingStatus

FrameRow = dict[str, Any]
Evidence = dict[str, Any]
FailureDetail = dict[str, str]


@dataclass(frozen=True)
class OCRConfig:
    """Settings identifying one reproducible OCR enrichment."""

    enabled: bool = True
    backend: str = "florence2"
    checkpoint: str | None = "florence-community/Florence-2-base-ft"
    revision: str | None = "0b03b6f15a4a211370fb204aee4e7dd48887ea37"
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = 32
    image_size: int | None = 768
    enrichment_version: str = "florence2_ocr_v1"
    dataset_version: str = "unknown"
    min_region_confidence: float = 0.0
    min_context_quality: float = 0.5
    artifact_version: str = "ocr-v1"

    def __post_init__(self) -> None:
        if self.batch_size < 1 or (self.image_size is not None and self.image_size < 1):
            raise ValueError("batch_size and image_size must be positive")
        if not 0.0 <= self.min_region_confidence <= 1.0:
            raise ValueError("min_region_confidence must be in [0, 1]")
        if not 0.0 <= self.min_context_quality <= 1.0:
            raise ValueError("min_context_quality must be in [0, 1]")

    @property
    def model_name(self) -> str:
        """Return the canonical backend identity."""
        return self.checkpoint or self.backend


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


class OCRAdapter(Protocol):
    """Recognize an ordered image batch without exposing model internals."""

    resolved_revision: str | None

    def recognize_batch(
        self, images: Sequence[Image.Image]
    ) -> Sequence[OCRResult]: ...


class OCRRegion(ContractModel):
    """One raw OCR region aligned to its canonical parent frame."""

    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    region_id: NonEmptyString
    region_order: int = Field(ge=0)
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_box(self) -> Self:
        """Reject inverted normalized region boxes."""
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("OCR region maximum coordinates must not precede minimums")
        return self


class OCREvidence(ContractModel):
    """Frame-level OCR evidence retaining raw and normalized text."""

    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None
    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    raw_text: str | None = None
    normalized_text: str | None = None
    quality_score: float = Field(default=0.0, ge=0, le=1)
    region_count: int = Field(default=0, ge=0)
    frame_store_id: NonEmptyString | None = None
    artifact_version: NonEmptyString
    model_name: NonEmptyString
    model_revision: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_failure_details(self) -> Self:
        """Require diagnostics when OCR generation failed."""
        if self.status is ProcessingStatus.FAILED and (
            self.error_code is None or self.error_message is None
        ):
            raise ValueError("failed evidence requires error_code and error_message")
        return self


def usable_completed_text(row: OCREvidence) -> str | None:
    """Return usable completed normalized OCR text."""
    if row.status is not ProcessingStatus.COMPLETED:
        return None
    value = row.normalized_text
    return value if value is not None and value.strip() else None


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
    min_confidence: float = 0.0,
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


def json_safe_ocr_raw(
    value: object,
    *,
    max_depth: int = 8,
    max_items: int = 1_000,
    max_nodes: int | None = None,
    max_bytes: int = 65_536,
    max_string_bytes: int = 4_096,
) -> Any:
    """Return a bounded deterministic JSON-safe OCR diagnostic value."""
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


__all__ = [
    "Evidence",
    "FailureDetail",
    "FrameRow",
    "NormalizedRegions",
    "OCRAdapter",
    "OCRConfig",
    "OCREvidence",
    "OCRRegion",
    "OCRRegionResult",
    "OCRResult",
    "json_safe_ocr_raw",
    "normalize_regions",
    "usable_completed_text",
]
