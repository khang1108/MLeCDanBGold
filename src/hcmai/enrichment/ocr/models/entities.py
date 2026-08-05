"""Internal OCR result and artifact types."""

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
