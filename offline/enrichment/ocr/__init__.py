"""Offline OCR enrichment package."""

from .adapter import FlorenceAdapter, RemoteOCRAdapter
from .generator import generate_ocr
from .models import (
    OCRAdapter,
    OCRConfig,
    OCREvidence,
    OCRRegion,
    OCRRegionResult,
    OCRResult,
    json_safe_ocr_raw,
    usable_completed_text,
)

__all__ = [
    "FlorenceAdapter",
    "OCRAdapter",
    "OCRConfig",
    "OCREvidence",
    "OCRRegion",
    "OCRRegionResult",
    "OCRResult",
    "RemoteOCRAdapter",
    "generate_ocr",
    "json_safe_ocr_raw",
    "usable_completed_text",
]
