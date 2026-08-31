"""OCR enrichment contracts, artifacts, and adapter result values."""

from .evidence import OCREvidence, OCRRegion, usable_completed_text
from .entities import OCRRegionResult, OCRResult, json_safe_ocr_raw

__all__ = [
    "OCREvidence",
    "OCRRegion",
    "OCRRegionResult",
    "OCRResult",
    "json_safe_ocr_raw",
    "usable_completed_text",
]
