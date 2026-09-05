"""Protocols an OCR model must satisfy to join the pipeline."""

from __future__ import annotations

from typing import Protocol, Sequence

from PIL import Image

from offline.enrichment.ocr.models.entities import OCRResult


class OCRAdapter(Protocol):
    """Recognize an ordered image batch without exposing model internals."""

    resolved_revision: str | None

    def recognize_batch(
        self, images: Sequence[Image.Image]
    ) -> Sequence[OCRResult]: ...
