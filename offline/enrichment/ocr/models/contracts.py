"""Định nghĩa giao ước (Contracts) cho OCR.

Quy định interface đầu vào/đầu ra mà các mô hình OCR cần phải trả về để tương thích với pipeline chính."""

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
