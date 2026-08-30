"""Remote OCR adapter with pinned provenance checks."""

from __future__ import annotations

from typing import Protocol, Sequence

from PIL import Image

from hcmai.common.schemas import InferenceReadiness, OCRResponse
from offline.enrichment.ocr.config import OCRConfig
from offline.enrichment.ocr.models.entities import OCRRegionResult, OCRResult


class OCRClient(Protocol):
    def readiness(self) -> InferenceReadiness: ...

    def ocr(self, images: Sequence[Image.Image]) -> OCRResponse: ...


class RemoteOCRAdapter:
    """Gọi remote worker (thông qua InferenceClientPool) để trích xuất văn bản từ hình ảnh.
    Adapter này đảm bảo kiểm tra cấu hình model (checkpoint, revision) khớp với thiết lập cục bộ.
    """

    def __init__(self, client: OCRClient, config: OCRConfig) -> None:
        self.client = client
        self.config = config
        self.resolved_revision: str | None = None

    def resolve_revision(self) -> str | None:
        status = self.client.readiness().models.get("ocr")
        if status is None or not status.loaded:
            raise RuntimeError("remote OCR model is not ready")
        if status.checkpoint != self.config.checkpoint:
            raise ValueError("remote OCR checkpoint mismatch")
        if self.config.revision is not None and status.revision != self.config.revision:
            raise ValueError("remote OCR revision mismatch")
        self.resolved_revision = status.revision
        return status.revision

    def recognize_batch(
        self, images: Sequence[Image.Image]
    ) -> Sequence[OCRResult]:
        response = self.client.ocr(images)
        if response.model != self.config.checkpoint:
            raise ValueError("remote OCR checkpoint mismatch")
        expected = self.resolved_revision or self.config.revision
        if expected is not None and response.revision != expected:
            raise ValueError("remote OCR revision changed")
        self.resolved_revision = response.revision
        return [
            OCRResult(
                text=item.text,
                regions=tuple(
                    OCRRegionResult(
                        text=region.text,
                        confidence=region.confidence,
                        x_min=region.x_min,
                        y_min=region.y_min,
                        x_max=region.x_max,
                        y_max=region.y_max,
                    )
                    for region in item.regions
                ),
                raw_output=item.raw_output,
            )
            for item in response.items
        ]
