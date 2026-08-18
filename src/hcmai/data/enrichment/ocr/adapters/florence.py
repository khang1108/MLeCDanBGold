"""Adapter cho mô hình OCR Florence-2.

Giao tiếp trực tiếp với mô hình Florence-2 (Microsoft) cho nhiệm vụ OCR đa năng.

Các tính năng chính:
1. Tạo Task Prompt: Định dạng câu lệnh (VD: `<OCR>`) chuyên dụng cho Florence-2.
2. Xử lý Tensor: Chuyển đổi ảnh PIL sang dạng tensor và chạy mô hình (hỗ trợ fp16 tối ưu RAM).
3. Phân tích kết quả (Parsing): Tách chuỗi text trả về thành danh sách các cặp (Bounding Box, Text)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from hcmai.data.enrichment.ocr.config import OCRConfig
from hcmai.data.enrichment.ocr.models.entities import OCRRegionResult, OCRResult


def _parse_regions(
    raw: object, *, image_size: tuple[int, int]
) -> tuple[OCRRegionResult, ...]:
    """Convert ordered Florence quadrilaterals to normalized axis-aligned boxes."""

    if not isinstance(raw, dict):
        return ()
    labels = raw.get("labels", [])
    quad_boxes = raw.get("quad_boxes", [])
    if not isinstance(labels, list) or not isinstance(quad_boxes, list):
        raise ValueError("Florence OCR regions must be lists")
    if len(labels) != len(quad_boxes):
        raise ValueError("Florence OCR label/box count mismatch")

    width, height = image_size
    parsed: list[OCRRegionResult] = []
    for label, quad in zip(labels, quad_boxes):
        if not isinstance(quad, (list, tuple)) or len(quad) != 8:
            raise ValueError("Florence OCR quadrilateral must have 8 values")
        coordinates = [float(value) for value in quad]
        xs, ys = coordinates[0::2], coordinates[1::2]

        def clamp(value: float) -> float:
            return min(1.0, max(0.0, value))

        parsed.append(
            OCRRegionResult(
                text=str(label),
                confidence=None,
                x_min=clamp(min(xs) / width),
                y_min=clamp(min(ys) / height),
                x_max=clamp(max(xs) / width),
                y_max=clamp(max(ys) / height),
            )
        )
    return tuple(parsed)


class FlorenceAdapter:
    """Lazily load Florence-2 and return ordered OCR results."""

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self.resolved_revision = config.revision
        self._failure: Exception | None = None

    def _load(self) -> None:
        if self._failure is not None:
            raise RuntimeError("OCR backend initialization failed") from self._failure
        if self.model is not None and self.processor is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            dtype = {"bfloat16": torch.bfloat16}.get(
                self.config.dtype, torch.float32
            )
            options = {
                "revision": self.config.revision,
                "trust_remote_code": True,
            }
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name, **options
            )
            loaded_model: Any = AutoModelForImageTextToText.from_pretrained(
                self.config.model_name, dtype=dtype, **options
            )
            self.model = loaded_model.to(self.config.device).eval()
            self.resolved_revision = (
                getattr(self.model.config, "_commit_hash", None)
                or self.config.revision
            )
        except Exception as error:
            self._failure = error
            raise

    def recognize_batch(self, images: Sequence[Image.Image]) -> list[OCRResult]:
        """Return one OCR result per image in input order."""
        self._load()
        import torch

        inputs = self.processor(
            text=["<OCR_WITH_REGION>"] * len(images),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        dtype = {"bfloat16": torch.bfloat16}.get(
            self.config.dtype, torch.float32
        )
        inputs = {
            key: value.to(self.config.device, dtype=dtype)
            if value.is_floating_point()
            else value.to(self.config.device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=256, num_beams=3, do_sample=False
            )
        decoded = self.processor.batch_decode(
            generated, skip_special_tokens=False
        )
        results: list[OCRResult] = []
        for text, image in zip(decoded, images):
            raw = self.processor.post_process_generation(
                text, task="<OCR_WITH_REGION>", image_size=image.size
            ).get("<OCR_WITH_REGION>", {})
            parsed_regions = _parse_regions(raw, image_size=image.size)
            value = "\n".join(region.text for region in parsed_regions)
            results.append(
                OCRResult(text=value, regions=parsed_regions, raw_output=raw)
            )
        return results
