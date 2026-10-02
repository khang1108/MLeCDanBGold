"""OCR inference adapters for local Florence-2 and remote gateway."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any, Protocol

from PIL import Image

from llm.contracts import InferenceReadiness, OCRResponse
from offline.enrichment.ocr.models import OCRConfig, OCRRegionResult, OCRResult


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
        if any(not math.isfinite(value) for value in coordinates):
            raise ValueError("Florence OCR quadrilateral coordinates must be finite")
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
            from transformers.utils import logging as hf_logging

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
            hf_logging.set_verbosity_error()
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


class OCRClient(Protocol):
    def readiness(self) -> InferenceReadiness: ...
    def ocr(self, images: Sequence[Image.Image]) -> OCRResponse: ...


class RemoteOCRAdapter:
    """Invoke remote worker (via InferenceClientPool) to extract text from images.

    Verifies checkpoint and revision matching.
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


__all__ = ["FlorenceAdapter", "OCRClient", "RemoteOCRAdapter"]
