"""Native Florence-2 OCR backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from hcmai.enrichment.ocr.config import OCRConfig
from hcmai.enrichment.ocr.models.entities import OCRResult


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
                "trust_remote_code": False,
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
            text=["<OCR>"] * len(images),
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
                text, task="<OCR>", image_size=image.size
            ).get("<OCR>", "")
            value = "" if str(raw).strip().casefold() == "unanswerable" else str(raw)
            results.append(OCRResult(text=value, raw_output=raw))
        return results
