"""vLLM / OpenAI Vision API adapter for frame caption enrichment.

This module formats keyframes into OpenAI-compatible multimodal chat completion
requests for standalone vLLM serving, avoiding in-process PyTorch model loading.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import io
import os
from typing import Any

from PIL import Image
import requests

from offline.enrichment.caption.adapters.qwen_vl import _QWEN_VL_PROMPT, _clean_caption
from offline.enrichment.caption.models.contracts import CaptionModelConfig


def _encode_image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Encode a PIL image to a base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format=format, quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class VLLMCaptionAdapter:
    """Adapt standalone vLLM OpenAI Vision serving to the enrichment batch contract."""

    def __init__(
        self,
        config: Any,
        base_url: str | None = None,
        batch_fn: Callable[[Sequence[Any]], Sequence[str]] | None = None,
        max_workers: int = 8,
        timeout: float = 30.0,
    ) -> None:
        self.config = config
        self.base_url = (
            base_url
            or os.getenv("HCMAI_VLM_CAPTION_URL")
            or "http://localhost:8001/v1"
        ).rstrip("/")
        self.batch_fn = batch_fn
        self.max_workers = max_workers
        self.timeout = timeout
        self.resolved_revision: str | None = getattr(config, "revision", None)

    def resolve_revision(self) -> str:
        """Resolve or query model revision from vLLM endpoint or config."""
        if self.resolved_revision:
            return self.resolved_revision

        if self.batch_fn is not None:
            self.resolved_revision = "mock-vllm-revision"
            return self.resolved_revision

        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    self.resolved_revision = str(models[0].get("id", "vllm-serving"))
                    return self.resolved_revision
        except Exception:
            pass

        self.resolved_revision = "vllm-serving"
        return self.resolved_revision

    def _caption_single_image(self, image: Image.Image) -> str:
        """Call vLLM chat completions with a single base64-encoded image."""
        b64 = _encode_image_to_base64(image)
        model = getattr(self.config, "model_checkpoint", "Qwen/Qwen2.5-VL-7B-Instruct")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": _QWEN_VL_PROMPT},
                    ],
                }
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM caption request failed ({resp.status_code}): {resp.text}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"vLLM caption response has no choices: {data}")

        raw_text = choices[0].get("message", {}).get("content", "")
        return _clean_caption(raw_text)

    def caption_batch(self, images: Sequence[Any]) -> list[str]:
        """Generate captions for a batch of images concurrently."""
        if not images:
            return []

        if self.batch_fn is not None:
            return list(self.batch_fn(images))

        # Run concurrent calls against vLLM server to utilize continuous batching
        workers = min(len(images), self.max_workers)
        if workers <= 1:
            return [self._caption_single_image(img) for img in images]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._caption_single_image, images))
