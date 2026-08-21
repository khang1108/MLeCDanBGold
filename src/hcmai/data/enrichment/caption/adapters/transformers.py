"""Adapter cho mô hình Captioning (Transformers).

Giao tiếp với các mô hình HuggingFace (ví dụ: BLIP, LLaVA) để sinh mô tả trực quan cho ảnh.

Các tính năng chính:
1. Lazy Loading: Chỉ load weights của mô hình vào VRAM khi hàm sinh (generate) được gọi lần đầu.
2. Tiền xử lý (Preprocessing): Resize và chuẩn hoá ảnh bằng bộ Processor của thư viện Transformers.
3. Sinh Text (Inference): Chạy hàm tạo ngôn ngữ (text generation) và trả về danh sách chuỗi."""

from __future__ import annotations

from typing import Callable, Any, Sequence

from hcmai.data.enrichment.caption.models.contracts import CaptionModelConfig

class TransformersCaptionAdapter:
    """Lazy, single-instance caption model boundary."""

    def __init__(
        self,
        config: CaptionModelConfig,
        model: Any = None,
        processor: Any = None,
        batch_fn: Callable[[Sequence[Any]], Sequence[Any]] | None = None,
    ):
        self.config = config
        self.model: Any = model
        self.processor: Any = processor
        self.batch_fn = batch_fn
        self.resolved_revision: str | None = None
        self._dtype: Any = None

    def _load(self) -> None:
        if self.model is None or self.processor is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            revision = {"revision": self.config.revision} if self.config.revision else {}
            types = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            self._dtype = types.get(self.config.dtype, torch.float32)
            self.processor = self.processor or AutoProcessor.from_pretrained(
                self.config.model_checkpoint, **revision
            )
            loaded_model: Any = AutoModelForImageTextToText.from_pretrained(
                self.config.model_checkpoint, torch_dtype=self._dtype, **revision
            )
            self.model = self.model or loaded_model.to(self.config.device)
            self.model.eval()
        self.resolved_revision = (
            getattr(getattr(self.model, "config", None), "_commit_hash", None)
            or self.config.revision
        )

    def resolve_revision(self) -> str:
        """Resolve the immutable model revision before writing reusable rows."""
        if self.resolved_revision:
            return self.resolved_revision
        if self.batch_fn is not None:
            self.resolved_revision = self.config.revision
        else:
            self._load()
        if not self.resolved_revision:
            raise ValueError("Cannot create resumable captions without a resolved model revision")
        return self.resolved_revision

    def caption_batch(self, images: Sequence[Any]) -> list[Any]:
        """Return captions or per-image exceptions for one batch."""
        if self.batch_fn is not None:
            return list(self.batch_fn(images))
        try:
            self._load()
        except Exception as error:
            self.batch_fn = lambda items, failure=error: [failure] * len(items)
            raise
        inputs = self.processor(
            text=[self.config.prompt] * len(images),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        for key, value in inputs.items():
            value = value.to(self.config.device)
            inputs[key] = (
                value.to(self._dtype)
                if self._dtype is not None and value.is_floating_point()
                else value
            )
        generated = self.model.generate(**inputs, **self.config.decoding)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=False)
        return [
            self.processor.post_process_generation(
                text, task=self.config.prompt, image_size=image.size
            ).get(self.config.prompt, "")
            for text, image in zip(decoded, images)
        ]
