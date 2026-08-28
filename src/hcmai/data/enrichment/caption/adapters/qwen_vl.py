"""Local Qwen VL adapter for frame caption enrichment.

This module owns Qwen VL prompt construction, multimodal preprocessing, and
deterministic text generation. The prompt is deliberately kept inside the
adapter: caption artifacts expose the public ``qwen vl`` configuration label,
not the instruction text used to steer the model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from hcmai.data.enrichment.caption.models.contracts import CaptionModelConfig

_MIN_PIXELS = 256 * 28 * 28
_MAX_PIXELS = 512 * 28 * 28

# Keep the instruction private to this adapter. ``CaptionConfig.prompt`` is a
# stable public backend label used for lineage/resume checks, not this text.
_QWEN_VL_PROMPT = (
    "Describe the visible contents of this video frame for visual retrieval.\n\n"
    "Write exactly one concise, factual sentence describing the most important "
    "visual information. Prioritize the main scene or setting; the main people "
    "or subjects and their visible actions; important objects, products, or "
    "visual elements; and short, clearly readable on-screen text when relevant.\n\n"
    "Use only information directly visible in the frame. Do not infer identities, "
    "locations, time, intentions, causes, emotions, occupations, roles, or events "
    "outside the frame. Do not guess unclear details.\n\n"
    "Use specific visual nouns and verbs rather than vague descriptions. Mention "
    "only details useful for distinguishing or retrieving this frame. MANDATORY "
    "text rule: do not transcribe on-screen text. You may mention one short, "
    "clearly readable channel or logo label of at most three words when useful, "
    "but never repeat words from a headline, subtitle, banner, slide, ticker, "
    "product label, or sentence-length overlay, even when the text is clear. "
    "For those overlays, use a generic description such as ‘a news headline’ or "
    "‘an educational slide’, or omit the text. If text is partially visible, "
    "blurry, or uncertain, omit it rather than guessing.\n\n"
    "Output only the caption, with no labels, explanations, bullet points, or "
    "additional text. Silently check that the result is one grammatical sentence "
    "with a finished ending. Keep it concise and normally below roughly 100 words; "
    "this is a soft ceiling, not a reason to cut off a sentence."
)


def _dtype_for(torch: Any, name: str) -> Any:
    """Resolve the configured torch dtype without importing torch at module load."""

    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name.lower(), torch.float32)


def _clean_caption(value: Any) -> str:
    """Remove tokenizer artifacts while preserving the model's sentence."""

    text = str(value).replace("<pad>", " ").strip()
    # ``skip_special_tokens`` handles normal Qwen special tokens. This fallback
    # also protects the artifact contract when a custom processor leaves one.
    text = re.sub(r"<\|[^|]+\|>", " ", text)
    return " ".join(text.split())


class QwenVLCaptionAdapter:
    """Lazily load Qwen3-VL and return one caption per input image.

    The adapter accepts an optional ``batch_fn`` so enrichment tests can run
    without model weights. Real inference uses the official Qwen chat template
    and trims the prompt tokens before decoding; this prevents the instruction
    or padding from appearing in stored captions.
    """

    def __init__(
        self,
        config: CaptionModelConfig,
        model: Any = None,
        processor: Any = None,
        batch_fn: Callable[[Sequence[Any]], Sequence[Any]] | None = None,
    ) -> None:
        self.config = config
        self.model: Any = model
        self.processor: Any = processor
        self.batch_fn = batch_fn
        self.resolved_revision: str | None = None
        self._dtype: Any = None

    def _load(self) -> None:
        """Load the processor and Qwen model exactly once."""

        if self.model is None or self.processor is None:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            from transformers.utils import logging as hf_logging

            revision = {"revision": self.config.revision} if self.config.revision else {}
            self._dtype = _dtype_for(torch, self.config.dtype)
            self.processor = self.processor or AutoProcessor.from_pretrained(
                self.config.model_checkpoint,
                min_pixels=_MIN_PIXELS,
                max_pixels=_MAX_PIXELS,
                **revision,
            )
            loaded_model: Any = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.model_checkpoint,
                dtype=self._dtype,
                **revision,
            )
            self.model = self.model or loaded_model.to(self.config.device)
            self.model.eval()
            hf_logging.set_verbosity_error()
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
            raise ValueError(
                "Cannot create resumable captions without a resolved model revision"
            )
        return self.resolved_revision

    @staticmethod
    def _messages(images: Sequence[Any]) -> list[list[dict[str, Any]]]:
        """Build one Qwen user conversation for each PIL image."""

        return [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": _QWEN_VL_PROMPT},
                    ],
                }
            ]
            for image in images
        ]

    def caption_batch(self, images: Sequence[Any]) -> list[Any]:
        """Generate captions in input order, returning per-image failures."""

        if self.batch_fn is not None:
            return list(self.batch_fn(images))
        if not images:
            return []
        try:
            self._load()
        except Exception as error:
            self.batch_fn = lambda items, failure=error: [failure] * len(items)
            raise

        import torch

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            # Decoder-only generation is safest with left padding when a caller
            # supplies more than one image in a batch.
            tokenizer.padding_side = "left"
        inputs = self.processor.apply_chat_template(
            self._messages(images),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
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

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **self.config.decoding)
        input_length = inputs["input_ids"].shape[-1]
        decoded = self.processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [_clean_caption(value) for value in decoded]
