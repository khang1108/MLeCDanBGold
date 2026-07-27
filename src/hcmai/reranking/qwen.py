"""Native Qwen3-VL relevance scoring for ordered query-image pairs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import Any, cast

from PIL import Image

from hcmai.reranking.config import QwenRerankerConfig

_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
VisionInfo = Callable[..., tuple[Any, Any, dict[str, Any]]]

class QwenRerankerScorer:
    """Lazily score ordered query-image pairs with official yes/no logits."""

    def __init__(
        self,
        config: QwenRerankerConfig,
        model: Any | None = None,
        processor: Any | None = None,
        vision_info: VisionInfo | None = None,
    ) -> None:
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be injected together")
        self.config = config
        self.model: Any = model
        self.processor: Any = processor
        self.vision_info = vision_info
        self._base_model: Any = None
        self._weights: Any = None
        self._load_failure: QwenRerankerError | None = None
        self.resolved_revision: str | None = None

    def _setup(self) -> None:
        import torch
        if self.model is None or self.processor is None:
            raise QwenRerankerError("model and processor are not initialized")
        self.model.to(self.config.device).eval()
        vocab = self.processor.tokenizer.get_vocab()
        weight = self.model.lm_head.weight.detach()
        self._weights = torch.stack([weight[vocab["no"]], weight[vocab["yes"]]])
        self._weights = self._weights.to(self.config.device)
        self._base_model = self.model.model
        model_config = getattr(self.model, "config", None)
        self.resolved_revision = self.config.revision or getattr(
            model_config, "_commit_hash", None)

    def _ensure_loaded(self) -> None:
        if self._load_failure is not None:
            raise self._load_failure
        if self._base_model is not None:
            return
        try:
            if self.model is None:
                self.model, self.processor = _load_native(self.config)
            self._setup()
        except Exception as error:
            failure = QwenRerankerError(
                f"Qwen initialization failed: {_bounded(error)}")
            self._load_failure = failure
            raise failure from error

    def _encode(
        self, query: str, images: Sequence[Image.Image]
    ) -> Mapping[str, Any]:
        if self.vision_info is None:
            module = import_module("qwen_vl_utils")
            vision_info = cast(VisionInfo, module.process_vision_info)
        else:
            vision_info = self.vision_info
        pairs = [_messages(self.config, query, image) for image in images]
        text = self.processor.apply_chat_template(
            pairs, tokenize=False, add_generation_prompt=True
        )
        vision, videos, video_kwargs = vision_info(
            pairs, image_patch_size=16, return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadata = None
        if videos:
            videos, video_metadata = zip(*videos)
        inputs = self.processor(
            text=text,
            images=vision,
            videos=list(videos) if videos else None,
            video_metadata=list(video_metadata) if video_metadata else None,
            padding=True, truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt", do_resize=False,
            **video_kwargs,
        )
        if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
            raise QwenRerankerError("processor returned malformed inputs")
        return inputs

    def score_batch(self, query: str, images: Sequence[Image.Image]) -> list[float]:
        """Return one finite official relevance probability per image."""
        if not images:
            return []
        self._ensure_loaded()
        try:
            import torch
            inputs = {
                key: value.to(self.config.device) if hasattr(value, "to") else value
                for key, value in self._encode(query, images).items()
            }
            with torch.inference_mode():
                output = self._base_model(**inputs)
                hidden = output.last_hidden_state[:, -1].float()
                logits = hidden @ self._weights.float().T
                values = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            scores = [float(value) for value in values]
            if len(scores) != len(images) or not all(map(math.isfinite, scores)):
                raise QwenRerankerError("model returned invalid score count or values")
            return scores
        except QwenRerankerError:
            raise
        except Exception as error:
            raise QwenRerankerError(
                f"Qwen scoring failed: {_bounded(error)}") from error

    @property
    def metadata(self) -> dict[str, Any]:
        """Return bounded, read-only compatibility metadata."""
        return {
            "checkpoint": self.config.checkpoint,
            "revision": self.resolved_revision or self.config.revision,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "model_class": type(self.model).__name__ if self.model else None,
            "processor_class": type(self.processor).__name__
            if self.processor else None,
            "trust_remote_code": False,
        }

class QwenRerankerError(RuntimeError):
    """Bounded failure raised by the model-specific scoring boundary."""


def _torch_dtype(name: str) -> Any:
    import torch
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _messages(
    config: QwenRerankerConfig, query: str, image: Image.Image
) -> list[dict[str, Any]]:
    content = [
        {"type": "text", "text": f"<Instruct>: {config.instruction}"},
        {"type": "text", "text": "<Query>:"},
        {"type": "text", "text": query},
        {"type": "text", "text": "\n<Document>:"},
        {"type": "image", "image": image, "min_pixels": 4096,
            "max_pixels": config.max_pixels},
    ]
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
        {"role": "user", "content": content},
    ]


def _load_native(config: QwenRerankerConfig) -> tuple[Any, Any]:
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    snapshot = snapshot_download(config.checkpoint, revision=config.revision)
    processor = AutoProcessor.from_pretrained(
        snapshot, padding_side="left", trust_remote_code=False)
    model: Any = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=_torch_dtype(config.dtype),
        attn_implementation="eager",
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    return model.to(config.device).eval(), processor


def _bounded(error: Exception) -> str:
    return (str(error).strip() or type(error).__name__)[:240]