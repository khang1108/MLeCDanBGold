"""Transformers adapter for grounded VQA over supplied evidence frames."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from PIL import Image

from hcmai.common.schemas import VQAInferenceEvidence
from hcmai.thundercompute.config import HostedVQAConfig

BackendLoader = Callable[[HostedVQAConfig], tuple[Any, Any]]


class GroundedVQAModel:
    """Load one multimodal model for bounded, frame-grounded VQA calls."""

    def __init__(
        self,
        config: HostedVQAConfig,
        backend_loader: BackendLoader | None = None,
    ) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self.revision: str | None = config.revision
        self._backend_loader = backend_loader or _load_backend

    def load(self) -> None:
        if self.model is not None or self.config.checkpoint is None:
            return
        self.processor, self.model = self._backend_loader(self.config)
        self.revision = self.config.revision or getattr(
            self.model.config, "_commit_hash", None
        )

    def answer_vqa(
        self,
        question: str,
        image: Image.Image,
        evidence: VQAInferenceEvidence,
        *,
        scene_context: str = "",
    ) -> str:
        """Answer one frame-grounded question with the shared vision model."""
        context = evidence.model_dump(exclude_none=True)
        prompt = (
            f"Scene context: {scene_context or 'Not supplied'}\n"
            f"Question: {question}\n"
            f"Evidence: {json.dumps(context, ensure_ascii=False)}\n"
            "Answer from the image and evidence. Return only the final answer "
            "in the question's language, with no reasoning, at most 100 characters."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        return _short_answer(self.generate(messages))

    def answer_vqa_multi(
        self,
        question: str,
        images: list[Image.Image],
        frame_ids: list[str],
        evidence: VQAInferenceEvidence,
        *,
        scene_context: str = "",
    ) -> dict[str, Any]:
        """Answer from ordered frames and bind the answer to a supplied ID."""
        if not images or len(images) != len(frame_ids):
            raise ValueError("images and frame_ids must be non-empty and aligned")
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("frame_ids must be unique")
        context = evidence.model_dump(exclude_none=True)
        content: list[dict[str, Any]] = []
        timestamps = {
            item.frame_id: item.start_ms
            for item in evidence.items
        }
        for frame_id, image in zip(frame_ids, images):
            timestamp = timestamps.get(frame_id)
            label = f"Frame ID: {frame_id}"
            if timestamp is not None:
                label += f" | timestamp_ms: {timestamp}"
            content.extend((
                {"type": "text", "text": label},
                {"type": "image", "image": image},
            ))
        content.append({
            "type": "text",
            "text": (
                f"Scene context: {scene_context or 'Not supplied'}\n"
                f"Question: {question}\n"
                f"Evidence: {json.dumps(context, ensure_ascii=False)}\n"
                "Return only JSON with keys answer, selected_frame_id, "
                "answerable, confidence. selected_frame_id must be one of the "
                "supplied Frame IDs; confidence must be between 0 and 1."
            ),
        })
        payload = _grounded_answer(self.generate([{"role": "user", "content": content}]))
        if payload["selected_frame_id"] not in frame_ids:
            raise ValueError("VQA model selected a frame outside supplied evidence")
        return payload

    @property
    def supports_multi_image(self) -> bool:
        """Report capability from loaded multimodal model metadata."""
        if self.model is None:
            return False
        config = getattr(self.model, "config", None)
        model_type = str(getattr(config, "model_type", "")).lower()
        return bool(
            getattr(config, "vision_config", None) is not None
            or model_type in {"glm4v", "qwen2_vl", "qwen2_5_vl", "qwen3_vl"}
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate bounded text for one trusted structured prompt."""
        self.load()
        if self.model is None or self.processor is None:
            raise RuntimeError("VQA checkpoint is not configured")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(getattr(self.model, "device", self.config.device))
        generation = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation.update({"temperature": temperature, "top_p": top_p})
        output = self.model.generate(
            **inputs,
            **generation,
        )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generated, skip_special_tokens=True)

def _load_backend(config: HostedVQAConfig) -> tuple[Any, Any]:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Glm4vForConditionalGeneration,
    )

    checkpoint = config.checkpoint
    if checkpoint is None:
        raise RuntimeError("VQA checkpoint is not configured")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    metadata = AutoConfig.from_pretrained(
        checkpoint,
        revision=config.revision,
        trust_remote_code=True,
    )
    options = _model_options(config, dtype)
    model: Any
    if metadata.model_type == "glm4v":
        processor = AutoProcessor.from_pretrained(
            checkpoint, revision=config.revision, use_fast=True
        )
        model = Glm4vForConditionalGeneration.from_pretrained(
            checkpoint, **options
        )
    elif metadata.model_type in {"qwen2_5_vl", "qwen2_vl", "qwen3_vl"}:
        processor = AutoProcessor.from_pretrained(
            checkpoint, revision=config.revision, use_fast=True
        )
        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                Qwen2VLForConditionalGeneration,
            )
            model_cls = (
                Qwen2_5_VLForConditionalGeneration
                if metadata.model_type == "qwen2_5_vl"
                else Qwen2VLForConditionalGeneration
            )
            model = model_cls.from_pretrained(checkpoint, **options)
        except (ImportError, AttributeError):
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(checkpoint, **options)
    elif getattr(metadata, "vision_config", None) is not None:
        processor = AutoProcessor.from_pretrained(
            checkpoint, revision=config.revision, use_fast=True
        )
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(checkpoint, **options)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(checkpoint, **options)
    else:
        processor = AutoTokenizer.from_pretrained(
            checkpoint, revision=config.revision
        )
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **options)
    if not config.device.startswith("cuda"):
        model = model.to(config.device)
    return processor, model.eval()


def _model_options(config: HostedVQAConfig, dtype: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "revision": config.revision,
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if config.device.startswith("cuda"):
        options["device_map"] = "auto"
    return options


def _short_answer(text: str) -> str:
    value = text.rsplit("</think>", 1)[-1].strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    value = " ".join(value.split())
    if not value:
        raise ValueError("VQA model returned an empty answer")
    return value[:100].rstrip()


def _grounded_answer(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        answer = " ".join(str(value.get("answer", "")).split())[:100]
        frame_id = str(value.get("selected_frame_id", "")).strip()
        answerable = value.get("answerable", True)
        confidence = value.get("confidence", 0.5)
        if not answer or not frame_id or not isinstance(answerable, bool):
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        if 0 <= confidence <= 1:
            return {
                "answer": answer,
                "selected_frame_id": frame_id,
                "answerable": answerable,
                "confidence": confidence,
            }
    raise ValueError("VQA model did not return a grounded JSON answer")
