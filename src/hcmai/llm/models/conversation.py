"""Generic Transformers structured-output model for KISC resolution."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from PIL import Image

from hcmai.common.schemas import VQAEvidence
from hcmai.llm.config import HostedConversationConfig

BackendLoader = Callable[[HostedConversationConfig], tuple[Any, Any]]
_STATE_FIELDS = {
    "standalone_query",
    "positive_constraints",
    "negative_constraints",
    "uncertain_constraints",
    "accepted_frame_ids",
    "rejected_frame_ids",
}


class StructuredConversationModel:
    """Load one causal LLM and return one JSON object per bounded call."""

    def __init__(
        self,
        config: HostedConversationConfig,
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

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        text = self._generate(self._messages(request))
        return _conversation_state(text)

    def answer_vqa(
        self,
        question: str,
        image: Image.Image,
        evidence: VQAEvidence,
    ) -> str:
        """Answer one frame-grounded question with the shared vision model."""
        context = evidence.model_dump(exclude_none=True)
        prompt = (
            f"Question: {question}\n"
            f"Retrieved evidence: {json.dumps(context, ensure_ascii=False)}\n"
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
        return _short_answer(self._generate(messages))

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        self.load()
        if self.model is None or self.processor is None:
            raise RuntimeError("conversation checkpoint is not configured")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(getattr(self.model, "device", self.config.device))
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generated, skip_special_tokens=True)

    def _messages(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        context = {
            key: request.get(key)
            for key in (
                "history",
                "current_message",
                "feedback",
                "previous_state",
                "response_schema",
            )
        }
        user_text = (
            json.dumps(context, ensure_ascii=False)
            + "\nReturn only the complete JSON object. Do not use Markdown."
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": request["instruction"]}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
        ]
        return messages


def _load_backend(config: HostedConversationConfig) -> tuple[Any, Any]:
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
        raise RuntimeError("conversation checkpoint is not configured")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[config.dtype]
    metadata = AutoConfig.from_pretrained(
        checkpoint,
        revision=config.revision,
        trust_remote_code=False,
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
    else:
        processor = AutoTokenizer.from_pretrained(
            checkpoint, revision=config.revision
        )
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **options)
    if not config.device.startswith("cuda"):
        model = model.to(config.device)
    return processor, model.eval()


def _model_options(config: HostedConversationConfig, dtype: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "revision": config.revision,
        "torch_dtype": dtype,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if config.device.startswith("cuda"):
        options["device_map"] = "auto"
    return options


def _conversation_state(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and _STATE_FIELDS <= value.keys():
            return value
    raise ValueError("conversation model did not return a complete state object")


def _short_answer(text: str) -> str:
    value = text.rsplit("</think>", 1)[-1].strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    value = " ".join(value.split())
    if not value:
        raise ValueError("VQA model returned an empty answer")
    return value[:100].rstrip()
