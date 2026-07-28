"""Generic Transformers structured-output model for KISC resolution."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

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
        self.load()
        if self.model is None or self.processor is None:
            raise RuntimeError("conversation checkpoint is not configured")
        messages = self._messages(request)
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
        text = self.processor.decode(generated, skip_special_tokens=True)
        return _conversation_state(text)

    def _messages(self, request: dict[str, Any]) -> list[dict[str, str]]:
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
        messages = [
            {"role": "system", "content": request["instruction"]},
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False)
                + "\nReturn only the complete JSON object. Do not use Markdown.",
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
