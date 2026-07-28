"""Generic Transformers structured-output model for KISC resolution."""

from __future__ import annotations

import json
from typing import Any

from hcmai.llm.config import HostedConversationConfig


class StructuredConversationModel:
    """Load one causal LLM and return one JSON object per bounded call."""

    def __init__(self, config: HostedConversationConfig) -> None:
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        self.revision: str | None = config.revision

    def load(self) -> None:
        if self.model is not None or self.config.checkpoint is None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[
            self.config.dtype
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.checkpoint, revision=self.config.revision
        )
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.config.checkpoint,
            revision=self.config.revision,
            torch_dtype=dtype,
            trust_remote_code=False,
        )
        self.model = model.to(self.config.device).eval()
        self.revision = self.config.revision or getattr(
            self.model.config, "_commit_hash", None
        )

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.load()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("conversation checkpoint is not configured")
        prompt = self._prompt(request)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return _json_object(self.tokenizer.decode(generated, skip_special_tokens=True))

    def _prompt(self, request: dict[str, Any]) -> str:
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
                + "\nReturn only the JSON object.",
            },
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "\n".join(item["content"] for item in messages)


def _json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("conversation model did not return a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("conversation model output must be an object")
    return value
