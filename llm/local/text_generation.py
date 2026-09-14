"""Process-local adapter for generic causal language model text generation.

Owns tokenizer and causal language model lifecycle for structured JSON completion.
Does not parse domain schemas or contain HCMAI task logic.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.config import HostedTextGenerationConfig


class TextGenerationAdapter:
    """Lazy-loaded HuggingFace causal LM adapter for chat completion."""

    def __init__(self, config: HostedTextGenerationConfig) -> None:
        self.config = config
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load tokenizer and weights into device memory once per process."""
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.checkpoint,
            revision=self.config.revision,
        )
        dtype = getattr(torch, self.config.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.checkpoint,
            revision=self.config.revision,
            torch_dtype=dtype,
        ).to(self.config.device)
        self.model.eval()

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate text conforming to the supplied response schema."""
        self.load()
        assert self.tokenizer is not None and self.model is not None
        schema = json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
        augmented = list(messages) + [{
            "role": "system",
            "content": (
                "Return only one JSON object matching this JSON Schema exactly. "
                f"Do not use markdown fences. Schema: {schema}"
            ),
        }]
        prompt = self.tokenizer.apply_chat_template(
            augmented,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        ).to(self.model.device)
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": min(max_tokens, self.config.max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
