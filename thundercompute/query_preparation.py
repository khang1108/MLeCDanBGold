"""Own Qwen3-4B loading and structured query-preparation generation.

The model is loaded lazily once per Thundercompute process. HCMAI runtime
workers consume only the structured HTTP boundary defined by this service.
"""

from __future__ import annotations

import json
from typing import Any

from thundercompute.config import HostedQueryPreparationConfig

_TRANSLATE_INSTRUCTION = """
You translate Vietnamese video-retrieval events into concise literal English.

Rules:
- Translate each input event independently.
- Preserve the exact event count and event order.
- Output event i must describe exactly the same event as input event i.
- Do not merge, split, drop, duplicate, or reorder events.
- Preserve all factual information: people, objects, actions, colors, numbers,
    quantities, positions, relationships, acronyms, proper names, and visible text.
- Preserve unknown placeholders such as X exactly as written.
- Do not infer, explain, enrich, or add information that is not explicitly present.
- Prefer concise, natural English suitable for image/video retrieval.
- Do not output Markdown, comments, explanations, or code fences.

Return only valid JSON with exactly this schema:

{
    "events": [
        "translated event 1",
        "translated event 2"
]}
"""
_CANDIDATE_INSTRUCTION = """
Given Vietnamese video-retrieval events, produce:
1. one concise literal English translation bundle, and
2. exactly five distinct English retrieval paraphrase bundles.

Rules for every output bundle:
- Preserve the exact input event count and event order.
- Event i must correspond exactly to input event i.
- Never merge, split, drop, duplicate, or reorder events.
- Preserve all factual information: people, objects, actions, colors, numbers,
    quantities, positions, relationships, acronyms, proper names, and visible text.
- Preserve unknown placeholders such as X exactly as written.
- Do not infer missing entities or add facts not explicitly present.
- Do not replace an unknown object with a guessed object.
- Keep each event concise and suitable for image/video retrieval.

Rules for the five candidates:
- Use meaning-preserving lexical and syntactic variations.
- The five candidates should be meaningfully different in wording.
- Do not create diversity by changing facts, specificity, entities, or actions.
- Avoid five near-duplicate sentences differing only by one word.

Do not output Markdown, explanations, comments, or code fences.

Return only valid JSON with exactly this schema:

{
    "literal_en": [
        "literal event 1",
        "literal event 2"
    ],
    "candidates": [{
        "events": [
            "candidate 1 event 1",
            "candidate 1 event 2"
        ]},
        {
        "events": [
            "candidate 2 event 1",
            "candidate 2 event 2"
        ]},
        {
        "events": [
            "candidate 3 event 1",
            "candidate 3 event 2"
        ]},
        {
        "events": [
            "candidate 4 event 1",
            "candidate 4 event 2"
        ]},
        {
        "events": [
            "candidate 5 event 1",
            "candidate 5 event 2"
]}]}
"""


class QwenQueryPreparer:
    """Generate validated query-preparation structures with Qwen3-4B."""

    def __init__(self, config: HostedQueryPreparationConfig) -> None:
        """Retain pinned model settings without allocating weights."""

        self.config = config
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def _ensure_loaded(self) -> None:
        """Load tokenizer and model at most once for this process owner."""

        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.config.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_checkpoint,
            revision=self.config.revision,
        )
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.config.model_checkpoint,
            revision=self.config.revision,
            dtype=dtype,
        )
        model = model.to(self.config.device)
        model.eval()
        self.model = model

    def translate(self, events: list[str]) -> list[str]:
        """Return one literal English event for every input event."""

        value = self._generate(_TRANSLATE_INSTRUCTION, events, do_sample=False)
        translated = value.get("events")
        return _string_list(translated, expected=len(events), name="events")

    def generate_candidates(self, events: list[str], candidate_count: int = 5) -> dict[str, Any]:
        """Return literal English and exactly five aligned candidate bundles."""

        if candidate_count != 5:
            raise ValueError("candidate_count must be exactly 5")
        value = self._generate(
            _CANDIDATE_INSTRUCTION,
            events,
            do_sample=True,
        )
        literal_en = _string_list(value.get("literal_en"), expected=len(events), name="literal_en")
        raw_candidates = value.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) != 5:
            raise ValueError("candidates must contain exactly 5 bundles")
        candidates = [
            _string_list(candidate, expected=len(events), name="candidate")
            for candidate in raw_candidates
        ]
        return {"literal_en": literal_en, "candidates": candidates}

    def _generate(self, instruction: str, events: list[str], *, do_sample: bool) -> dict[str, Any]:
        """Generate and decode one strict JSON object with thinking disabled."""

        if not events or any(not event.strip() for event in events):
            raise ValueError("events must contain non-empty strings")
        
        self._ensure_loaded()

        assert self.model is not None
        assert self.tokenizer is not None

        user_content = f"Input events:\n{json.dumps(events, ensure_ascii=False)}"
        prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        generation = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
        }

        if do_sample:
            generation["temperature"] = self.config.candidate_temperature

        output = self.model.generate(**inputs, **generation)
        generated = output[0, inputs["input_ids"].shape[1] :]

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("model output must be a JSON object")
        return value


def _string_list(value: Any, *, expected: int, name: str) -> list[str]:
    """Validate one ordered list of non-empty generated strings."""

    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"{name} changed event count")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    return [item.strip() for item in value]