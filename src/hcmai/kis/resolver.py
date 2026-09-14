"""Domain resolver for semantic KIS intent graphs using an LLM.

This module owns transforming client-provided raw clue sequences into a validated
KISIntent graph. It delegates language reasoning to an LLMClient and enforces
structural invariants without deterministic fallbacks.
"""

from __future__ import annotations

from collections.abc import Sequence

from hcmai.inference.llm import LLMClient
from hcmai.kis.models import KISIntent
from hcmai.kis.prompts import build_kis_intent_messages


class KISIntentResolver:
    """Resolves an evolving sequence of KIS clues into a validated semantic graph."""

    def __init__(self, llm: LLMClient) -> None:
        """Initialize with a capability-level LLMClient."""
        self._llm = llm

    def resolve(self, inputs: Sequence[str]) -> KISIntent:
        """Resolve ordered clue strings into a structured KISIntent.

        Args:
            inputs: Ordered list of participant clues from revision 1 to current.

        Returns:
            A validated KISIntent graph containing resolved entities, sequential events,
            and temporal edges.

        Raises:
            ValueError: If inputs are empty, contain blank strings, or the LLM output
                fails revision/input invariance checks.
        """
        normalized = tuple(" ".join(value.split()) for value in inputs)
        if not normalized or any(not value for value in normalized):
            raise ValueError("KIS inputs must contain non-empty text")

        messages = build_kis_intent_messages(normalized)
        intent = self._llm.generate_structured(messages, KISIntent)

        if intent.revision != len(normalized) or intent.inputs != list(normalized):
            raise ValueError("resolver changed KIS revision or raw clue history")

        return intent
