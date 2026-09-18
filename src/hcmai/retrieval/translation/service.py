"""Translate ordered retrieval events without query expansion.

This module owns the one-to-one translation contract and its validation. It
does not generate paraphrases, create retrieval plans, or expose HTTP routes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from hcmai.common.config import EventTranslationConfig
from hcmai.inference.errors import InferenceResponseError
from hcmai.inference.clients.llm import LLMClient
from hcmai.retrieval.translation.cache import EventTranslationCache, translation_cache_key
from hcmai.retrieval.translation.models import LiteralTranslation
from hcmai.retrieval.translation.prompts import translation_messages
from hcmai.temporal.events import normalize_event_texts

_REQUIRED_TOKEN = re.compile(r"(?<!\w)[A-Z][A-Z0-9_-]*(?!\w)")


class EventTranslationError(RuntimeError):
    """Signal that a translation violates the ordered event contract."""


class EventTranslator:
    """Translate non-English event sequences while preserving positional meaning."""

    def __init__(
        self,
        llm: LLMClient,
        config: EventTranslationConfig,
        cache: EventTranslationCache | None = None,
    ) -> None:
        """Initialize the translator with a structured LLM and bounded cache."""
        self._llm = llm
        self._config = config
        self._cache = cache or EventTranslationCache(
            max_entries=config.cache_max_entries,
            ttl_seconds=config.cache_ttl_seconds,
        )

    def translate(self, events: Sequence[str], language: str) -> tuple[str, ...]:
        """Translate events to literal English without changing their sequence.

        English events bypass inference. Provider failures deliberately retain
        their inference error type; only contract violations become
        ``EventTranslationError``.
        """
        normalized = _normalize_events(events)
        if language == "en":
            return normalized

        key = translation_cache_key(
            operation="translate",
            events=normalized,
            model=self._llm.model,
            prompt_version=self._config.prompt_version,
        )
        cached = self._cache.get(key) if self._config.cache_enabled else None
        if cached is not None:
            return cached

        try:
            response = self._llm.generate_structured(
                translation_messages(normalized),
                LiteralTranslation,
                temperature=0.0,
            )
        except InferenceResponseError as error:
            raise EventTranslationError(
                "structured translation response is invalid"
            ) from error
        result = _validate_translation(normalized, response.events)

        if self._config.cache_enabled:
            self._cache.put(key, result)
        return result


def _normalize_events(events: Sequence[str]) -> tuple[str, ...]:
    """Convert shared event-shape validation into the translator error boundary."""
    try:
        return normalize_event_texts(events)
    except ValueError as error:
        raise EventTranslationError(str(error)) from error


def _validate_translation(
    original: tuple[str, ...], values: Sequence[str]
) -> tuple[str, ...]:
    """Validate positional cardinality and required case-sensitive source tokens."""
    if len(values) != len(original):
        raise EventTranslationError("translation changed event count")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise EventTranslationError("translation must contain non-empty strings")

    translated = tuple(" ".join(value.split()) for value in values)
    for source, generated in zip(original, translated):
        generated_tokens = set(_REQUIRED_TOKEN.findall(generated))
        for token in _REQUIRED_TOKEN.findall(source):
            if token not in generated_tokens:
                raise EventTranslationError(
                    f"translation omitted required token {token!r}"
                )
    return translated
