"""Validate, retry, and cache stateless query-preparation operations.

This service owns event-shape and exact-token invariants. It delegates structured
text generation to an LLMClient without binding to any concrete LLM provider.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from hcmai.common.config import QueryPreparationConfig
from hcmai.inference.llm import LLMClient
from hcmai.query_preparation.cache import QueryPreparationCache, cache_key
from hcmai.query_preparation.models import (
    CandidateBundle,
    LiteralTranslation,
    QueryCandidate,
    QueryCandidateSet,
)
from hcmai.temporal.events import normalize_event_texts

_REQUIRED_TOKEN = re.compile(r"(?<!\w)[A-Z][A-Z0-9_-]*(?!\w)")

_TRANSLATE_SYSTEM_PROMPT = """You translate video-retrieval events into concise literal English.

Rules:
- Translate each event independently preserving exact event count and event order.
- Output event i must correspond directly to input event i.
- Do not merge, split, drop, duplicate, or reorder events.
- Preserve all people, objects, actions, colors, numbers, positions, and exact uppercase tokens or codes.
- Do not infer, explain, enrich, or add information not present in the input.
- Output valid JSON matching the schema with "events": list[str].
"""

_CANDIDATE_SYSTEM_PROMPT = """Given video-retrieval events, produce:
1. One concise literal English translation bundle ("literal_en").
2. Exactly 5 distinct English retrieval paraphrase bundles ("candidates").

Rules:
- Every output bundle must preserve the exact input event count and order.
- Preserve all factual entities, actions, colors, numbers, and exact uppercase tokens/codes.
- The 5 candidates must be distinct paraphrases with meaning preserved.
- Output valid JSON matching the schema.
"""


class QueryPreparationError(RuntimeError):
    """Explicit failure to produce a complete validated query result."""


class QueryPreparationService:
    """Coordinate structured inference with validation, retry, and TTL cache."""

    def __init__(
        self,
        llm: LLMClient,
        config: QueryPreparationConfig,
        cache: QueryPreparationCache | None = None,
    ) -> None:
        """Initialize the service over a provider-agnostic LLMClient."""
        self._llm = llm
        self._config = config
        self._cache = (
            cache
            if cache is not None
            else QueryPreparationCache(
                max_entries=config.cache_max_entries,
                ttl_seconds=config.cache_ttl_seconds,
            )
        )

    def translate_literal(
        self,
        events: Sequence[str],
        language: str = "vi",
    ) -> tuple[str, ...]:
        """Translate ordered events into literal English.

        If language is already English ('en'), returns the normalized events
        directly without an LLM inference call.
        """
        normalized_events = _normalize_events(events)
        if language == "en":
            return normalized_events

        key = self._key("translate", normalized_events)
        cached = self._cache.get(key) if self._config.cache_enabled else None
        if cached is not None:
            return cached

        user_content = "\n".join(
            f"Event {i+1}: {event}" for i, event in enumerate(normalized_events)
        )
        messages = [
            {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate the following events:\n{user_content}"},
        ]

        try:
            response = self._llm.generate_structured(
                messages,
                LiteralTranslation,
                temperature=0.0,
            )
            result = _validate_bundle(
                normalized_events,
                response.events,
                name="translation",
            )
        except Exception as error:
            if isinstance(error, QueryPreparationError):
                raise
            raise QueryPreparationError(f"Translation inference failed: {error}") from error

        if self._config.cache_enabled:
            self._cache.put(key, result)
        return result

    def generate_candidates(self, events: Sequence[str]) -> QueryCandidateSet:
        """Generate exactly five aligned bundles with one malformed-output retry."""
        normalized_events = _normalize_events(events)
        key = self._key("candidates", normalized_events)
        cached = self._cache.get(key) if self._config.cache_enabled else None
        if cached is not None:
            return cached

        user_content = "\n".join(
            f"Event {i+1}: {event}" for i, event in enumerate(normalized_events)
        )
        messages = [
            {"role": "system", "content": _CANDIDATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Generate literal English and 5 candidate bundles for:\n{user_content}",
            },
        ]

        last_error: QueryPreparationError | None = None
        for _ in range(2):
            try:
                response = self._llm.generate_structured(
                    messages,
                    CandidateBundle,
                    temperature=0.7,
                )
                result = _build_candidate_set(
                    normalized_events,
                    response.literal_en,
                    response.candidates,
                )
            except Exception as error:
                if isinstance(error, QueryPreparationError):
                    last_error = error
                else:
                    last_error = QueryPreparationError(f"Candidate generation failed: {error}")
                continue

            if self._config.cache_enabled:
                self._cache.put(key, result)
            return result

        assert last_error is not None
        raise last_error

    def _key(self, operation: str, events: tuple[str, ...]) -> tuple[str, ...]:
        """Build a cache key containing prompt and immutable model identity."""
        return cache_key(
            operation=operation,
            events=events,
            model_name=self._config.model_name,
            model_revision=self._config.model_revision,
            prompt_version=self._config.prompt_version,
        )


def _normalize_events(events: Sequence[str]) -> tuple[str, ...]:
    """Translate shared explicit-event validation into the service error boundary."""
    try:
        return normalize_event_texts(events)
    except ValueError as error:
        raise QueryPreparationError(str(error)) from error


def _build_candidate_set(
    original: tuple[str, ...],
    literal_en: Sequence[str],
    candidates: Sequence[Sequence[str]],
) -> QueryCandidateSet:
    """Validate all positional bundles and construct immutable candidates."""
    literal = _validate_bundle(original, literal_en, name="literal translation")
    if len(candidates) != 5:
        raise QueryPreparationError("candidate response must contain exactly 5 bundles")
    validated = tuple(
        QueryCandidate(
            index=index,
            events=_validate_bundle(original, candidate, name=f"candidate {index}"),
        )
        for index, candidate in enumerate(candidates, start=1)
    )
    return QueryCandidateSet(
        original_events=original,
        literal_en=literal,
        candidates=validated,
    )


def _validate_bundle(
    original: tuple[str, ...], values: Sequence[str], *, name: str
) -> tuple[str, ...]:
    """Validate event count, content, and exact placeholders or acronyms."""
    if len(values) != len(original):
        raise QueryPreparationError(f"{name} changed event count")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise QueryPreparationError(f"{name} must contain non-empty strings")

    normalized = tuple(" ".join(value.split()) for value in values)
    for source, generated in zip(original, normalized):
        generated_tokens = set(_REQUIRED_TOKEN.findall(generated))
        for token in _REQUIRED_TOKEN.findall(source):
            if token not in generated_tokens:
                raise QueryPreparationError(f"{name} omitted required token {token!r}")
    return normalized