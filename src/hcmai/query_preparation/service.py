"""Validate, retry, and cache stateless query-preparation operations.

This service owns event-shape and exact-token invariants. It does not own the
Thundercompute gateway lifecycle or load model weights.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from hcmai.common.config import QueryPreparationConfig
from hcmai.query_preparation.cache import QueryPreparationCache, cache_key
from hcmai.query_preparation.models import (
    QueryCandidate,
    QueryCandidateSet,
    QueryPreparationAdapter,
)

_REQUIRED_TOKEN = re.compile(r"(?<!\w)[A-Z][A-Z0-9_-]*(?!\w)")



class QueryPreparationError(RuntimeError):
    """Explicit failure to produce a complete validated query result."""


class QueryPreparationService:
    """Coordinate structured inference with validation, retry, and TTL cache."""

    def __init__(
        self,
        adapter: QueryPreparationAdapter,
        config: QueryPreparationConfig,
        cache: QueryPreparationCache | None = None,
    ) -> None:
        """Initialize a stateless service over a non-owned inference adapter."""

        self._adapter = adapter
        self._config = config
        self._cache = (
            cache
            if cache is not None
            else QueryPreparationCache(
                max_entries=config.cache_max_entries,
                ttl_seconds=config.cache_ttl_seconds,
        ))

    def translate_literal(self, events_vi: Sequence[str]) -> tuple[str, ...]:
        """Translate ordered events once and cache only a validated result."""

        events = _normalize_events(events_vi)
        key = self._key("translate", events)
        cached = self._cache.get(key) if self._config.cache_enabled else None
        if cached is not None:
            return cached

        translated = self._adapter.translate(events)
        result = _validate_bundle(events, translated, name="translation")
        if self._config.cache_enabled:
            self._cache.put(key, result)
        return result

    def generate_candidates(self, events_vi: Sequence[str]) -> QueryCandidateSet:
        """Generate exactly five aligned bundles with one malformed-output retry."""

        events = _normalize_events(events_vi)
        key = self._key("candidates", events)
        cached = self._cache.get(key) if self._config.cache_enabled else None
        if cached is not None:
            return cached

        last_error: QueryPreparationError | None = None
        for _ in range(2):
            try:
                literal_en, candidates = self._adapter.generate_candidates(
                    events, self._config.candidate_count
                )
                result = _build_candidate_set(events, literal_en, candidates)
            except QueryPreparationError as error:
                last_error = error
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
    """Collapse whitespace and reject empty or non-string events."""

    if not events:
        raise QueryPreparationError("events must not be empty")
    if any(not isinstance(event, str) for event in events):
        raise QueryPreparationError("events must contain strings")
    normalized = tuple(" ".join(event.split()) for event in events)
    if any(not event for event in normalized):
        raise QueryPreparationError("events must contain non-empty strings")
    return normalized


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