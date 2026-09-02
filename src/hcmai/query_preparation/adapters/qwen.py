"""Thin adapter over the structured Thundercompute query-preparation API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hcmai.query_preparation.service import QueryPreparationError


class QwenQueryPreparationAdapter:
    """Convert Thundercompute structured arrays into immutable tuples."""

    def __init__(self, gateway: Any) -> None:
        """Retain the shared inference gateway without owning its lifecycle."""

        self._gateway = gateway

    def translate(self, events_vi: Sequence[str]) -> tuple[str, ...]:
        """Translate events through the structured gateway operation."""

        value = self._gateway.translate_query_events(list(events_vi))
        return _tuple_of_strings(value, field="events")

    def generate_candidates(
        self, events_vi: Sequence[str], candidate_count: int
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        """Convert a structured provider response without parsing prose."""

        value = self._gateway.generate_query_candidates(list(events_vi), candidate_count)
        if not isinstance(value, Mapping):
            raise QueryPreparationError("provider response must be a mapping")
        if "literal_en" not in value:
            raise QueryPreparationError("provider response is missing literal_en")
        if "candidates" not in value:
            raise QueryPreparationError("provider response is missing candidates")

        literal_en = _tuple_of_strings(value["literal_en"], field="literal_en")
        raw_candidates = value["candidates"]
        if not isinstance(raw_candidates, (list, tuple)):
            raise QueryPreparationError("candidates must be an array")
        candidates = tuple(
            _tuple_of_strings(candidate, field="candidate") for candidate in raw_candidates
        )
        return literal_en, candidates


def _tuple_of_strings(value: Any, *, field: str) -> tuple[str, ...]:
    """Validate one structured string array without semantic rewriting."""

    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise QueryPreparationError(f"{field} must be an array of strings")
    return tuple(value)