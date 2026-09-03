"""Immutable query-preparation models and the inference adapter boundary.

This module owns ordered event bundles. It does not perform model inference,
cache results, or expose HTTP contracts.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class QueryCandidate:
    """One numbered retrieval paraphrase with event order preserved."""

    index: int
    events: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryCandidateSet:
    """Literal translation and controlled candidates for original events."""

    original_events: tuple[str, ...]
    literal_en: tuple[str, ...]
    candidates: tuple[QueryCandidate, ...]


class QueryPreparationAdapter(Protocol):
    """Structured inference boundary implemented by Thundercompute clients."""

    @staticmethod
    def translate(events_vi: Sequence[str]) -> tuple[str, ...]:
        """Translate ordered Vietnamese events into literal English."""
        ...

    @staticmethod
    def generate_candidates(
        events_vi: Sequence[str], candidate_count: int
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        """Generate a literal translation and aligned candidate bundles."""
        ...