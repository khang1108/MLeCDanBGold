"""Immutable query-preparation models and structured LLM response schemas.

This module owns ordered event bundles and Pydantic models for structured
LLM translation and candidate generation. It does not perform model inference,
cache results, or expose HTTP contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LiteralTranslation(BaseModel):
    """Structured literal English translation response from an LLM."""

    model_config = ConfigDict(extra="forbid")

    events: list[NonBlank]


class CandidateBundle(BaseModel):
    """Structured literal translation and aligned candidate bundles from an LLM."""

    model_config = ConfigDict(extra="forbid")

    literal_en: list[NonBlank]
    candidates: list[list[NonBlank]]


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