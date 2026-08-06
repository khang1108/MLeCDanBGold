"""Bounded, auditable query variants for textual KIS."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from hcmai.common.schemas import QuerySuggestionRequest, QuerySuggestionResponse

_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_CAPITALIZED = re.compile(r"(?<![.!?]\s)\b[A-Z][\w'-]*\b")
_HARD_TERMS = {
    "black", "blue", "brown", "green", "grey", "gray", "orange", "pink",
    "purple", "red", "white", "yellow", "đen", "đỏ", "trắng", "xanh",
    "vàng", "hồng", "tím", "nâu", "cam", "không", "chẳng", "chưa",
    "never", "no", "not", "without",
}


class SuggestionProvider(Protocol):
    """Small boundary shared with the existing suggestion service."""

    def suggest(self, request: QuerySuggestionRequest) -> QuerySuggestionResponse: ...


@dataclass(frozen=True, slots=True)
class QueryVariant:
    """One retrieval query and its auditable influence on fusion."""

    query: str
    kind: str
    weight: float


@dataclass(frozen=True, slots=True)
class VariantPlan:
    """Variants plus non-fatal planning diagnostics."""

    variants: tuple[QueryVariant, ...]
    warnings: tuple[str, ...] = ()


class ControlledQueryExpander:
    """Retain only bounded suggestions that preserve hard query constraints."""

    def __init__(
        self,
        provider: SuggestionProvider | None = None,
        *,
        generated_count: int = 5,
        generated_weight: float = 0.35,
        max_query_chars: int = 500,
    ) -> None:
        if not 5 <= generated_count <= 10:
            raise ValueError("generated_count must be between 5 and 10")
        if not 0 < generated_weight < 1:
            raise ValueError("generated_weight must be between zero and one")
        if max_query_chars < 1:
            raise ValueError("max_query_chars must be positive")
        self.provider = provider
        self.generated_count = generated_count
        self.generated_weight = generated_weight
        self.max_query_chars = max_query_chars

    def expand(self, query: str) -> VariantPlan:
        original = QueryVariant(query=query, kind="original", weight=1.0)
        if self.provider is None:
            return VariantPlan((original,))
        try:
            response = self.provider.suggest(QuerySuggestionRequest(
                query=query,
                count=self.generated_count,
            ))
        except Exception as error:
            category = getattr(getattr(error, "category", None), "value", None)
            return VariantPlan(
                (original,),
                (f"query expansion fallback ({category or type(error).__name__})",),
            )

        protected = _protected_constraints(query)
        seen = {_normalize(query)}
        variants = [original]
        rejected = 0
        for suggestion in response.suggestions[: self.generated_count]:
            candidate = suggestion.query.strip()
            normalized = _normalize(candidate)
            if (
                not candidate
                or len(candidate) > self.max_query_chars
                or normalized in seen
                or not protected.issubset(_protected_vocabulary(candidate))
            ):
                rejected += 1
                continue
            seen.add(normalized)
            variants.append(QueryVariant(
                query=candidate,
                kind=f"generated:{suggestion.focus}",
                weight=self.generated_weight,
            ))
        warnings = (
            (f"query expansion rejected {rejected} unfaithful variant(s)",)
            if rejected else ()
        )
        return VariantPlan(tuple(variants), warnings)


def _protected_constraints(query: str) -> set[str]:
    normalized = _normalize(query)
    words = set(re.findall(r"[\w'-]+", normalized))
    terms = words.intersection(_HARD_TERMS)
    terms.update(match.casefold() for match in _NUMBER.findall(query))
    terms.update(match.casefold() for match in _CAPITALIZED.findall(query))
    return terms


def _protected_vocabulary(query: str) -> set[str]:
    words = set(re.findall(r"[\w'-]+", _normalize(query)))
    words.update(match.casefold() for match in _NUMBER.findall(query))
    return words


def _normalize(query: str) -> str:
    return " ".join(query.casefold().split())
