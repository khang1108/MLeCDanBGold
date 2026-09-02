"""HTTP contracts for stateless query-candidate generation.

This module owns public request/response validation. It does not split KIS
queries, call inference, or persist candidate state.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

_NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class QueryCandidatesRequest(BaseModel):
    """Accept either one raw KIS query or explicit TRAKE event boundaries."""

    model_config = ConfigDict(extra="forbid")

    query: _NonBlankString | None = None
    events: list[_NonBlankString] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_exactly_one_input(self) -> Self:
        """Require exactly one caller-owned input representation."""

        if (self.query is None) == (self.events is None):
            raise ValueError("exactly one of query or events is required")
        return self


class QueryCandidateResponse(BaseModel):
    """One indexed aligned retrieval-candidate bundle."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1, le=5)
    events: list[_NonBlankString] = Field(min_length=1)


class QueryCandidatesResponse(BaseModel):
    """Complete stateless literal translation and five candidate bundles."""

    model_config = ConfigDict(extra="forbid")

    original_events: list[_NonBlankString] = Field(min_length=1)
    literal_en: list[_NonBlankString] = Field(min_length=1)
    candidates: list[QueryCandidateResponse] = Field(min_length=5, max_length=5)
    query_preparation_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_event_alignment(self) -> Self:
        """Keep every response bundle aligned to the original event count."""

        expected = len(self.original_events)
        if len(self.literal_en) != expected or any(
            len(candidate.events) != expected for candidate in self.candidates
        ):
            raise ValueError("query candidate event arrays must have equal lengths")
        if [candidate.index for candidate in self.candidates] != list(range(1, 6)):
            raise ValueError("query candidate indexes must be ordered 1 through 5")
        return self