from __future__ import annotations

from collections.abc import Iterator
from typing import Any, overload

from pydantic import Field, field_validator

from .base import ContractModel, NonEmptyString
from .enum import RetrievalSource
from .telemetry import RetrievalTrace


class RetrievalCandidate(ContractModel):
    """Internal frame candidate shared by retrieval pipeline stages."""

    frame_id: NonEmptyString
    source_scores: dict[RetrievalSource, float] = Field(default_factory=dict)
    source_ranks: dict[RetrievalSource, int] = Field(default_factory=dict)
    fusion_score: float | None = None
    reranker_score: float | None = None
    final_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_ranks")
    @classmethod
    def validate_source_ranks(
        cls,
        source_ranks: dict[RetrievalSource, int],
    ) -> dict[RetrievalSource, int]:
        """Require retrieval ranks to be one-based positive integers."""

        if any(rank < 1 for rank in source_ranks.values()):
            raise ValueError("source ranks must be greater than or equal to 1")

        return source_ranks


class RetrievalResult(ContractModel):
    """Candidates and telemetry owned by one retrieval call."""

    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    trace: RetrievalTrace = Field(default_factory=RetrievalTrace)
    warnings: list[NonEmptyString] = Field(default_factory=list)

    def __len__(self) -> int:
        """Preserve sequence convenience while exposing trace explicitly."""

        return len(self.candidates)

    def __iter__(self) -> Iterator[RetrievalCandidate]:
        return iter(self.candidates)

    @overload
    def __getitem__(self, index: int) -> RetrievalCandidate: ...

    @overload
    def __getitem__(self, index: slice) -> list[RetrievalCandidate]: ...

    def __getitem__(
        self, index: int | slice
    ) -> RetrievalCandidate | list[RetrievalCandidate]:
        return self.candidates[index]


class SearchScores(ContractModel):
    """Scores exposed for a returned frame."""

    visual: float | None = None
    caption: float | None = None
    ocr: float | None = None
    asr: float | None = None
    fusion: float | None = None
    reranker: float | None = None
    final: float
