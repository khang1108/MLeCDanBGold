"""Runtime retrieval values and evidence-source identifiers.

These frozen dataclasses move internal ranking state out of the former shared
Pydantic schema package. They preserve canonical candidate identity and
modality provenance; artifact and HTTP validation belong to their boundaries.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, overload

from hcmai.common.observability.models import RetrievalTrace


class RetrievalSource(str, Enum):
    """Evidence channels used to retrieve a frame."""

    VISUAL = "visual"
    CONTEXT = "context"
    CAPTION = "caption"
    OCR = "ocr"
    ASR = "asr"


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """Internal frame candidate shared by retrieval pipeline stages."""

    frame_id: str
    source_scores: dict[RetrievalSource, float] = field(default_factory=dict)
    source_ranks: dict[RetrievalSource, int] = field(default_factory=dict)
    fusion_score: float | None = None
    reranker_score: float | None = None
    final_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject blank identity and invalid one-based source ranks."""

        if not self.frame_id.strip():
            raise ValueError("frame_id must be non-empty")
        if any(rank < 1 for rank in self.source_ranks.values()):
            raise ValueError("source ranks must be greater than or equal to 1")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Candidates and request-local telemetry owned by one retrieval call."""

    candidates: list[RetrievalCandidate] = field(default_factory=list)
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    warnings: list[str] = field(default_factory=list)
    time_to_first_candidate_ms: float | None = None

    def __post_init__(self) -> None:
        """Validate optional latency and non-empty warning messages."""

        if (
            self.time_to_first_candidate_ms is not None
            and self.time_to_first_candidate_ms < 0
        ):
            raise ValueError("time_to_first_candidate_ms must be non-negative")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("warnings must be non-empty")

    def __len__(self) -> int:
        """Return the number of candidates."""

        return len(self.candidates)

    def __iter__(self) -> Iterator[RetrievalCandidate]:
        """Iterate candidates in ranking order."""

        return iter(self.candidates)

    @overload
    def __getitem__(self, index: int) -> RetrievalCandidate: ...

    @overload
    def __getitem__(self, index: slice) -> list[RetrievalCandidate]: ...

    def __getitem__(
        self, index: int | slice
    ) -> RetrievalCandidate | list[RetrievalCandidate]:
        """Return one candidate or a ranked slice."""

        return self.candidates[index]


__all__ = [
    "RetrievalCandidate",
    "RetrievalResult",
    "RetrievalSource",
]
