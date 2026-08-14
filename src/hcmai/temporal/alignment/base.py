from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from hcmai.temporal.plan import TemporalQueryPlan

EvidenceT = TypeVar("EvidenceT", contravariant=True)
CandidateT = TypeVar("CandidateT", covariant=True)


@dataclass(frozen=True, slots=True)
class AlignmentResult(Generic[CandidateT]):
    """Alignment candidates ordered from strongest to weakest."""

    candidates: tuple[CandidateT, ...] = ()


class TemporalAligner(Protocol[EvidenceT, CandidateT]):
    """Convert temporal evidence into ranked scene or path candidates."""

    def align(
        self,
        plan: TemporalQueryPlan,
        evidence: Sequence[EvidenceT],
    ) -> AlignmentResult[CandidateT]: ...
