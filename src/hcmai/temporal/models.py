from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class RelationType(StrEnum):
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    SAME_SCENE = "same_scene"


class EvidenceStatus(StrEnum):
    UNKNOWN = "unknown"
    EVALUATED_NO_MATCH = "evaluated_no_match"
    MATCHED = "matched"


@dataclass(frozen=True, slots=True)
class QueryUnit:
    unit_id: str
    text: str
    reveal_index: int


@dataclass(frozen=True, slots=True)
class TemporalConstraint:
    left_unit_id: str
    relation: RelationType
    right_unit_id: str
    hard: bool = True


@dataclass(frozen=True, slots=True)
class EvidencePoint:
    unit_id: str
    video_id: str
    frame_id: str
    frame_idx: int
    timestamp_ms: int
    relevance_score: float
    source_scores: Mapping[str, float] = field(default_factory=dict)
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if any(
            not 0.0 <= score <= 1.0
            for score in (self.relevance_score, *self.source_scores.values())
        ):
            raise ValueError("scores must be between 0 and 1")

    @property
    def canonical_identity(self) -> tuple[str, str, int]:
        return (self.video_id, self.frame_id, self.frame_idx)


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    status: EvidenceStatus = EvidenceStatus.UNKNOWN
    points: tuple[EvidencePoint, ...] = ()

    def __post_init__(self) -> None:
        identities = tuple(point.canonical_identity for point in self.points)
        if len(identities) != len(set(identities)):
            raise ValueError("evidence set contains duplicate canonical identities")


@dataclass(frozen=True, slots=True)
class SceneCandidate:
    video_id: str
    start_ms: int
    end_ms: int
    evidence_by_unit: Mapping[str, EvidenceSet] = field(default_factory=dict)
    unit_scores: Mapping[str, float] = field(default_factory=dict)
    coverage_score: float = 0.0
    semantic_score: float = 0.0
    temporal_score: float = 0.0
    relation_score: float = 0.0
    final_score: float = 0.0

    def __post_init__(self) -> None:
        for name, bound in (("start_ms", self.start_ms), ("end_ms", self.end_ms)):
            if bound < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.end_ms < self.start_ms:
            raise ValueError("scene candidate end_ms must be greater than or equal to start_ms")
        if any(
            not 0.0 <= score <= 1.0
            for score in (
                *self.unit_scores.values(),
                self.coverage_score,
                self.semantic_score,
                self.temporal_score,
                self.relation_score,
                self.final_score,
            )
        ):
            raise ValueError("scores must be between 0 and 1")
