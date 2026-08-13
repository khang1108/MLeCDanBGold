"""Shared contracts for frame-level evidence and bounded temporal scenes."""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString
from .enum import RetrievalSource
from .frame import FrameRecord


class QueryUnit(ContractModel):
    """One stable semantic unit supplied to temporal evidence retrieval."""

    unit_id: NonEmptyString
    text: NonEmptyString
    order: int = Field(ge=0)


class TemporalRelation(str, Enum):
    """Small normalized relation set consumed by KIS/VQA scene scoring."""

    BEFORE = "before"
    AFTER = "after"
    OVERLAP = "overlap"
    NEAR = "near"
    AT_END = "at_end"


class TemporalConstraint(ContractModel):
    """Soft, explainable relation attached to progressive query-unit IDs."""

    relation: TemporalRelation
    subject_unit_id: NonEmptyString
    object_unit_id: NonEmptyString | None = None
    reason: NonEmptyString


class FrameEvidence(ContractModel):
    """Canonical frame identity with query-unit and retrieval provenance."""

    frame: FrameRecord
    unit_scores: dict[NonEmptyString, float] = Field(default_factory=dict)
    source_scores: dict[RetrievalSource, float] = Field(default_factory=dict)
    source_ranks: dict[RetrievalSource, int] = Field(default_factory=dict)
    score: float
    provenance: tuple[NonEmptyString, ...] = ()


class SceneCandidate(ContractModel):
    """Bounded temporal localization unit built from canonical frame evidence."""

    scene_id: NonEmptyString
    video_id: NonEmptyString
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    evidence: tuple[FrameEvidence, ...] = Field(min_length=1)
    unit_scores: dict[NonEmptyString, float] = Field(default_factory=dict)
    semantic_score: float = 0.0
    coverage_score: float = 0.0
    evaluation_coverage_score: float = 0.0
    temporal_score: float = 0.0
    relation_score: float | None = None
    final_score: float = 0.0
    reason_labels: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_scene_identity(self) -> Self:
        """Keep every evidence frame inside the canonical scene identity."""

        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        for item in self.evidence:
            if item.frame.video_id != self.video_id:
                raise ValueError("scene evidence must belong to scene.video_id")
            if not self.start_ms <= item.frame.timestamp_ms <= self.end_ms:
                raise ValueError("scene evidence timestamp must be inside scene range")
        return self
