"""Shared contracts for frame-level evidence and bounded temporal scenes."""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString
from .enum import RetrievalSource, TaskType
from .frame import FrameRecord
from .search import SearchFilters


class QueryUnit(ContractModel):
    """One stable semantic unit supplied to temporal evidence retrieval."""

    unit_id: NonEmptyString
    text: NonEmptyString
    order: int = Field(ge=0)


class TemporalRelation(str, Enum):
    """Small normalized relation set consumed by temporal alignment scoring."""

    BEFORE = "before"
    AFTER = "after"
    OVERLAP = "overlap"
    NEAR = "near"
    AT_END = "at_end"


class TemporalAlignmentMode(str, Enum):
    """Explicit alignment semantics selected by a task adapter."""

    PROGRESSIVE_SCENE = "progressive_scene"
    ORDERED_PATH = "ordered_path"


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


class TemporalQueryPlan(ContractModel):
    """Validated task query units and constraints supplied to an aligner."""

    task_type: TaskType
    units: tuple[QueryUnit, ...] = Field(min_length=1)
    constraints: tuple[TemporalConstraint, ...] = ()
    filters: SearchFilters | None = None
    alignment_mode: TemporalAlignmentMode

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        unit_ids = [unit.unit_id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("temporal query-unit IDs must be unique")
        if [unit.order for unit in self.units] != list(range(len(self.units))):
            raise ValueError("temporal query-unit order must be consecutive")
        expected_mode = (
            TemporalAlignmentMode.ORDERED_PATH
            if self.task_type is TaskType.TRAKE
            else TemporalAlignmentMode.PROGRESSIVE_SCENE
        )
        if self.alignment_mode is not expected_mode:
            raise ValueError(
                f"task {self.task_type.value!r} requires "
                f"{expected_mode.value!r} alignment"
            )
        known = set(unit_ids)
        for constraint in self.constraints:
            if constraint.subject_unit_id not in known:
                raise ValueError("temporal constraint references an unknown subject")
            if (
                constraint.object_unit_id is not None
                and constraint.object_unit_id not in known
            ):
                raise ValueError("temporal constraint references an unknown object")
            if (
                self.alignment_mode is TemporalAlignmentMode.ORDERED_PATH
                and constraint.relation is not TemporalRelation.BEFORE
            ):
                raise ValueError("ordered-path plans accept only BEFORE constraints")
        return self


class OrderedPathCandidate(ContractModel):
    """One canonical same-video frame path aligned to ordered query units."""

    path_id: NonEmptyString
    video_id: NonEmptyString
    frames: tuple[FrameRecord, ...] = Field(min_length=1)
    query_unit_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    score: float
    reason_labels: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if len(self.frames) != len(self.query_unit_ids):
            raise ValueError("ordered path must contain one frame per query unit")
        if len(self.query_unit_ids) != len(set(self.query_unit_ids)):
            raise ValueError("ordered path query-unit IDs must be unique")
        if any(frame.video_id != self.video_id for frame in self.frames):
            raise ValueError("ordered path frames must belong to path.video_id")
        if any(
            current.timestamp_ms < previous.timestamp_ms
            for previous, current in zip(self.frames, self.frames[1:])
        ):
            raise ValueError("ordered path frames must be chronological")
        return self
