"""Typed private VQA domain models; shared API contracts remain authoritative."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from hcmai.common.schemas import FrameRecord, QueryLanguage, RetrievalSource


class QuestionType(str, Enum):
    GENERAL = "general"
    COUNT = "count"
    COLOR = "color"
    TEXT = "text"
    SPEECH = "speech"
    TEMPORAL = "temporal"
    IDENTITY = "identity"


@dataclass(frozen=True)
class ParsedVQAQuery:
    retrieval_query: str
    question: str
    question_type: QuestionType
    required_modalities: tuple[RetrievalSource, ...]
    answer_language: QueryLanguage
    clue_queries: tuple[str, ...] = ()
    parser_confidence: float = 1.0


@dataclass(frozen=True)
class BranchCandidate:
    frame: FrameRecord
    branch_scores: dict[str, float]
    source_scores: dict[RetrievalSource, float]
    source_ranks: dict[RetrievalSource, int]
    score: float
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class VideoEvidenceCandidate:
    video_id: str
    frames: tuple[BranchCandidate, ...]
    score: float
    best_event_rank: int | None
    best_question_rank: int | None
    modality_count: int
    neighborhood_count: int
    clue_coverage: float


@dataclass(frozen=True)
class TemporalWindow:
    window_id: str
    video_id: str
    start_ms: int
    end_ms: int
    source_frames: tuple[BranchCandidate, ...]
    sampled_frames: tuple[FrameRecord, ...]
    score: float

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return tuple(frame.frame_id for frame in self.sampled_frames)


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    value: str
    frame_id: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    provenance: str = "local_artifact"


@dataclass(frozen=True)
class EvidenceBundle:
    window: TemporalWindow
    items: tuple[EvidenceItem, ...]
    image_frame_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalizedWindow:
    bundle: EvidenceBundle
    score: float
    reason_labels: tuple[str, ...]


@dataclass(frozen=True)
class GroundedAnswerCandidate:
    window: TemporalWindow
    evidence_frame_id: str
    answer: str
    normalized_answer: str
    video_score: float
    frame_score: float
    localization_score: float
    evidence_coverage_score: float
    answer_confidence: float
    consistency_score: float = 0.0
    grounded: bool = True
    joint_score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
