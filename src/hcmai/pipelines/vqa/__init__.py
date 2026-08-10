"""Training-free competition VQA domain components."""

from .models import (
    BranchCandidate,
    EvidenceBundle,
    EvidenceItem,
    GroundedAnswerCandidate,
    LocalizedWindow,
    ParsedVQAQuery,
    QuestionType,
    TemporalWindow,
    VideoEvidenceCandidate,
)
from .parser import parse_vqa_query

__all__ = [
    "BranchCandidate",
    "EvidenceBundle",
    "EvidenceItem",
    "GroundedAnswerCandidate",
    "LocalizedWindow",
    "ParsedVQAQuery",
    "QuestionType",
    "TemporalWindow",
    "VideoEvidenceCandidate",
    "parse_vqa_query",
]
