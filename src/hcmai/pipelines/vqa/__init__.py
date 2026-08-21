"""Training-free competition VQA domain components."""

from .domain import (
    EvidenceBundle,
    EvidenceItem,
    GroundedAnswerCandidate,
    ParsedVQAQuery,
    QuestionType,
    VideoEvidenceCandidate,
)
from .query import parse_vqa_query

__all__ = [
    "EvidenceBundle",
    "EvidenceItem",
    "GroundedAnswerCandidate",
    "ParsedVQAQuery",
    "QuestionType",
    "VideoEvidenceCandidate",
    "parse_vqa_query",
]
