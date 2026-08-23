"""Multimodal evidence construction and bounded VQA inference."""

from .answerer import answer_windows
from .evidence import build_evidence_bundle, select_question_evidence
from .windows import expand_neighbor_window

__all__ = [
    "answer_windows",
    "build_evidence_bundle",
    "expand_neighbor_window",
    "select_question_evidence",
]
