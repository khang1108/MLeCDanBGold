"""Multimodal evidence construction and bounded VQA inference."""

from .answerer import answer_windows
from .evidence import build_evidence_bundle, select_question_evidence

__all__ = ["answer_windows", "build_evidence_bundle", "select_question_evidence"]
