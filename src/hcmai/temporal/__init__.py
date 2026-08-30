"""Task-agnostic ordered event-to-frame alignment interfaces.

This package owns deterministic planning, score-matrix alignment, and canonical
path materialization. Task-specific output projection remains in workflows.
"""

from .planner import build_alignment_plan
from .service import AlignmentResult, TemporalAlignmentService

__all__ = [
    "AlignmentResult",
    "TemporalAlignmentService",
    "build_alignment_plan",
]
