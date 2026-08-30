"""Task-agnostic ordered event-to-frame alignment interfaces.

This package owns deterministic event splitting and the numerical DP decoder.
Timed search orchestration lives in ``hcmai.orchestration.temporal_search``.
"""

from .dp import AlignedPath
from .planner import split_query_events

__all__ = [
    "AlignedPath",
    "split_query_events",
]
