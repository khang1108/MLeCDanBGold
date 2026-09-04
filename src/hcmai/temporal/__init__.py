"""Task-agnostic ordered event-to-frame alignment interfaces.

This package owns deterministic event splitting and the numerical DP decoder.
Timed search orchestration lives in ``hcmai.orchestration.temporal_search``.
"""

from .dp import DPPath, AlignedPath, align_video, cluster_starts, rank_paths
from .planner import plan_query_events, split_query_events

__all__ = [
    "AlignedPath",
    "DPPath",
    "align_video",
    "cluster_starts",
    "plan_query_events",
    "rank_paths",
    "split_query_events",
]
