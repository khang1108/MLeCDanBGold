"""Task-agnostic ordered event-to-frame alignment interfaces.

This package owns deterministic event splitting and the numerical DP decoder.
Timed search orchestration lives in ``hcmai.orchestration.temporal_search``.
"""

from .dp import DPPath, AlignedPath, align_video, cluster_starts, rank_paths
from .events import normalize_event_texts

__all__ = [
    "AlignedPath",
    "DPPath",
    "align_video",
    "cluster_starts",
    "normalize_event_texts",
    "rank_paths",
]
