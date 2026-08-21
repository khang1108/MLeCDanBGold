"""Fallback VQA localization used only when the shared temporal core is absent."""

from .candidates import retrieve_candidates
from .localizer import SimilarityLocalizer
from .video_aggregation import aggregate_videos
from .windows import build_windows, expand_neighbor_window

__all__ = [
    "SimilarityLocalizer",
    "aggregate_videos",
    "build_windows",
    "expand_neighbor_window",
    "retrieve_candidates",
]
