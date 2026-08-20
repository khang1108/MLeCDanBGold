"""Segment-native retrieval artifacts for timestamped ASR evidence.

This package keeps ASR segments separate from frame-native retrieval until a
later projection step explicitly maps segment evidence to canonical frames.
"""

from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

__all__ = ["SegmentDenseIndex"]
