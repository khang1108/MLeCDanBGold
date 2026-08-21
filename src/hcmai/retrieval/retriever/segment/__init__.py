"""Segment-native retrieval artifacts for timestamped ASR evidence.

This package keeps ASR segments separate from frame-native retrieval until a
later projection step explicitly maps segment evidence to canonical frames.
"""

from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
from hcmai.retrieval.retriever.segment.projector import (
    SegmentFrameProjection,
    SegmentFrameProjector,
)
from hcmai.retrieval.retriever.segment.retriever import ASRSegmentRetriever

__all__ = [
    "ASRSegmentRetriever",
    "SegmentDenseIndex",
    "SegmentFrameProjection",
    "SegmentFrameProjector",
]
