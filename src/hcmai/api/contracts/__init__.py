"""Public FastAPI request and response contracts.

This package owns thin Pydantic HTTP-boundary models for KIS, TRAKE, frame
catalog, and submission routes.
"""

from .frames import CatalogTranscriptSegment, FrameCatalogEntry
from .latency import SearchLatency
from .query_candidates import (
    QueryCandidateResponse,
    QueryCandidatesRequest,
    QueryCandidatesResponse,
)
from .search import SearchRequest, SearchResponse, SearchResult, SearchResultMetadata
from .submission import SubmissionResult
from .trake import TRAKEPath, TRAKERequest, TRAKEResponse

__all__ = [
    "CatalogTranscriptSegment",
    "FrameCatalogEntry",
    "QueryCandidateResponse",
    "QueryCandidatesRequest",
    "QueryCandidatesResponse",
    "SearchLatency",
    "SearchRequest",
    "SearchResult",
    "SearchResultMetadata",
    "SearchResponse",
    "SubmissionResult",
    "TRAKEPath",
    "TRAKERequest",
    "TRAKEResponse",
]
