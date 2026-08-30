"""Public FastAPI request and response contracts.

This package owns thin Pydantic HTTP-boundary models for the KIS and TRAKE
routes. It does not replace the current legacy shared schema package until all
consumers are migrated in later cleanup tasks.
"""

from .latency import SearchLatency
from .search import SearchRequest, SearchResponse, SearchResult, SearchResultMetadata
from .trake import TRAKEPath, TRAKERequest, TRAKEResponse

__all__ = [
    "SearchLatency",
    "SearchRequest",
    "SearchResult",
    "SearchResultMetadata",
    "SearchResponse",
    "TRAKEPath",
    "TRAKERequest",
    "TRAKEResponse",
]
