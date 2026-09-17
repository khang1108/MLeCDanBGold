"""Standalone HTTP serving package for retrieval capabilities."""

from hcmai.retrieval.serving.client import (
    RemoteImageCandidate,
    RemoteImageSearchResult,
    RemoteRetrievalStatus,
    RetrievalHttpClient,
)
from hcmai.retrieval.serving.remote import (
    RemoteImageSearchService,
    RemoteTemporalSearchService,
)
from hcmai.retrieval.serving.runtime import RetrievalCapabilities, RetrievalRuntime
from hcmai.retrieval.serving.server import create_app
from hcmai.retrieval.serving.utils.config import RetrievalClientSettings
from hcmai.retrieval.serving.utils.errors import (
    RetrievalClientError,
    RetrievalInternalError,
    RetrievalInvalidRequestError,
    RetrievalNotFoundError,
    RetrievalProtocolError,
    RetrievalTooLargeError,
    RetrievalUnavailableError,
)

__all__ = [
    "RemoteImageCandidate",
    "RemoteImageSearchResult",
    "RemoteImageSearchService",
    "RemoteRetrievalStatus",
    "RemoteTemporalSearchService",
    "RetrievalCapabilities",
    "RetrievalClientError",
    "RetrievalClientSettings",
    "RetrievalHttpClient",
    "RetrievalInternalError",
    "RetrievalInvalidRequestError",
    "RetrievalNotFoundError",
    "RetrievalProtocolError",
    "RetrievalRuntime",
    "RetrievalTooLargeError",
    "RetrievalUnavailableError",
    "create_app",
]
