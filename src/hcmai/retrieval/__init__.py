"""Public runtime retrieval services and ranking values.

The service is loaded lazily because its concrete index adapters depend on
optional serving-time packages such as FAISS; importing value types must stay
safe for lightweight tooling and architecture checks.
"""

from typing import TYPE_CHECKING, Any

from .models import RetrievalCandidate, RetrievalResult, RetrievalSource

if TYPE_CHECKING:
    from .retriever.pipeline import RetrievalService


def __getattr__(name: str) -> Any:
    """Load the optional retrieval service only when it is requested."""

    if name == "RetrievalService":
        from .retriever.pipeline import RetrievalService

        return RetrievalService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RetrievalCandidate",
    "RetrievalResult",
    "RetrievalService",
    "RetrievalSource",
]
