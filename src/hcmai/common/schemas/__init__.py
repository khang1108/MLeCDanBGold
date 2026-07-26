from .base import *
from .conversation import (
    ConversationConstraint,
    ConversationSession,
    ConversationTurn,
    FrameFeedback,
    SubmissionResult,
)
from .enum import *
from .frame import FrameEnrichment, FrameRecord
from .retrieval import RetrievalCandidate, SearchScores
from .search import (
    SearchFilters,
    SearchLatency,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

# Backward-compatible names used by the earlier frontend/backend contract.
# They intentionally point at the canonical search models instead of creating
# a second request/response shape.
MessageRequest = SearchRequest
MessageResponse = SearchResponse

__all__ = [
    "NonEmptyString",
    "ContractModel",
    "ProcessingStatus",
    "RetrievalSource",
    "QueryLanguage",
    "TaskType",
    "QueryDifficulty",
    "SearchScores",
    "RetrievalCandidate",
    "SearchMode",
    "SearchFilters",
    "SearchRequest",
    "MessageRequest",
    "SearchLatency",
    "SearchResult",
    "SearchResponse",
    "MessageResponse",
    "ConversationConstraint",
    "ConversationTurn",
    "FrameFeedback",
    "ConversationSession",
    "SubmissionResult",
    "FrameRecord",
    "FrameEnrichment",
]
