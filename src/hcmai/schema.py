"""Compatibility alias for shared schema models."""

from hcmai.common.schemas.base import ContractModel, NonEmptyString
from hcmai.common.schemas.conversation import ConversationTurn, FrameFeedback
from hcmai.common.schemas.enum import (
    ProcessingStatus,
    QueryDifficulty,
    QueryLanguage,
    RetrievalSource,
    SearchMode,
    TaskType,
)
from hcmai.common.schemas.evaluation import EvaluationQuery
from hcmai.common.schemas.frame import FrameEnrichment, FrameRecord
from hcmai.common.schemas.retrieval import RetrievalCandidate, SearchScores
from hcmai.common.schemas.search import (
    SearchFilters,
    SearchLatency,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

__all__ = [
    "ContractModel",
    "NonEmptyString",
    "ConversationTurn",
    "FrameFeedback",
    "ProcessingStatus",
    "QueryDifficulty",
    "QueryLanguage",
    "RetrievalSource",
    "SearchMode",
    "TaskType",
    "EvaluationQuery",
    "FrameEnrichment",
    "FrameRecord",
    "RetrievalCandidate",
    "SearchScores",
    "SearchFilters",
    "SearchLatency",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]
