"""Public Pydantic contracts for the HCMAI HTTP and WebSocket APIs."""

from .database import (
    DatabaseColumn,
    DatabaseQueryRequest,
    DatabaseQueryResponse,
    DatabaseRowsPage,
    DatabaseTable,
    DatabaseTableList,
)
from .frames import CatalogTranscriptSegment, FrameCatalogEntry, FrameInspectionResponse
from .filter import FilterMetadataFilters, FilterRequest, FilterResponse, FilterResult
from .history import (
    QueryHistoryCreate,
    QueryHistoryList,
    QueryHistoryRecord,
    QueryHistoryViewedFrameUpdate,
)
from .kis import (
    KISInput,
    KISIntent,
    KISRevisionSearchRequest,
    KISRevisionSearchResponse,
)
from .latency import SearchLatency
from .query_candidates import (
    QueryCandidateResponse,
    QueryCandidatesRequest,
    QueryCandidatesResponse,
)
from .search import (
    ImageSearchResponse,
    SearchResult,
    SearchResultMetadata,
)
from .trake import TRAKEPath, TRAKERequest, TRAKEResponse
from .workspace import (
    AnswerAddFrame,
    AnswerAddText,
    AnswerCandidate,
    AnswerCandidateMutation,
    AnswerClear,
    AnswerDelete,
    AnswerModeSet,
    AnswerModeMutation,
    AnswerTaskClearAndSwitch,
    AnswerUpdateFrame,
    AnswerUpdateText,
    AnswerWorkspaceCommand,
    AnswerWorkspaceEvent,
    AnswerWorkspaceSnapshot,
    SubmissionAttemptSummary,
)
from .vbs import (
    VbsAvsSubmissionRequest,
    VbsCandidateRevision,
    VbsSessionConnectRequest,
    VbsSessionStatus,
    VbsSingleSubmissionRequest,
    VbsSubmissionResolutionRequest,
    VbsSubmissionResponse,
)


__all__ = [
    "CatalogTranscriptSegment",
    "AnswerAddFrame",
    "AnswerAddText",
    "AnswerCandidate",
    "AnswerCandidateMutation",
    "AnswerClear",
    "AnswerDelete",
    "AnswerModeSet",
    "AnswerModeMutation",
    "AnswerTaskClearAndSwitch",
    "AnswerUpdateFrame",
    "AnswerUpdateText",
    "AnswerWorkspaceCommand",
    "AnswerWorkspaceEvent",
    "AnswerWorkspaceSnapshot",
    "DatabaseColumn",
    "DatabaseQueryRequest",
    "DatabaseQueryResponse",
    "DatabaseRowsPage",
    "DatabaseTable",
    "DatabaseTableList",
    "FrameCatalogEntry",
    "FrameInspectionResponse",
    "FilterMetadataFilters",
    "FilterRequest",
    "FilterResponse",
    "FilterResult",
    "ImageSearchResponse",
    "KISInput",
    "KISIntent",
    "KISRevisionSearchRequest",
    "KISRevisionSearchResponse",
    "QueryCandidateResponse",
    "QueryCandidatesRequest",
    "QueryCandidatesResponse",
    "QueryHistoryCreate",
    "QueryHistoryList",
    "QueryHistoryRecord",
    "QueryHistoryViewedFrameUpdate",
    "SearchLatency",
    "SearchResult",
    "SearchResultMetadata",
    "SubmissionAttemptSummary",
    "VbsAvsSubmissionRequest",
    "VbsCandidateRevision",
    "VbsSessionConnectRequest",
    "VbsSessionStatus",
    "VbsSingleSubmissionRequest",
    "VbsSubmissionResolutionRequest",
    "VbsSubmissionResponse",
    "TRAKEPath",
    "TRAKERequest",
    "TRAKEResponse",
]
