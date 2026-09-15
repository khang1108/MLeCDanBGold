"""Public Pydantic contracts for the HCMAI HTTP and WebSocket APIs."""


from .frames import CatalogTranscriptSegment, FrameCatalogEntry, FrameInspectionResponse
from .filter import FilterMetadataFilters, FilterRequest, FilterResponse, FilterResult
from .history import (
    QueryHistoryCreate,
    QueryHistoryList,
    QueryHistoryRecord,
    QueryHistoryViewedFrameUpdate,
)
from .kis import (
    EventPatch,
    GlobalRewriteOperation,
    InitialResolveOperation,
    KISExplorationEventSeed,
    KISExplorationSeed,
    KISIntent,
    KISOperation,
    KISOperationSummary,
    KISSearchRequest,
    KISSearchResponse,
    PatchEventsOperation,
    SearchOnlyOperation,
)
from .latency import SearchLatency
from .search import (
    ImageSearchResponse,
    SearchResult,
    SearchResultMetadata,
)
from .trake import TRAKEPath, TRAKERequest, TRAKEResponse
from .vbs import (
    VbsDirectSubmissionNotRecorded,
    VbsDirectSubmissionOutcome,
    VbsDirectSubmissionRecorded,
    VbsDirectSubmissionRequest,
    VbsDirectSubmissionUnknown,
    VbsSessionConnectRequest,
    VbsSessionStatus,
    VbsTaskResponse,
    VbsTemporalAnswer,
    VbsTextAnswer,
)


__all__ = [
    "CatalogTranscriptSegment",
    "FrameCatalogEntry",
    "FrameInspectionResponse",
    "FilterMetadataFilters",
    "FilterRequest",
    "FilterResponse",
    "FilterResult",
    "ImageSearchResponse",
    "EventPatch",
    "GlobalRewriteOperation",
    "InitialResolveOperation",
    "KISExplorationEventSeed",
    "KISExplorationSeed",
    "KISIntent",
    "KISOperation",
    "KISOperationSummary",
    "KISSearchRequest",
    "KISSearchResponse",
    "PatchEventsOperation",
    "SearchOnlyOperation",
    "QueryHistoryCreate",
    "QueryHistoryList",
    "QueryHistoryRecord",
    "QueryHistoryViewedFrameUpdate",
    "SearchLatency",
    "SearchResult",
    "SearchResultMetadata",
    "VbsDirectSubmissionNotRecorded",
    "VbsDirectSubmissionOutcome",
    "VbsDirectSubmissionRecorded",
    "VbsDirectSubmissionRequest",
    "VbsDirectSubmissionUnknown",
    "VbsSessionConnectRequest",
    "VbsSessionStatus",
    "VbsTaskResponse",
    "VbsTemporalAnswer",
    "VbsTextAnswer",
    "TRAKEPath",
    "TRAKERequest",
    "TRAKEResponse",
]
