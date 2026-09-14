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
    KISInput,
    KISIntent,
    KISRevisionSearchRequest,
    KISRevisionSearchResponse,
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
    "KISInput",
    "KISIntent",
    "KISRevisionSearchRequest",
    "KISRevisionSearchResponse",
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
