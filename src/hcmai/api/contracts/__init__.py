"""Public Pydantic contracts for the HCMAI HTTP and WebSocket APIs."""


from .frames import CatalogTranscriptSegment, FrameCatalogEntry, FrameInspectionResponse
from .filter import FilterMetadataFilters, FilterRequest, FilterResponse, FilterResult
from .history import (
    QueryHistoryCreate,
    QueryHistoryList,
    QueryHistoryRecord,
    QueryHistoryViewedFrameUpdate,
)
from .latency import SearchLatency
from .query_candidates import (
    QueryCandidateResponse,
    QueryCandidatesRequest,
    QueryCandidatesResponse,
)
from .search import (
    ImageSearchResponse,
    SearchRequest,
    SearchResponse,
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
    "QueryCandidateResponse",
    "QueryCandidatesRequest",
    "QueryCandidatesResponse",
    "QueryHistoryCreate",
    "QueryHistoryList",
    "QueryHistoryRecord",
    "QueryHistoryViewedFrameUpdate",
    "SearchLatency",
    "SearchRequest",
    "SearchResponse",
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
