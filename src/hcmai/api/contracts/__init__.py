"""Public Pydantic contracts for the HCMAI HTTP and WebSocket APIs."""

from .database import (
    DatabaseColumn,
    DatabaseQueryRequest,
    DatabaseQueryResponse,
    DatabaseRowsPage,
    DatabaseTable,
    DatabaseTableList,
)
from .frames import CatalogTranscriptSegment, FrameCatalogEntry
from .filter import FilterRequest, FilterResponse, FilterResult
from .history import (
    QueryHistoryCreate,
    QueryHistoryList,
    QueryHistoryRecord,
    QueryHistorySubmissionUpdate,
    QueryHistoryViewedFrameUpdate,
    SubmissionFile,
    SubmissionFileClear,
    SubmissionFileCreate,
    SubmissionFileDelete,
    SubmissionFileList,
    SubmissionFileUpdate,
    SubmissionFileValidate,
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
from .submission import SubmissionResult
from .trake import TRAKEPath, TRAKERequest, TRAKEResponse


__all__ = [
    "CatalogTranscriptSegment",
    "DatabaseColumn",
    "DatabaseQueryRequest",
    "DatabaseQueryResponse",
    "DatabaseRowsPage",
    "DatabaseTable",
    "DatabaseTableList",
    "FrameCatalogEntry",
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
    "QueryHistorySubmissionUpdate",
    "QueryHistoryViewedFrameUpdate",
    "SearchLatency",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchResultMetadata",
    "SubmissionFile",
    "SubmissionFileClear",
    "SubmissionFileCreate",
    "SubmissionFileDelete",
    "SubmissionFileList",
    "SubmissionFileUpdate",
    "SubmissionFileValidate",
    "SubmissionResult",
    "TRAKEPath",
    "TRAKERequest",
    "TRAKEResponse",
]
