"""Public Pydantic contracts for the HCMAI HTTP and WebSocket APIs."""

from .database import (
    DatabaseColumn,
    DatabaseRowsPage,
    DatabaseTable,
    DatabaseTableList,
)
from .frames import CatalogTranscriptSegment, FrameCatalogEntry
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
from .search import SearchRequest, SearchResponse, SearchResult, SearchResultMetadata
from .submission import SubmissionResult
from .trake import TRAKEPath, TRAKERequest, TRAKEResponse


__all__ = [
    "CatalogTranscriptSegment",
    "DatabaseColumn",
    "DatabaseRowsPage",
    "DatabaseTable",
    "DatabaseTableList",
    "FrameCatalogEntry",
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
