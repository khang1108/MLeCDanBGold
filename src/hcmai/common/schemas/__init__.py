from .base import *
from .enum import *
from .frame import FrameEnrichment, FrameRecord, validate_frame_enrichment
from .inference import (
    CaptionItem,
    CaptionResponse,
    InferenceCapabilities,
    InferenceReadiness,
    ModelStatus,
    RerankItem,
    RerankResponse,
    TextEmbeddingRequest,
    TextEmbeddingResponse,
)
from .retrieval import RetrievalCandidate, RetrievalResult, SearchScores
from .search import (
    SearchFilters,
    SearchLatency,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from .submission import SubmissionResult
from .task import TaskRequest, TaskResponse
from .telemetry import PipelineTrace, RetrievalTrace, StageStatus, StageTrace
from .temporal import (
    FrameEvidence,
    OrderedPathCandidate,
    QueryUnit,
    SceneCandidate,
    TemporalAlignmentMode,
    TemporalConstraint,
    TemporalQueryPlan,
    TemporalRelation,
)
from .trake import (
    TRAKERequest,
    TRAKEResponse,
    TRAKESubmission,
)
from .transcript import TranscriptSegment
from .vqa import (
    VQABaselineProfile,
    VQARetrievalEvidence,
    VQAInferenceEvidence,
    VQAInferenceEvidenceItem,
    VQAInferenceRequest,
    VQAInferenceResponse,
    VQARequest,
    VQAResponse,
    VQASubmission,
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
    "ExecutionProfile",
    "QueryDifficulty",
    "SearchScores",
    "RetrievalCandidate",
    "RetrievalResult",
    "StageStatus",
    "StageTrace",
    "PipelineTrace",
    "RetrievalTrace",
    "SearchFilters",
    "SearchRequest",
    "MessageRequest",
    "SearchLatency",
    "SearchResult",
    "SearchResponse",
    "MessageResponse",
    "SubmissionResult",
    "FrameRecord",
    "FrameEnrichment",
    "validate_frame_enrichment",
    "QueryUnit",
    "FrameEvidence",
    "OrderedPathCandidate",
    "SceneCandidate",
    "TemporalAlignmentMode",
    "TemporalConstraint",
    "TemporalQueryPlan",
    "TemporalRelation",
    "CaptionItem",
    "CaptionResponse",
    "InferenceCapabilities",
    "InferenceReadiness",
    "ModelStatus",
    "RerankItem",
    "RerankResponse",
    "TextEmbeddingRequest",
    "TextEmbeddingResponse",
    "VQAInferenceEvidence",
    "VQAInferenceEvidenceItem",
    "VQABaselineProfile",
    "VQARetrievalEvidence",
    "VQAInferenceRequest",
    "VQAInferenceResponse",
    "VQARequest",
    "VQAResponse",
    "VQASubmission",
    "TRAKERequest",
    "TRAKEResponse",
    "TRAKESubmission",
    "TaskRequest",
    "TaskResponse",
    "TranscriptSegment",
]
