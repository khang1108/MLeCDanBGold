"""Transport contracts owned by the model-serving layer."""
from .common import HTTPContract, NonEmptyString
from .embeddings import TextEmbeddingData, TextEmbeddingRequest, TextEmbeddingResponse
from .enrichment import CaptionItem, CaptionResponse, OCRItem, OCRRegionItem, OCRResponse, ObjectItem, ObjectResponse
from .generation import BoundaryScoreResponse, ChatChoice, ChatChoiceMessage, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, JsonSchemaSpec, ResponseFormat
from .readiness import InferenceReadiness, ReadinessCapabilities, ReadinessModel
from .transcripts import AudioReferenceRequest, DiarizationRequest, TranscriptInferenceResponse, TranscriptSegment

__all__ = [name for name in globals() if not name.startswith("_")]
