"""Contracts shared by the local search backend and remote model service."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from .base import ContractModel, NonEmptyString
from .transcript import TranscriptSegment


class TextEmbeddingRequest(ContractModel):
    """Ordered text batch routed to one configured retrieval encoder.

    The service validates the batch ceiling against the selected encoder at
    runtime. Keeping that limit out of this shared schema lets each deployment
    tune its visual and text models independently.
    """

    source: Literal["visual", "text"] = "visual"
    texts: list[NonEmptyString] = Field(min_length=1)


class EmbeddingResponse(ContractModel):
    """Schema chuẩn hóa cho kết quả trả về từ endpoint tính toán Embedding.
    Bao gồm metadata, trạng thái model và danh sách vectors (visual hoặc dino).
    """

    model: NonEmptyString
    revision: str | None = None
    dimension: int = Field(gt=0)
    normalized: bool
    item_ids: list[NonEmptyString] | None = None
    embeddings: list[list[float]]
    latency_ms: float = Field(ge=0)

    @field_validator("embeddings")
    @classmethod
    def validate_shape(cls, values: list[list[float]]) -> list[list[float]]:
        if not values or not values[0]:
            raise ValueError("embeddings must be a non-empty matrix")
        if any(len(row) != len(values[0]) for row in values):
            raise ValueError("embedding rows must have equal dimensions")
        return values

    @model_validator(mode="after")
    def validate_metadata(self) -> EmbeddingResponse:
        if len(self.embeddings[0]) != self.dimension:
            raise ValueError("embedding dimension metadata does not match vectors")
        if self.item_ids is not None:
            if len(self.item_ids) != len(self.embeddings):
                raise ValueError("item/embedding count mismatch")
            if len(set(self.item_ids)) != len(self.item_ids):
                raise ValueError("embedding item_ids must be unique")
        return self


# Compatibility name retained for existing online text-embedding consumers.
TextEmbeddingResponse = EmbeddingResponse


class BoundaryScoreResponse(ContractModel):
    """Schema trả về kết quả chấm điểm biên (Boundary) cho tính năng Shot/Event Detection.
    Bao gồm danh sách điểm phân đoạn tương ứng với các khung hình (frames) đầu vào.
    """

    request_id: NonEmptyString
    model: NonEmptyString
    revision: str | None = None
    scores: list[float] = Field(min_length=1)
    latency_ms: float = Field(ge=0)


class AudioReferenceRequest(ContractModel):
    """Payload gửi yêu cầu xử lý âm thanh tới các worker thông qua một S3 Presigned URL.
    Cho phép worker tải trực tiếp file audio qua HTTPS để chạy ASR/Diarization.
    """

    request_id: NonEmptyString
    video_id: NonEmptyString
    audio_url: NonEmptyString
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)

    @field_validator("audio_url")
    @classmethod
    def require_https_audio(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("audio_url must use HTTPS")
        return value


class DiarizationRequest(AudioReferenceRequest):
    """Payload yêu cầu chạy Diarization (tách người nói) trên một đoạn hội thoại đã có chữ.
    Bao gồm URL audio và danh sách các đoạn text (segments) để phân bổ người nói.
    """

    segments: list[TranscriptSegment]


class TranscriptInferenceResponse(ContractModel):
    """Kết quả trả về chung cho tác vụ ASR và Diarization từ Inference Worker.
    Đảm bảo danh sách segments trả về phải được sắp xếp theo đúng thứ tự thời gian.
    """

    request_id: NonEmptyString
    video_id: NonEmptyString
    model: NonEmptyString
    revision: str | None = None
    segments: list[TranscriptSegment]
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_segments(self) -> TranscriptInferenceResponse:
        previous_start = previous_end = -1
        for index, segment in enumerate(self.segments):
            if segment.video_id != self.video_id or segment.segment_index != index:
                raise ValueError("transcript identity/order mismatch")
            if (
                segment.start_ms < previous_start
                or segment.end_ms < previous_end
            ):
                raise ValueError("transcript segments must be chronological")
            previous_start, previous_end = segment.start_ms, segment.end_ms
        return self


class CaptionItem(ContractModel):
    """One caller-owned image and its generated caption."""

    item_id: NonEmptyString
    caption: NonEmptyString


class CaptionResponse(ContractModel):
    """Ordered captions returned by the hosted vision-language model."""

    model: NonEmptyString
    revision: NonEmptyString
    items: list[CaptionItem]
    latency_ms: float = Field(ge=0)


class OCRRegionItem(ContractModel):
    """One OCR region with normalized axis-aligned image coordinates."""

    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_box(self) -> OCRRegionItem:
        """Reject inverted region boxes at the hosted inference boundary."""
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("OCR region maximums must not precede minimums")
        return self


class OCRItem(ContractModel):
    """One caller-owned image and its extracted OCR text."""

    item_id: NonEmptyString
    text: str
    raw_output: JsonValue | None = None
    regions: list[OCRRegionItem] = Field(default_factory=list)


class OCRResponse(ContractModel):
    """Ordered OCR results returned by the hosted vision model."""

    model: NonEmptyString
    revision: str | None = None
    items: list[OCRItem]
    latency_ms: float = Field(ge=0)


class RerankItem(ContractModel):
    """One caller-owned item and its model relevance score."""

    item_id: NonEmptyString
    score: float


class RerankResponse(ContractModel):
    """Ordered multimodal scores returned by the remote reranker."""

    model: NonEmptyString
    revision: str | None = None
    items: list[RerankItem]
    latency_ms: float = Field(ge=0)


class ModelStatus(ContractModel):
    """Readiness and provenance for one hosted model."""

    enabled: bool = True
    loaded: bool
    checkpoint: str | None = None
    revision: str | None = None


class InferenceCapabilities(ContractModel):
    """Feature-level readiness advertised by one inference deployment."""

    embedding: bool = False
    reranking: bool = False
    structured_parsing: bool = False
    shot_detection: bool = False
    event_detection: bool = False
    dino_embedding: bool = False
    image_embedding: bool = False
    caption: bool = False
    ocr: bool = False
    asr: bool = False
    diarization: bool = False


class InferenceReadiness(ContractModel):
    """Readiness snapshot for all configured inference capabilities."""

    ready: bool
    models: dict[str, ModelStatus]
    capabilities: InferenceCapabilities = Field(
        default_factory=InferenceCapabilities
    )
