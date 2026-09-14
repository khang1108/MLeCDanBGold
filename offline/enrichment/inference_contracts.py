"""External inference contracts consumed by offline enrichment workers."""

from __future__ import annotations

from typing import Self

from pydantic import Field, JsonValue, field_validator, model_validator

from offline.contracts import ContractModel, NonEmptyString
from offline.enrichment.transcripts.models import TranscriptSegment


class AudioReferenceRequest(ContractModel):
    """HTTPS audio reference sent to a remote transcript worker."""

    request_id: NonEmptyString
    video_id: NonEmptyString
    audio_url: NonEmptyString
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)

    @field_validator("audio_url")
    @classmethod
    def require_https_audio(cls, value: str) -> str:
        """Reject mutable or local audio reference schemes."""

        if not value.startswith("https://"):
            raise ValueError("audio_url must use HTTPS")
        return value


class DiarizationRequest(AudioReferenceRequest):
    """Audio reference plus ordered transcript segments for diarization."""

    segments: list[TranscriptSegment]


class TranscriptInferenceResponse(ContractModel):
    """Chronological ASR or diarization result from a remote worker."""

    request_id: NonEmptyString
    video_id: NonEmptyString
    model: NonEmptyString
    revision: str | None = None
    segments: list[TranscriptSegment]
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_segments(self) -> Self:
        """Require envelope identity and chronological segment order."""

        previous_start = previous_end = -1
        for index, segment in enumerate(self.segments):
            if segment.video_id != self.video_id or segment.segment_index != index:
                raise ValueError("transcript identity/order mismatch")
            if segment.start_ms < previous_start or segment.end_ms < previous_end:
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
    def validate_box(self) -> Self:
        """Reject inverted OCR boxes at the inference boundary."""

        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("OCR region maximums must not precede minimums")
        return self


class OCRItem(ContractModel):
    """One caller-owned image and its extracted OCR data."""

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


class _ReadinessModel(ContractModel):
    """Validated readiness and provenance for one hosted model."""

    enabled: bool = True
    loaded: bool
    checkpoint: str | None = None
    revision: str | None = None


class _ReadinessCapabilities(ContractModel):
    """Feature readiness advertised by one inference deployment."""

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
    """Readiness snapshot for configured inference capabilities."""

    ready: bool
    models: dict[str, _ReadinessModel]
    capabilities: _ReadinessCapabilities = Field(
        default_factory=_ReadinessCapabilities
    )


__all__ = [
    "AudioReferenceRequest",
    "CaptionItem",
    "CaptionResponse",
    "DiarizationRequest",
    "InferenceReadiness",
    "OCRItem",
    "OCRRegionItem",
    "OCRResponse",
    "TranscriptInferenceResponse",
]
