from __future__ import annotations
from typing import Self
from pydantic import Field, field_validator, model_validator
from .common import HTTPContract, NonEmptyString

class TranscriptSegment(HTTPContract):
    segment_id: NonEmptyString
    video_id: NonEmptyString
    segment_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: NonEmptyString
    language: NonEmptyString
    speaker_id: NonEmptyString | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: str = "completed"
    model_name: NonEmptyString | None = None
    model_revision: NonEmptyString | None = None
    artifact_version: NonEmptyString = "asr-segment-v1"
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None
    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self

class AudioReferenceRequest(HTTPContract):
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
    segments: list[TranscriptSegment]

class TranscriptInferenceResponse(HTTPContract):
    request_id: NonEmptyString
    video_id: NonEmptyString
    model: NonEmptyString
    revision: str | None = None
    segments: list[TranscriptSegment]
    latency_ms: float = Field(ge=0)
    @model_validator(mode="after")
    def validate_segments(self) -> Self:
        previous_start = previous_end = -1
        for index, segment in enumerate(self.segments):
            if segment.video_id != self.video_id or segment.segment_index != index:
                raise ValueError("transcript identity/order mismatch")
            if segment.start_ms < previous_start or segment.end_ms < previous_end:
                raise ValueError("transcript segments must be chronological")
            previous_start, previous_end = segment.start_ms, segment.end_ms
        return self
