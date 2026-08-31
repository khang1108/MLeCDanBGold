"""Remote ASR and diarization adapters over scoped immutable audio references."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from hcmai.common.config import ASRConfig, DiarizationConfig
from offline.enrichment.inference_contracts import (
    AudioReferenceRequest,
    DiarizationRequest,
    InferenceReadiness,
    TranscriptInferenceResponse,
)
from offline.enrichment.transcripts.models import TranscriptSegment

ASR_SEGMENT_ARTIFACT_VERSION = "asr-segment-v1"


class AudioReferenceProvider(Protocol):
    """Giao thức định nghĩa cách cấp quyền truy cập file audio cho worker remote (VD: qua S3 URL)."""

    def reference(
        self, video_path: Path, video_id: str, sample_rate: int
    ) -> AudioReferenceRequest: ...


class TranscriptClient(Protocol):
    def readiness(self) -> InferenceReadiness: ...

    def transcribe_audio_reference(
        self, payload: AudioReferenceRequest
    ) -> TranscriptInferenceResponse: ...

    def diarize_audio_reference(
        self, payload: DiarizationRequest
    ) -> TranscriptInferenceResponse: ...


class RemoteASRAdapter:
    """Gửi đường dẫn audio tới remote worker để chạy ASR (VD: Whisper).
    Nhận về danh sách TranscriptSegment (đoạn thoại được bóc tách).
    """

    def __init__(
        self,
        client: TranscriptClient,
        config: ASRConfig,
        references: AudioReferenceProvider,
    ) -> None:
        self.client = client
        self.config = config
        self.references = references
        self._resolved_revision: str | None = None

    @property
    def resolved_revision(self) -> str:
        if self._resolved_revision is None:
            self._resolved_revision = _ready_revision(
                self.client, "asr", self.config.model_name, self.config.revision
            )
        return self._resolved_revision

    def transcribe(
        self, video_path: str | Path, video_id: str
    ) -> list[TranscriptSegment]:
        request = self.references.reference(
            Path(video_path), video_id, self.config.audio_sample_rate
        )
        response = self.client.transcribe_audio_reference(request)
        _validate_response(
            response, self.config.model_name, self.resolved_revision, video_id
        )
        return _stamp_asr_lineage(
            response.segments,
            model_name=response.model,
            model_revision=self.resolved_revision,
        )


class RemoteDiarizationAdapter:
    """Gửi đường dẫn audio và danh sách transcript hiện có tới remote worker.
    Worker sẽ phân tách người nói (Speaker Diarization) và gán nhãn cho từng segment.
    Đảm bảo tính toàn vẹn của transcript (không thay đổi text, chỉ thêm metadata speaker).
    """

    def __init__(
        self,
        client: TranscriptClient,
        config: DiarizationConfig,
        references: AudioReferenceProvider,
    ) -> None:
        self.client = client
        self.config = config
        self.references = references
        self._resolved_revision: str | None = None

    @property
    def resolved_revision(self) -> str:
        if self._resolved_revision is None:
            self._resolved_revision = _ready_revision(
                self.client,
                "diarization",
                self.config.model_name,
                self.config.revision,
            )
        return self._resolved_revision

    def assign_speakers_audio(
        self,
        audio: object,
        segments: list[TranscriptSegment],
        *,
        video_path: str | Path | None = None,
    ) -> list[TranscriptSegment]:
        """Satisfy the assign_speakers_audio capability check.

        The remote adapter operates over audio references (e.g. S3 URLs) and
        cannot consume a locally decoded waveform directly.  When the caller
        supplies a ``video_path`` we use it; otherwise we fall back to
        reconstructing the path from the first segment's ``video_id``.  In
        both cases we delegate to the reference-based remote endpoint.
        """
        if not segments:
            return segments
        video_id = segments[0].video_id
        if any(segment.video_id != video_id for segment in segments):
            raise ValueError("diarization segments must belong to one video")
        # Derive a usable path: prefer an explicit argument, otherwise use
        # video_id as the path stem so the reference provider can resolve it.
        resolved_path = Path(video_path) if video_path is not None else Path(video_id)
        ref = self.references.reference(
            resolved_path, video_id, self.config.audio_sample_rate
        )
        request = DiarizationRequest(**ref.model_dump(mode="python"), segments=segments)
        response = self.client.diarize_audio_reference(request)
        _validate_response(
            response, self.config.model_name, self.resolved_revision, video_id
        )
        if [item.segment_id for item in response.segments] != [
            item.segment_id for item in segments
        ]:
            raise ValueError("remote diarization changed segment identity/order")
        return _apply_speaker_labels(segments, response.segments)

    def assign_speakers(
        self,
        video_path: str | Path,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        return self.assign_speakers_audio(None, segments, video_path=video_path)


def _ready_revision(
    client: TranscriptClient,
    capability: str,
    model_name: str,
    revision: str,
) -> str:
    status = client.readiness().models.get(capability)
    if status is None or not status.loaded:
        raise RuntimeError(f"remote {capability} model is not ready")
    if status.checkpoint != model_name or status.revision != revision:
        raise ValueError(f"remote {capability} model provenance mismatch")
    return revision


def _validate_response(
    response: TranscriptInferenceResponse,
    model_name: str,
    revision: str,
    video_id: str,
) -> None:
    if response.model != model_name or response.revision != revision:
        raise ValueError("remote transcript model provenance mismatch")
    if response.video_id != video_id:
        raise ValueError("remote transcript changed video identity")


def _stamp_asr_lineage(
    segments: list[TranscriptSegment],
    *,
    model_name: str,
    model_revision: str,
) -> list[TranscriptSegment]:
    """Fill missing ASR lineage and reject segment/envelope contradictions."""

    stamped: list[TranscriptSegment] = []
    for segment in segments:
        if segment.model_name is not None and segment.model_name != model_name:
            raise ValueError("remote segment model_name conflicts with envelope")
        if (
            segment.model_revision is not None
            and segment.model_revision != model_revision
        ):
            raise ValueError(
                "remote segment model_revision conflicts with envelope"
            )
        if segment.artifact_version != ASR_SEGMENT_ARTIFACT_VERSION:
            raise ValueError(
                "remote segment artifact_version conflicts with ASR contract"
            )
        stamped.append(segment.model_copy(update={
            "model_name": model_name,
            "model_revision": model_revision,
            "artifact_version": ASR_SEGMENT_ARTIFACT_VERSION,
        }))
    return stamped


def _apply_speaker_labels(
    original: list[TranscriptSegment],
    diarized: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Accept only speaker labels while preserving canonical ASR segments."""

    merged: list[TranscriptSegment] = []
    for source, result in zip(original, diarized, strict=True):
        source_fields = source.model_dump(exclude={"speaker_id"})
        result_fields = result.model_dump(exclude={"speaker_id"})
        if result_fields != source_fields:
            raise ValueError("remote diarization changed canonical ASR segment")
        merged.append(
            source.model_copy(update={"speaker_id": result.speaker_id})
        )
    return merged
