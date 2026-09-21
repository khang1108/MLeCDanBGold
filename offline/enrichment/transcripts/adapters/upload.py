"""Remote ASR adapter that uploads locally extracted FLAC directly to the model API."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Protocol

from hcmai.common.config import ASRConfig
from offline.enrichment.inference_contracts import InferenceReadiness, TranscriptInferenceResponse
from offline.enrichment.transcripts.models import TranscriptSegment


class UploadTranscriptClient(Protocol):
    def readiness(self) -> InferenceReadiness: ...

    def transcribe_audio_file(
        self,
        audio_path: str | Path,
        *,
        video_id: str,
        sample_rate: int = 16_000,
    ) -> TranscriptInferenceResponse: ...


class RemoteUploadASRAdapter:
    """Extract compact mono FLAC locally, upload it, and retain timestamped ASR."""

    def __init__(self, client: UploadTranscriptClient, config: ASRConfig) -> None:
        self.client = client
        self.config = config
        self._resolved_revision: str | None = None

    @property
    def resolved_revision(self) -> str:
        if self._resolved_revision is None:
            status = self.client.readiness().models.get("asr")
            if status is None or not status.loaded:
                raise RuntimeError("remote asr model is not ready")
            if status.checkpoint != self.config.model_name or status.revision != self.config.revision:
                raise ValueError("remote asr model provenance mismatch")
            self._resolved_revision = self.config.revision
        return self._resolved_revision

    def transcribe(self, video_path: str | Path, video_id: str) -> list[TranscriptSegment]:
        source = Path(video_path)
        with tempfile.TemporaryDirectory(prefix=f"vbs-asr-{video_id}-") as directory:
            audio_path = Path(directory) / f"{video_id}.flac"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", str(source),
                    "-vn", "-ac", "1", "-ar", str(self.config.audio_sample_rate),
                    "-c:a", "flac", str(audio_path),
                ],
                check=True,
            )
            response = self.client.transcribe_audio_file(
                audio_path,
                video_id=video_id,
                sample_rate=self.config.audio_sample_rate,
            )
        if response.model != self.config.model_name or response.revision != self.resolved_revision:
            raise ValueError("remote transcript model provenance mismatch")
        if response.video_id != video_id:
            raise ValueError("remote transcript changed video identity")
        return [
            TranscriptSegment.model_validate(
                segment.model_dump(mode="python")
                | {
                    "model_name": response.model,
                    "model_revision": response.revision,
                    "artifact_version": "asr-segment-v1",
                }
            )
            for segment in response.segments
        ]
