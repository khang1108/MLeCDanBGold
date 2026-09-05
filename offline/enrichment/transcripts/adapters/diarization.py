"""Diarization adapter tagging transcript segments with a speaker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hcmai.common.config import DiarizationConfig
from offline.enrichment.transcripts.models import TranscriptSegment
from offline.enrichment.transcripts.adapters.asr import DecodedAudio, read_audio


def _speaker_id(
    segment: TranscriptSegment,
    turns: list[tuple[Any, str]],
    *,
    audio_start_ms: int = 0,
) -> str | None:
    """Return the speaker with the largest temporal overlap."""

    matches = (
        (
            min(segment.end_ms, audio_start_ms + round(turn.end * 1000))
            - max(segment.start_ms, audio_start_ms + round(turn.start * 1000)),
            str(speaker),
        )
        for turn, speaker in turns
    )
    overlap, speaker_id = max(matches, default=(0, None))
    return speaker_id if overlap > 0 else None


class DiarizationAdapter:
    """Assign speakers with one lazily loaded Pyannote pipeline."""

    def __init__(
        self,
        config: DiarizationConfig,
        pipeline: Any | None = None,
        hf_token: str | None = None,
    ) -> None:
        """Store configuration and an optional preloaded pipeline."""

        self.config = config
        self._pipeline = pipeline
        self._hf_token = hf_token or os.getenv("HF_TOKEN")

    def _load_pipeline(self) -> Any:
        """Load the configured Pyannote pipeline once."""

        if self._pipeline is None:
            import torch
            from pyannote.audio import Pipeline  # pyright: ignore[reportMissingImports]

            self._pipeline = Pipeline.from_pretrained(
                self.config.model_name,
                revision=self.config.revision,
                token=self._hf_token,
            )
            self._pipeline.to(torch.device(self.config.device))
        return self._pipeline

    @property
    def resolved_revision(self) -> str:
        """Return the immutable diarization revision used for this job."""

        return self.config.revision

    def assign_speakers_audio(
        self,
        audio: DecodedAudio,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        """Assign speakers using an already-decoded immutable waveform."""

        import torch

        if not segments or not audio.samples.size:
            return segments
        output = self._load_pipeline()({
            "waveform": torch.from_numpy(audio.samples.copy()).unsqueeze(0),
            "sample_rate": audio.sample_rate,
        })
        turns = list(output.exclusive_speaker_diarization)
        return [
            segment.model_copy(update={
                "speaker_id": _speaker_id(
                    segment, turns, audio_start_ms=audio.start_ms
                )
            })
            for segment in segments
        ]

    def assign_speakers(
        self,
        video_path: str | Path,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        """Decode audio and assign speakers through the compatibility API."""

        audio = read_audio(
            Path(video_path), self.config.audio_sample_rate
        )
        return self.assign_speakers_audio(audio, segments)
