"""Speaker diarization for video files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hcmai.common.config import DiarizationConfig
from hcmai.common.schemas import TranscriptSegment
from hcmai.transcripts.adapters.asr import read_audio


def _speaker_id(
    segment: TranscriptSegment,
    turns: list[tuple[Any, str]],
) -> str | None:
    """Return the speaker with the largest temporal overlap."""

    matches = (
        (
            min(segment.end_ms, round(turn.end * 1000))
            - max(segment.start_ms, round(turn.start * 1000)),
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
                token=self._hf_token,
            )
            self._pipeline.to(torch.device(self.config.device))
        return self._pipeline

    def assign_speakers(
        self,
        video_path: str | Path,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        """Assign one dominant speaker to each transcript segment."""

        import torch

        if not segments:
            return segments
        audio = read_audio(
            Path(video_path), self.config.audio_sample_rate
        )
        if not audio.size:
            return segments
        output = self._load_pipeline()({
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": self.config.audio_sample_rate,
        })
        turns = list(output.exclusive_speaker_diarization)
        return [
            segment.model_copy(update={
                "speaker_id": _speaker_id(segment, turns)
            })
            for segment in segments
        ]
