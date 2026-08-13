"""Multilingual speech-to-text for video files."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hcmai.common.config import ASRConfig
from hcmai.common.schemas import TranscriptSegment


def _clean_text(text: str) -> str:
    """Normalize Unicode and whitespace without changing words."""

    return " ".join(unicodedata.normalize("NFC", text).split())


def _language_label(language: str | None) -> str:
    """Normalize the language label returned by Qwen."""

    return _clean_text(language).lower() if language else "und"


@dataclass(frozen=True)
class DecodedAudio:
    """Immutable mono waveform anchored to its canonical media timeline."""

    samples: np.ndarray
    sample_rate: int
    start_ms: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32).reshape(-1).copy()
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.start_ms < 0:
            raise ValueError("audio start_ms must be non-negative")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)


def _time_ms(pts: object, time_base: object) -> int | None:
    """Convert one valid PTS/time-base pair to non-negative media time."""

    if pts is None or time_base is None:
        return None
    try:
        value = round(float(pts * time_base) * 1_000)  # type: ignore[operator]
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value >= 0 else None


def read_audio(path: Path, sample_rate: int) -> DecodedAudio:
    """Decode mono audio and retain the first valid media-time coordinate."""

    import av  # pyright: ignore[reportMissingImports]

    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return DecodedAudio(np.empty(0, dtype=np.float32), sample_rate, 0)
        stream = container.streams.audio[0]
        fallback_ms = _time_ms(
            getattr(stream, "start_time", None),
            getattr(stream, "time_base", None),
        )
        start_ms: int | None = None
        resampler = av.AudioResampler(
            format="fltp", layout="mono", rate=sample_rate
        )
        for frame in container.decode(stream):
            if start_ms is None:
                start_ms = _time_ms(
                    getattr(frame, "pts", None),
                    getattr(frame, "time_base", None)
                    or getattr(stream, "time_base", None),
                )
            chunks.extend(
                item.to_ndarray().reshape(-1)
                for item in resampler.resample(frame)
            )
        chunks.extend(
            item.to_ndarray().reshape(-1)
            for item in resampler.resample(None)
        )
    if not chunks:
        return DecodedAudio(
            np.empty(0, dtype=np.float32), sample_rate, fallback_ms or 0
        )
    if start_ms is None:
        if fallback_ms is None:
            raise ValueError("audio has no valid frame PTS or stream start time")
        start_ms = fallback_ms
    return DecodedAudio(np.concatenate(chunks), sample_rate, start_ms)


class ASRAdapter:
    """Transcribe speech regions with lazily loaded Qwen and Silero models."""

    def __init__(
        self,
        config: ASRConfig,
        model: Any | None = None,
        processor: Any | None = None,
        vad_model: Any | None = None,
    ) -> None:
        """Store configuration and optional preloaded test models."""

        self.config = config
        self._model = model
        self._processor = processor
        self._vad_model = vad_model

    def _load_asr(self) -> tuple[Any, Any]:
        """Load the configured Qwen model and processor once."""

        if self._model is None or self._processor is None:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                revision=self.config.revision,
            )
            model_options = {"dtype": getattr(torch, self.config.dtype)}
            if self.config.attn_implementation:
                model_options["attn_implementation"] = (
                    self.config.attn_implementation
                )
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.config.model_name,
                revision=self.config.revision,
                **model_options,
            ).to(self.config.device).eval()
            if self.config.compile_model:
                self._model.forward = torch.compile(self._model.forward)
        return self._model, self._processor

    @property
    def resolved_revision(self) -> str:
        """Return the immutable configured or backend-confirmed ASR revision."""

        config = getattr(self._model, "config", None)
        return str(getattr(config, "_commit_hash", None) or self.config.revision)

    def _load_vad(self) -> Any:
        """Load the Silero VAD model once."""

        if self._vad_model is None:
            from silero_vad import load_silero_vad  # pyright: ignore[reportMissingImports]

            self._vad_model = load_silero_vad()
        return self._vad_model

    def _speech_regions(
        self, waveform: np.ndarray
    ) -> list[dict[str, int]]:
        """Return speech sample boundaries from Silero VAD."""

        import torch
        from silero_vad import get_speech_timestamps  # pyright: ignore[reportMissingImports]

        return get_speech_timestamps(
            torch.from_numpy(waveform),
            self._load_vad(),
            sampling_rate=self.config.audio_sample_rate,
            threshold=self.config.vad_threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            max_speech_duration_s=self.config.max_segment_seconds,
        )

    def _infer_batch(
        self,
        clips: list[np.ndarray],
    ) -> list[dict[str, str | None]]:
        """Run model on one batch of speech waveforms."""

        import torch

        model, processor = self._load_asr()
        language = (
            [self.config.language] * len(clips)
            if self.config.language else None
        )
        prompt = (
            [self.config.prompt] * len(clips)
            if self.config.prompt else None
        )
        inputs = processor.apply_transcription_request(
            audio=clips,
            language=language,
            prompt=prompt,
            sampling_rate=self.config.audio_sample_rate,
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        results = processor.decode(
            generated, return_format="parsed"
        )
        if len(results) != len(clips):
            raise RuntimeError("Model returned an incomplete ASR batch")
        return results

    def transcribe(
        self, video_path: str | Path, video_id: str
    ) -> list[TranscriptSegment]:
        """Return normalized transcript segments for one video."""

        decoded = read_audio(
            Path(video_path), self.config.audio_sample_rate
        )
        audio = decoded.samples
        regions = self._speech_regions(audio) if audio.size else []
        records = []
        for offset in range(0, len(regions), self.config.batch_size):
            batch = regions[offset:offset + self.config.batch_size]
            clips = [
                audio[int(region["start"]):int(region["end"])]
                for region in batch
            ]
            for region, result in zip(batch, self._infer_batch(clips)):
                text = _clean_text(str(result.get("transcription") or ""))
                if not text:
                    continue
                start, end = int(region["start"]), int(region["end"])
                index = len(records)
                language = self.config.language or result.get("language")
                records.append(TranscriptSegment(
                    segment_id=f"{video_id}_segment_{index:06d}",
                    video_id=video_id,
                    segment_index=index,
                    start_ms=decoded.start_ms + round(
                        start * 1000 / self.config.audio_sample_rate
                    ),
                    end_ms=decoded.start_ms + round(
                        end * 1000 / self.config.audio_sample_rate
                    ),
                    text=text,
                    language=_language_label(language),
                ))
        _validate_segments(records)
        return records


def _validate_segments(records: list[TranscriptSegment]) -> None:
    """Reject negative, duplicate, or non-monotonic final media intervals."""

    previous_start = -1
    previous_end = -1
    for record in records:
        if record.start_ms < 0 or record.end_ms <= record.start_ms:
            raise ValueError("transcript intervals must be positive media time")
        if record.start_ms < previous_start or record.end_ms < previous_end:
            raise ValueError("transcript intervals must be monotonic")
        previous_start, previous_end = record.start_ms, record.end_ms
