"""Public service boundary for transcript preparation and access."""

from __future__ import annotations

from pathlib import Path

from hcmai.common.config import ASRConfig, DiarizationConfig
from hcmai.transcripts.adapters.asr import ASRAdapter
from hcmai.transcripts.adapters.diarization import DiarizationAdapter
from hcmai.transcripts.prepare import TranscriptReport, prepare_transcripts
from hcmai.transcripts.store import TranscriptStore


class TranscriptService:
    """Own ASR/diarization adapters and canonical transcript artifacts."""

    def __init__(
        self,
        asr: ASRAdapter | None = None,
        diarization: DiarizationAdapter | None = None,
    ) -> None:
        self.asr = asr
        self.diarization = diarization

    @classmethod
    def from_configs(
        cls,
        asr: ASRConfig,
        diarization: DiarizationConfig,
    ) -> "TranscriptService":
        return cls(ASRAdapter(asr), DiarizationAdapter(diarization))

    def prepare(
        self,
        videos_root: str | Path,
        output_path: str | Path,
        *,
        resume: bool = True,
        limit: int | None = None,
    ) -> TranscriptReport:
        if self.asr is None or self.diarization is None:
            raise RuntimeError("ASR and diarization adapters are not configured")
        return prepare_transcripts(
            videos_root,
            output_path,
            self.asr,
            diarizer=self.diarization,
            resume=resume,
            limit=limit,
        )

    @staticmethod
    def load_store(metadata_path: str | Path) -> TranscriptStore:
        return TranscriptStore(metadata_path)
