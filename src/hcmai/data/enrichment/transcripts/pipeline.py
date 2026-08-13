"""Public service boundary for transcript preparation and access."""

from __future__ import annotations

from pathlib import Path

from hcmai.common.config import ASRConfig, DiarizationConfig, TranscriptJobConfig
from hcmai.data.enrichment.transcripts.adapters.asr import ASRAdapter
from hcmai.data.enrichment.transcripts.adapters.diarization import (
    DiarizationAdapter,
)
from hcmai.data.enrichment.transcripts.prepare import (
    TranscriptReport,
    prepare_transcripts,
)
from hcmai.data.enrichment.transcripts.store import TranscriptStore


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
        return cls(
            ASRAdapter(asr),
            DiarizationAdapter(diarization) if diarization.enabled else None,
        )

    @classmethod
    def from_job_config(cls, config: TranscriptJobConfig) -> "TranscriptService":
        """Build adapters from one reproducible transcript job contract."""

        return cls.from_configs(config.asr, config.diarization)

    def prepare(
        self,
        videos_root: str | Path,
        output_path: str | Path,
        *,
        resume: bool = True,
        limit: int | None = None,
        schema_version: str = "transcript-segment-v1",
        pipeline_version: str = "transcript-pipeline-v1",
    ) -> TranscriptReport:
        if self.asr is None:
            raise RuntimeError("ASR adapter is not configured")
        return prepare_transcripts(
            videos_root,
            output_path,
            self.asr,
            diarizer=self.diarization,
            resume=resume,
            limit=limit,
            schema_version=schema_version,
            pipeline_version=pipeline_version,
        )

    @staticmethod
    def load_store(metadata_path: str | Path) -> TranscriptStore:
        return TranscriptStore(metadata_path)
