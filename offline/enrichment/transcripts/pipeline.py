"""Pipeline xử lý Transcript (Lời thoại).

Điều phối luồng công việc trích xuất lời thoại từ âm thanh video thành văn bản có timestamp.

Các tính năng chính:
1. Tiền xử lý âm thanh: Trích xuất audio từ video và tối ưu hóa định dạng (sample rate, mono/stereo).
2. Speech-to-Text (ASR): Dùng mô hình nhận diện giọng nói để sinh văn bản thô theo thời gian.
3. Diarization: Phân biệt người nói (Speaker A, Speaker B) cho từng phân đoạn âm thanh.
4. Đồng bộ (Alignment): Khớp text với timestamp để map chính xác vào đúng khung hình video."""

from __future__ import annotations

from pathlib import Path

from hcmai.common.config import ASRConfig, DiarizationConfig, TranscriptJobConfig
from offline.enrichment.transcripts.adapters.asr import ASRAdapter
from offline.enrichment.transcripts.adapters.diarization import (
    DiarizationAdapter,
)
from offline.enrichment.transcripts.adapters.remote import (
    RemoteASRAdapter,
    RemoteDiarizationAdapter,
)
from offline.enrichment.transcripts.prepare import (
    TranscriptReport,
    prepare_transcript_video,
    prepare_transcripts,
)
from hcmai.corpus.stores.transcript import TranscriptStore


class TranscriptService:
    """Own ASR/diarization adapters and canonical transcript artifacts."""

    def __init__(
        self,
        asr: ASRAdapter | RemoteASRAdapter | None = None,
        diarization: DiarizationAdapter | RemoteDiarizationAdapter | None = None,
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

    def prepare_video(
        self,
        video_path: str | Path,
        output_root: str | Path,
        *,
        resume: bool = True,
        schema_version: str = "transcript-segment-v1",
        pipeline_version: str = "transcript-pipeline-v1",
    ) -> tuple[Path, int]:
        """Consume one already-staged video without another source download."""

        if self.asr is None:
            raise RuntimeError("ASR adapter is not configured")
        return prepare_transcript_video(
            video_path,
            output_root,
            self.asr,
            diarizer=self.diarization,
            resume=resume,
            schema_version=schema_version,
            pipeline_version=pipeline_version,
        )

    @staticmethod
    def load_store(metadata_path: str | Path) -> TranscriptStore:
        return TranscriptStore(metadata_path)
