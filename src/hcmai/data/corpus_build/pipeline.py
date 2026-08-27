"""Pipeline xây dựng Corpus (Tập dữ liệu).

Module này định nghĩa quy trình tạo ra bộ corpus chính thức từ các video và frame đã xử lý.

Các nhiệm vụ chính:
1. Thu thập metadata: Tổng hợp thông tin từ các frame đã được trích xuất (metadata, timestamp).
2. Liên kết Media: Đảm bảo các file ảnh và video gốc được ánh xạ đúng đường dẫn tuyệt đối.
3. Chuẩn bị Indexing: Tạo cấu trúc chuẩn để các công cụ tìm kiếm (retrieval) có thể ingest dễ dàng."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from tqdm import tqdm

logger = logging.getLogger(__name__)
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.io import atomic_write, read_json, write_json
from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
from hcmai.data.corpus_build.publish import S3Publication
from hcmai.data.s3 import (
    S3VideoObject,
    create_s3_client,
    list_video_objects,
    staged_video,
)

_PIPELINE_VERSION = "s3-corpus-preparation-v1"
_TEXT_SOURCES = (
    RetrievalSource.CAPTION,
    RetrievalSource.OCR,
    RetrievalSource.ASR,
)


@dataclass(frozen=True, slots=True)
class PreparationPaths:
    """Every mutable path owned by one isolated full or smoke run."""

    artifacts_root: Path
    state_root: Path
    frame_store_root: Path
    transcripts_root: Path
    asr_root: Path
    caption_root: Path
    ocr_root: Path
    object_root: Path
    context_root: Path
    visual_index_root: Path
    caption_index_root: Path
    ocr_index_root: Path
    asr_index_root: Path

    @classmethod
    def from_config(
        cls,
        config: S3CorpusPreparationConfig,
        limit: int | None,
        offset: int | None = None,
    ) -> PreparationPaths:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero")
        if offset is not None and offset < 0:
            raise ValueError("offset must be non-negative")
            
        suffix_parts = []
        if limit is not None:
            suffix_parts.append(f"limit-{limit}")
        if offset is not None:
            suffix_parts.append(f"offset-{offset}")

        if not suffix_parts:
            artifacts = config.artifacts_root
            frame_store = artifacts / "frame_store"
            state = config.work_root / ".preparation"
        else:
            suffix = "-".join(suffix_parts)
            artifacts = config.work_root / f"artifacts.{suffix}"
            frame_store = artifacts / "frame_store"
            state = config.work_root / ".preparation" / suffix

        return cls(
            artifacts_root=artifacts,
            state_root=state,
            frame_store_root=frame_store,
            transcripts_root=artifacts / "enrichment/transcripts",
            asr_root=artifacts / "enrichment/asr",
            caption_root=artifacts / "enrichment/caption",
            ocr_root=artifacts / "enrichment/ocr",
            object_root=artifacts / "enrichment/objects",
            context_root=artifacts / "enrichment/context",
            visual_index_root=artifacts / "indexes/visual",
            caption_index_root=artifacts / "indexes/caption",
            ocr_index_root=artifacts / "indexes/ocr",
            asr_index_root=artifacts / "indexes/asr",
        )

    @classmethod
    def for_group(
        cls,
        config: S3CorpusPreparationConfig,
        group_id: str,
    ) -> PreparationPaths:
        root = config.work_root / "groups" / group_id
        artifacts = root / "artifacts"
        return cls(
            artifacts_root=artifacts,
            state_root=root / ".preparation",
            frame_store_root=artifacts / "frame_store",
            transcripts_root=artifacts / "transcripts",
            asr_root=artifacts / "enrichment/asr",
            caption_root=artifacts / "enrichment/caption",
            ocr_root=artifacts / "enrichment/ocr",
            object_root=artifacts / "enrichment/objects",
            context_root=artifacts / "enrichment/context",
            visual_index_root=artifacts / "embeddings/visual",
            caption_index_root=artifacts / "embeddings/caption",
            ocr_index_root=artifacts / "embeddings/ocr",
            asr_index_root=artifacts / "embeddings/asr",
        )

    @classmethod
    def from_enrichment_job(
        cls, config: S3CorpusPreparationConfig, job: Any
    ) -> "PreparationPaths":
        """Align BTC-native orchestration paths with the enrichment contract."""

        artifacts = job.frame_store_output.parent
        enrichment = artifacts / "enrichment"
        return cls(
            artifacts_root=artifacts,
            state_root=config.work_root / ".preparation",
            frame_store_root=job.frame_store_output,
            transcripts_root=job.transcript_output_dir,
            asr_root=enrichment / "asr",
            caption_root=job.caption_output_dir,
            ocr_root=job.ocr_output_dir,
            object_root=job.object_output_dir,
            context_root=job.context_output_dir,
            visual_index_root=artifacts / "indexes/visual",
            caption_index_root=artifacts / "indexes/caption",
            ocr_index_root=artifacts / "indexes/ocr",
            asr_index_root=artifacts / "indexes/asr",
        )

    @property
    def frames_path(self) -> Path:
        return self.frame_store_root / "frames.parquet"

    @property
    def asr_enrichment_path(self) -> Path:
        return self.asr_root / "frame_enrichment.parquet"

    @property
    def visual_embeddings_path(self) -> Path:
        return self.artifacts_root / "embeddings/visual_embeddings.npy"

    @property
    def visual_mapping_path(self) -> Path:
        return self.artifacts_root / "embeddings/frame_mapping.parquet"

    def enrichment_path(self, source: RetrievalSource) -> Path:
        roots = {
            RetrievalSource.CAPTION: self.caption_root,
            RetrievalSource.OCR: self.ocr_root,
            RetrievalSource.ASR: self.asr_root,
        }
        return roots[source] / "frame_enrichment.parquet"

    def index_root(self, source: RetrievalSource) -> Path:
        return {
            RetrievalSource.CAPTION: self.caption_index_root,
            RetrievalSource.OCR: self.ocr_index_root,
            RetrievalSource.ASR: self.asr_index_root,
        }[source]

    def text_embeddings_path(self, source: RetrievalSource) -> Path:
        return self.index_root(source) / f"{source.value}_embeddings.npy"

    def text_mapping_path(self, source: RetrievalSource) -> Path:
        return self.index_root(source) / "frame_mapping.parquet"


@dataclass(frozen=True, slots=True)
class PreparationRun:
    """Compact result returned by the service and printed by the CLI."""

    run_id: str
    inventory_path: Path
    artifacts_root: Path
    source_count: int
    completed_stages: tuple[str, ...]
    skipped_stages: tuple[str, ...]
    publication: S3Publication | None = None


@dataclass(frozen=True, slots=True)
class PreparationCacheRun:
    """Result of materializing an immutable S3 inventory into local cache."""

    run_id: str
    inventory_path: Path
    cache_root: Path
    source_count: int
    downloaded_count: int
    reused_count: int
    total_bytes: int
    duration_seconds: float



class PreparationOperations(Protocol):
    """Existing stage services adapted to the shared S3 source lifecycle."""

    def prepare_btc_frame_store(self) -> Path: ...

    def prepare_transcript(self, video: Path) -> Path: ...

    def materialize_asr(self) -> Path: ...

    def generate_caption(self) -> Path: ...

    def generate_ocr(self) -> Path: ...

    def detect_objects(self) -> Path: ...

    def build_frame_context(self) -> Path: ...

    def build_visual_index(self) -> Path: ...

    def build_text_index(self, source: RetrievalSource) -> Path: ...

    def build_visual_artifacts(self) -> tuple[Path, Path]: ...

    def build_text_embeddings(
        self, source: RetrievalSource
    ) -> tuple[Path, Path]: ...


class DefaultPreparationOperations:
    """Production adapter over the repository's current offline services."""

    def __init__(
        self,
        config: S3CorpusPreparationConfig,
        paths: PreparationPaths,
        *,
        resume: bool,
        limit: int | None,
        enrichment_config: str | Path = "configs/prepare.yaml",
        model_config: str | Path = "thundercompute/config.yaml",
        retrieval_config: str | Path = "configs/baseline.yaml",
        dataset: Mapping[str, Any] | None = None,
        s3_client: Any | None = None,
    ) -> None:
        from hcmai.common.config import TranscriptJobConfig
        from hcmai.data.enrichment.caption.config import CaptionJobConfig
        from hcmai.data.enrichment.pipeline import EnrichmentJobConfig
        from thundercompute.config import LLMServiceConfig

        storage = config.preprocessing.s3
        if storage is None:
            raise ValueError("Default preparation operations require S3 storage")
        
        self.config = config

        self.storage = storage
        self.paths = paths
        self.resume = resume
        self.limit = limit

        self.enrichment_config = Path(enrichment_config)
        self.model_config_path = Path(model_config)
        self.retrieval_config = Path(retrieval_config)
        self.dataset = dict(dataset) if dataset is not None else None
        self.s3_client = s3_client
        self.enrichment_job = EnrichmentJobConfig.from_yaml(
            self.enrichment_config,
            dataset=self.dataset,
        )
        self.caption_job = CaptionJobConfig.from_yaml(
            self.enrichment_config,
            dataset=self.dataset,
        )
        self.transcript_job = TranscriptJobConfig.from_yaml(
            self.enrichment_config,
            dataset=self.dataset,
        )
        self.model_config = LLMServiceConfig.from_yaml(self.model_config_path)
        
        self._validate_model_pins()
        self._transcripts: Any | None = None
        self._text_encoder: Any | None = None
        self._remote_pools: dict[str, Any] = {}
        self._audio_references: Any | None = None

    def _remote_pool(self, capability: str) -> Any | None:
        pool_config = getattr(self.config.remote_inference, capability)
        if pool_config is None:
            return None
        if capability not in self._remote_pools:
            from thundercompute.adapters.pool import InferenceClientPool

            self._remote_pools[capability] = InferenceClientPool.from_config(
                pool_config
            )
        return self._remote_pools[capability]

    def _validate_model_pins(self) -> None:
        """Kiểm tra tính nhất quán giữa các phiên bản mô hình (Model Pins)."""
        pins = self.config.models

        # Tạo tuple (actual_name, actual_revision, pin_name, pin_revision)
        # Đây là cách để so sánh hai phiên bản: Tên và Version đều phải khớp.
        pairs = {
            "caption": (
                self.caption_job.caption.model_checkpoint,
                self.caption_job.caption.revision,
                pins.caption.model_name,
                pins.caption.revision,
            ),
            "asr": (
                self.transcript_job.asr.model_name,
                self.transcript_job.asr.revision,
                pins.asr.model_name,
                pins.asr.revision,
            ),
            "diarization": (
                self.transcript_job.diarization.model_name,
                self.transcript_job.diarization.revision,
                pins.diarization.model_name,
                pins.diarization.revision,
            ),
            "visual_embedding": (
                self.model_config.visual_embedding.model_name,
                self.model_config.visual_embedding.revision,
                pins.visual_embedding.model_name,
                pins.visual_embedding.revision,
            ),
            "text_embedding": (
                self.model_config.caption_embedding.model_name,
                self.model_config.caption_embedding.revision,
                pins.text_embedding.model_name,
                pins.text_embedding.revision,
            ),
        }

        # Tìm các stage có version không khớp
        mismatched = [
            name
            for name, (actual_name, actual_revision, name_pin, revision_pin)
            in pairs.items()
            if (actual_name, actual_revision) != (name_pin, revision_pin)
        ]

        # Báo lỗi nếu phát hiện sai khác phiên bản
        if mismatched:
            raise ValueError(
                "Preparation model pins differ from stage configuration: "
                + ", ".join(mismatched)
            )

    def _transcript_service(self) -> Any:
        """Khởi tạo TranscriptService (Xử lý phụ đề/phân đoạn) nếu chưa có."""
        if self._transcripts is None:
            from hcmai.data.enrichment.transcripts.pipeline import TranscriptService

            pool = self._remote_pool("transcript")
            if pool is None:
                self._transcripts = TranscriptService.from_job_config(
                    self.transcript_job
                )
            else:
                from hcmai.data.corpus_build.audio import S3AudioReferenceProvider
                from hcmai.data.enrichment.transcripts.adapters.remote import (
                    RemoteASRAdapter,
                    RemoteDiarizationAdapter,
                )

                if self.s3_client is None:
                    raise RuntimeError("remote transcripts require an S3 client")
                
                run_id = getattr(self, "_current_run_id", "unbound")
                self._audio_references = S3AudioReferenceProvider(
                    self.s3_client,
                    bucket=self.storage.bucket,
                    prefix=f"{self.storage.artifacts_prefix}/runs/{run_id}",
                    work_root=self.paths.state_root,
                )
                
                asr = RemoteASRAdapter(
                    pool, self.transcript_job.asr, self._audio_references
                )
                diarization = (
                    RemoteDiarizationAdapter(
                        pool,
                        self.transcript_job.diarization,
                        self._audio_references,
                    )
                    if self.transcript_job.diarization.enabled
                    else None
                )
                
                self._transcripts = TranscriptService(asr, diarization)
        return self._transcripts

    def prepare_btc_frame_store(self) -> Path:
        """Import organizer keyframes without constructing a video preprocessor."""

        from hcmai.data.pipeline import DataService

        frames_path = DataService.prepare(
            self.enrichment_config,
            dataset=self.dataset,
        )
        if frames_path != self.paths.frames_path:
            raise ValueError(
                "BTC frame store path differs from the active enrichment contract"
            )
        return frames_path

    def prepare_transcript(self, video: Path) -> Path:
        output, _ = self._transcript_service().prepare_video(
            video,
            self.paths.transcripts_root,
            resume=self.resume,
            schema_version=self.transcript_job.schema_version,
            pipeline_version=self.transcript_job.pipeline_version,
        )
        return output

    def materialize_asr(self) -> Path:
        from hcmai.data.enrichment.transcripts.materialize import (
            materialize_transcript_artifact,
        )

        return materialize_transcript_artifact(
            self.paths.frames_path,
            self.paths.transcripts_root,
            self.paths.asr_enrichment_path,
            window_ms=self.transcript_job.frame_evidence_window_ms,
            enrichment_version=self.transcript_job.enrichment_version,
            model_name=(
                f"{self.transcript_job.asr.model_name}@"
                f"{self.transcript_job.asr.revision}:"
                f"{self.transcript_job.pipeline_version}"
            ),
            frame_store_id=self._specialist_frame_store_id(),
        )

    def _specialist_frame_store_id(self) -> str | None:
        """Keep BTC evidence tied to its canonical frame store, not an S3 run."""

        return self.enrichment_job.frame_store_id

    def _runtime_caption_config(self) -> Any:
        """Return Caption policy with only the active dataset version replaced."""

        return replace(
            self.caption_job.caption,
            dataset_version=self.config.corpus_revision,
        )

    def _runtime_ocr_config(self) -> Any:
        """Preserve configured OCR policy while applying runtime pins/placement."""

        pin = self.config.models.ocr
        return replace(
            self.enrichment_job.ocr,
            checkpoint=pin.model_name,
            revision=pin.revision,
            dataset_version=self.config.corpus_revision,
        )

    def _runtime_object_config(self) -> Any:
        """Return the configured YOLOE policy for this preparation run."""

        return self.enrichment_job.objects

    def stage_dependency_identity(self, stage: str) -> dict[str, Any]:
        """Return policy dependencies plus expected stable manifest identity.

        Context includes all three specialist policies because unchanged
        artifact-version labels do not prove unchanged derived content.
        """

        lineage = self._specialist_frame_store_id()
        caption = self._runtime_caption_config()
        ocr = self._runtime_ocr_config()
        objects = self._runtime_object_config()
        context = self.enrichment_job.context
        common = {
            "frames_path": str(self.paths.frames_path),
            "frame_store_id": lineage,
        }
        caption_config = _identity_value(asdict(caption))
        ocr_config = _identity_value(asdict(ocr))
        object_config = _identity_value(objects.as_dict())
        context_config = _identity_value(asdict(context))

        if stage == "caption":
            return {
                "dependencies": {**common, "configuration": caption_config},
                "manifest": {
                    "artifact_version": caption.enrichment_version,
                    "enrichment_version": caption.enrichment_version,
                    "dataset_version": caption.dataset_version,
                    "frame_store_id": lineage,
                    "model_checkpoint": caption.model_checkpoint,
                    "prompt": caption.prompt,
                    "decoding": caption.decoding,
                    "device": caption.device,
                    "precision": caption.precision,
                    "dtype": caption.dtype,
                    "image_size": caption.image_size,
                    "batch_size": caption.batch_size,
                    "effective_configuration": caption_config,
                },
            }
        if stage == "ocr":
            return {
                "dependencies": {**common, "configuration": ocr_config},
                "manifest": {
                    "artifact_version": ocr.artifact_version,
                    "enrichment_version": ocr.enrichment_version,
                    "dataset_version": ocr.dataset_version,
                    "frame_store_id": lineage,
                    "backend": ocr.backend,
                    "checkpoint": ocr.checkpoint,
                    "device": ocr.device,
                    "dtype": ocr.dtype,
                    "batch_size": ocr.batch_size,
                    "runtime_settings": ocr_config,
                },
            }
        if stage == "objects":
            return {
                "dependencies": {
                    **common,
                    "configuration": object_config,
                },
                "manifest": {
                    "artifact_version": objects.artifact_version,
                    "frame_store_id": lineage,
                    "source": "yoloe",
                    **object_config,
                },
            }
        if stage == "frame_context":
            serializer = {
                "caption_token_budget": context.caption_token_budget,
                "ocr_token_budget": context.ocr_token_budget,
                "object_token_budget": context.object_token_budget,
                "min_ocr_quality": context.min_ocr_quality,
            }
            return {
                "dependencies": {
                    **common,
                    "configuration": context_config,
                    "caption_configuration": caption_config,
                    "ocr_configuration": ocr_config,
                    "object_configuration": object_config,
                },
                "manifest": {
                    "context_version": context.context_version,
                    "caption_version": caption.enrichment_version,
                    "ocr_version": ocr.artifact_version,
                    "object_version": objects.artifact_version,
                    "frame_store_id": lineage,
                    "serializer_config": serializer,
                },
            }
        return {}

    def generate_caption(self) -> Path:
        from hcmai.data.enrichment.pipeline import EnrichmentService

        caption = self._runtime_caption_config()
        pool = self._remote_pool("caption")
        if pool is None:
            adapter = EnrichmentService.create_caption_adapter(caption)
        else:
            from hcmai.data.enrichment.caption.adapters.remote import (
                RemoteCaptionAdapter,
            )

            adapter = RemoteCaptionAdapter(pool, caption)
        try:
            EnrichmentService.generate_captions(
                self.paths.frames_path,
                self.paths.caption_root,
                caption,
                adapter=adapter,
                dataset_root=self.paths.frame_store_root,
                frame_store_id=self._specialist_frame_store_id(),
            )
        finally:
            del adapter
        return self.paths.caption_root / "frame_enrichment.parquet"

    def generate_ocr(self) -> Path:
        from hcmai.data.enrichment.pipeline import EnrichmentService

        ocr = self._runtime_ocr_config()
        pool = self._remote_pool("ocr")
        if pool is None:
            adapter = EnrichmentService.create_ocr_adapter(ocr)
        else:
            from hcmai.data.enrichment.ocr.adapters.remote import RemoteOCRAdapter

            adapter = RemoteOCRAdapter(pool, replace(ocr, backend="remote"))
        try:
            EnrichmentService.generate_ocr(
                self.paths.frames_path,
                self.paths.ocr_root,
                ocr,
                adapter=adapter,
                dataset_root=self.paths.frame_store_root,
                frame_store_id=self._specialist_frame_store_id(),
            )
        finally:
            del adapter
        return self.paths.ocr_root / "frame_enrichment.parquet"

    def detect_objects(self) -> Path:
        """Run YOLOE and publish object evidence through the public facade."""

        from hcmai.data.enrichment.pipeline import EnrichmentService

        config = self._runtime_object_config()
        EnrichmentService.detect_objects(
            self.paths.frames_path,
            self.paths.object_root,
            config,
            dataset_root=self.enrichment_job.data_root,
            frame_store_id=self._specialist_frame_store_id(),
        )
        return self.paths.object_root / "frames.parquet"

    def build_frame_context(self) -> Path:
        """Build derived context only from already materialized specialists."""

        from hcmai.data.enrichment.pipeline import EnrichmentService

        return EnrichmentService.build_frame_context(
            self.paths.frames_path,
            self.paths.caption_root / "captions.parquet",
            self.paths.ocr_root / "frames.parquet",
            self.paths.object_root / "frames.parquet",
            self.paths.context_root,
            self.enrichment_job.context,
            frame_store_id=self._specialist_frame_store_id(),
        )

    def build_visual_index(self) -> Path:
        import numpy as np
        import pandas as pd

        from hcmai.retrieval.embedding.pipeline import EmbeddingService
        from hcmai.retrieval.retriever.pipeline import RetrievalService

        pool = self._remote_pool("visual_embedding")
        encoder = (
            EmbeddingService.create_remote_visual_adapter(
                pool, self.model_config.visual_embedding
            )
            if pool is not None
            else None
        )
        run = EmbeddingService.build_visual_artifacts(
            self.paths.frames_path,
            self.paths.frame_store_root,
            self.paths.artifacts_root,
            self.model_config.visual_embedding,
            self.config.corpus_revision,
            encoder=encoder,
        )
        if not run.generated_count:
            raise RuntimeError("No visual embeddings were generated")
        index = RetrievalService.build_index(
            np.load(run.embeddings_file, mmap_mode="r"),
            pd.read_parquet(run.mapping_file),
            dataset_version=self.config.corpus_revision,
            model_name=self.model_config.visual_embedding.model_name,
        )
        index.save(self.paths.visual_index_root)
        return self.paths.visual_index_root / RetrievalService.INDEX_FILENAME

    def build_visual_artifacts(self) -> tuple[Path, Path]:
        from hcmai.retrieval.embedding.pipeline import EmbeddingService

        pool = self._remote_pool("visual_embedding")
        encoder = (
            EmbeddingService.create_remote_visual_adapter(
                pool, self.model_config.visual_embedding
            )
            if pool is not None
            else None
        )
        run = EmbeddingService.build_visual_artifacts(
            self.paths.frames_path,
            self.paths.frame_store_root,
            self.paths.artifacts_root,
            self.model_config.visual_embedding,
            self.config.corpus_revision,
            encoder=encoder,
        )
        if not run.generated_count:
            raise RuntimeError("No visual embeddings were generated")
        return run.embeddings_file, run.mapping_file

    def build_text_index(self, source: RetrievalSource) -> Path:
        from hcmai.retrieval.embedding.pipeline import EmbeddingService
        from hcmai.retrieval.retriever.pipeline import RetrievalService

        if self._text_encoder is None:
            pool = self._remote_pool("text_embedding")
            self._text_encoder = (
                EmbeddingService.create_remote_adapter(
                    pool,
                    self.model_config.caption_embedding,
                    embedding_dim=0,
                    source="text",
                )
                if pool is not None
                else EmbeddingService.create_text_adapter(
                    self.model_config.caption_embedding
                )
            )
        RetrievalService.build_text_artifacts(
            self.retrieval_config,
            self.model_config_path,
            source=source,
            enrichment_path=self.paths.enrichment_path(source),
            frames_path=self.paths.frames_path,
            output_dir=self.paths.index_root(source),
            encoder=self._text_encoder,
        )
        return self.paths.index_root(source) / RetrievalService.INDEX_FILENAME

    def build_text_embeddings(
        self, source: RetrievalSource
    ) -> tuple[Path, Path]:
        from hcmai.common.config import AppConfig
        from hcmai.data.pipeline import DataService
        from hcmai.retrieval.embedding.pipeline import EmbeddingService
        from hcmai.retrieval.retriever.pipeline import RetrievalService

        if self._text_encoder is None:
            pool = self._remote_pool("text_embedding")
            self._text_encoder = (
                EmbeddingService.create_remote_adapter(
                    pool,
                    self.model_config.caption_embedding,
                    embedding_dim=0,
                    source="text",
                )
                if pool is not None
                else EmbeddingService.create_text_adapter(
                    self.model_config.caption_embedding
                )
            )
        settings = AppConfig.from_yaml(self.retrieval_config)
        data = DataService.load(
            self.paths.frames_path,
            {source: self.paths.enrichment_path(source)},
        )
        return RetrievalService.build_text_embedding_artifacts(
            data,
            self._text_encoder,
            source,
            self.paths.index_root(source),
            embeddings_filename=settings.index.text_embedding_filenames[source],
        )


class S3CorpusPreparationService:
    """Coordinate an immutable S3 inventory through resumable offline stages."""

    def __init__(
        self,
        config: S3CorpusPreparationConfig,
        *,
        client: Any | None = None,
        operations: PreparationOperations | None = None,
        resume: bool = True,
        limit: int | None = None,
        offset: int | None = None,
        enrichment_config: str | Path = "configs/prepare.yaml",
        model_config: str | Path = "thundercompute/config.yaml",
        retrieval_config: str | Path = "configs/baseline.yaml",
        paths: PreparationPaths | None = None,
        dataset: Mapping[str, Any] | None = None,
    ) -> None:
        storage = config.preprocessing.s3
        if storage is None:
            raise ValueError("S3 corpus preparation requires S3 storage")
        self.config = config
        self.storage = storage
        self.dataset = dict(dataset) if dataset is not None else None
        if paths is not None:
            self.paths = paths
        else:
            from hcmai.data.enrichment.pipeline import EnrichmentJobConfig

            self.paths = PreparationPaths.from_enrichment_job(
                config,
                EnrichmentJobConfig.from_yaml(
                    enrichment_config,
                    dataset=self.dataset,
                ),
            )
        self.paths.state_root.mkdir(parents=True, exist_ok=True)
        self.client = client if client is not None else create_s3_client(storage)
        self.resume = resume
        self.limit = limit
        self.offset = offset
        self.operations = operations or DefaultPreparationOperations(
            config,
            self.paths,
            resume=resume,
            limit=limit,
            enrichment_config=enrichment_config,
            model_config=model_config,
            retrieval_config=retrieval_config,
            dataset=self.dataset,
            s3_client=self.client,
        )
    def run(self) -> PreparationRun:
        """Thực thi chuỗi Pipeline Preparation qua từng Stage.

        Lifecycle:
        1. Kéo danh sách S3 Video (Inventory).
        2. Tính toán Fingerprint (run_id) của lượt chạy.
        3. Resume/Skip những stage đã hoàn thành (nhờ marker .json).
        4. Import BTC frames and run ASR extraction when enabled.
        5. Lần lượt chạy các Enrichment (Caption, OCR, Indexing).
        """
        # =====================================================================
        # 1. INVENTORY & RUN ID INITIALIZATION
        # =====================================================================
        # Kéo danh sách file từ S3 và tạo ID cho lần chạy này
        sources, run_id, inventory = self._sources_and_inventory()
        setattr(self.operations, "_current_run_id", run_id)
        
        # Danh sách theo dõi tiến độ các stage
        completed: list[str] = []
        skipped: list[str] = []

        # =====================================================================
        # 2. STAGE 1: BTC FRAME IMPORT & ASR EXTRACTION
        # =====================================================================
        # Check FrameStore and ASR markers before doing any work.
        frame_pending = self.config.stages.frame_store and self._pending(
            "frame_store",
            run_id,
            self._stage_outputs("frame_store"),
            skipped,
        )
        
        asr_pending = self.config.stages.asr and self._pending(
            "asr",
            run_id,
            self._stage_outputs("asr"),
            skipped,
            record_skip=False,
        )
        
        if frame_pending:
            self.operations.prepare_btc_frame_store()
            self._complete_stage("frame_store", run_id)
            completed.append("frame_store")

        # BTC frame import is independent of videos; source videos are needed
        # only for timestamped ASR preparation.
        if asr_pending:
            logger.info("Starting ASR preparation stage...")
            
            # Xử lý tuần tự từng video
            for i, source in enumerate(
                tqdm(sources, desc="Videos Processed", unit="video")
            ):
                logger.info(
                    "Processing video %d/%d: %s",
                    i + 1,
                    len(sources),
                    source.video_id,
                )
                
                # Quản lý vòng đời video: Tải tạm thời -> Decode -> Xoá file sau khi xong
                with self._source_video(source) as video:
                    
                    # Bước lấy Audio track từ Video phục vụ cho ASR
                    if asr_pending:
                        self.operations.prepare_transcript(video)
                
            logger.info("Completed Video Frame & ASR Preparation Stage.")

        # =====================================================================
        # 3. STAGE 2: ENRICHMENT & INDEXING
        # =====================================================================
        # Định nghĩa các stage phía sau theo định dạng: 
        # (Tên stage, Điều kiện kích hoạt trong config, Hàm callback để thực thi)
        simple_stages = (
            ("caption", self.config.stages.caption, self.operations.generate_caption),
            ("ocr", self.config.stages.ocr, self.operations.generate_ocr),
            ("objects", self.config.stages.objects, self.operations.detect_objects),
            ("asr", self.config.stages.asr, self.operations.materialize_asr),
            (
                "frame_context",
                self.config.stages.frame_context,
                self.operations.build_frame_context,
            ),
            (
                "visual_index",
                self.config.stages.visual_index,
                self.operations.build_visual_index,
            ),
            (
                "caption_index",
                self.config.stages.caption_index,
                lambda: self.operations.build_text_index(RetrievalSource.CAPTION),
            ),
            (
                "ocr_index",
                self.config.stages.ocr_index,
                lambda: self.operations.build_text_index(RetrievalSource.OCR),
            ),
            (
                "asr_index",
                self.config.stages.asr_index,
                lambda: self.operations.build_text_index(RetrievalSource.ASR),
            ),
        )
        
        # Chạy tuyến tính từng Enrichment Stage
        for stage_name, condition, executor in simple_stages:
            
            # Nếu người dùng disable tính năng này trong config
            if not condition:
                continue
                
            # Kiểm tra xem stage này đã có marker báo đã xong chưa
            stage_pending = (
                asr_pending
                if stage_name == "asr"
                else self._pending(
                    stage_name,
                    run_id,
                    self._stage_outputs(stage_name),
                    skipped,
                )
            )
            
            # Xử lý log "Skip" cho riêng ASR (để không in 2 lần do ASR gồm 2 bước: Audio Extract + Transcribe)
            if stage_name == "asr" and not stage_pending:
                skipped.append("asr")
                
            # Bắt đầu chạy nếu nó chưa hoàn thành
            if stage_pending:
                logger.info(f"Starting enrichment stage: {stage_name.upper()}")
                executor()
                self._complete_stage(stage_name, run_id)
                completed.append(stage_name)
                logger.info(f"Completed enrichment stage: {stage_name.upper()}")

        publication = None

        return PreparationRun(
            run_id=run_id,
            inventory_path=inventory,
            artifacts_root=self.paths.artifacts_root,
            source_count=len(sources),
            completed_stages=tuple(completed),
            skipped_stages=tuple(skipped),
            publication=publication,
        )

    def cache_sources(self) -> PreparationCacheRun:
        """Download and verify the source inventory without running any model."""

        from time import perf_counter

        started = perf_counter()
        sources, run_id, inventory = self._sources_and_inventory()
        cache_root = self.storage.cache_root
        if cache_root is None:
            raise ValueError("cache-only preparation requires s3.cache_root")
        root = cache_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        downloaded = reused = total = 0
        for source in sources:
            path = root / Path(source.key).name
            if path.is_file() and path.stat().st_size == source.size:
                reused += 1
            else:
                partial = path.with_suffix(f"{path.suffix}.partial")
                partial.unlink(missing_ok=True)
                try:
                    self.client.download_file(
                        self.storage.bucket, source.key, str(partial)
                    )
                    if partial.stat().st_size != source.size:
                        raise OSError(f"cached source size mismatch: {source.key}")
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
                downloaded += 1
            total += source.size
        free_gib = shutil.disk_usage(root).free / (1024 ** 3)
        if free_gib < self.config.execution.minimum_free_gib_after_cache:
            raise OSError("source cache leaves less free disk than configured")
        return PreparationCacheRun(
            run_id=run_id,
            inventory_path=inventory,
            cache_root=root,
            source_count=len(sources),
            downloaded_count=downloaded,
            reused_count=reused,
            total_bytes=total,
            duration_seconds=perf_counter() - started,
        )

    def _sources_and_inventory(
        self,
    ) -> tuple[list[S3VideoObject], str, Path]:
        sources = list_video_objects(self.client, self.storage, limit=None)
        if self.offset is not None:
            sources = sources[self.offset:]
        if self.limit is not None:
            sources = sources[:self.limit]
            
        run_id, inventory = self._record_inventory(sources)
        return sources, run_id, inventory

    @contextmanager
    def _source_video(self, source: S3VideoObject):
        cache_root = self.storage.cache_root
        if cache_root is not None:
            root = cache_root.expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            path = root / Path(source.key).name
            if not path.is_file() or path.stat().st_size != source.size:
                partial = path.with_suffix(f"{path.suffix}.partial")
                partial.unlink(missing_ok=True)
                try:
                    self.client.download_file(
                        self.storage.bucket, source.key, str(partial)
                    )
                    if partial.stat().st_size != source.size:
                        raise OSError(f"cached source size mismatch: {source.key}")
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
            yield path
            return
        with staged_video(self.client, self.storage, source) as video:
            yield video

    def _record_inventory(
        self,
        sources: Sequence[S3VideoObject],
    ) -> tuple[str, Path]:
        """Lưu lại danh sách Video và tính toán mã băm SHA256 (Run ID).
        
        Mã băm này đóng vai trò là `frame_store_id` (Pipeline Fingerprint), 
        ngăn chặn việc nối ghép sai lệch dữ liệu nếu Config hoặc Inventory thay đổi.
        """
        objects = [
            {**asdict(source), "source_version": source.source_version}
            for source in sources
        ]
        identity = {
            "pipeline_version": _PIPELINE_VERSION,
            "corpus_revision": self.config.corpus_revision,
            "limit": self.limit,
            "offset": self.offset,
            "configuration": self.config.model_dump(mode="json"),
            "source": {
                "bucket": self.storage.bucket,
                "videos_prefix": self.storage.videos_prefix,
                "objects": objects,
            },
        }
        encoded = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        run_id = hashlib.sha256(encoded).hexdigest()
        inventory = self.paths.state_root / "run.json"
        payload = {
            **identity,
            "run_id": run_id,
            "source_count": len(sources),
        }
        if inventory.exists():
            existing = read_json(inventory)
            if existing.get("run_id") != run_id:
                raise RuntimeError(
                    "S3 inventory or preparation configuration changed inside "
                    "an existing run directory"
                )
        else:
            _atomic_json(inventory, payload)
        return run_id, inventory

    def _pending(
        self,
        stage: str,
        run_id: str,
        outputs: Sequence[Path],
        skipped: list[str],
        *,
        record_skip: bool = True,
    ) -> bool:
        """State Machine Marker: Kiểm tra xem một Stage có cần chạy lại không.
        
        Thay vì chỉ check `path.exists()`, hệ thống đối soát `run_id` trong marker
        và so khớp `file size` của các artifact, đảm bảo khả năng khôi phục (Resume)
        chính xác nhất ngay cả khi file bị xóa dở dang.
        """
        if not self.resume:
            return True
        marker = self.paths.state_root / "stages" / f"{stage}.json"
        if marker.is_file():
            value = read_json(marker)
            expected_identity = self._stage_marker_identity(stage, run_id)
            if (
                value.get("run_id") == run_id
                and value.get("dependency_fingerprint")
                == expected_identity["dependency_fingerprint"]
            ):
                sizes = value.get("sizes", {})
                if all(
                    path.exists()
                    and path.stat().st_size == sizes.get(str(path), -1)
                    for path in outputs
                ) and self._stage_manifest_matches(
                    stage, expected_identity["dependency_identity"]
                ):
                    if record_skip:
                        skipped.append(stage)
                    return False
        return True

    def _complete_stage(self, stage: str, run_id: str) -> None:
        outputs = self._stage_outputs(stage)
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{stage} did not produce required outputs: " + ", ".join(missing)
            )
        identity = self._stage_marker_identity(stage, run_id)
        if not self._stage_manifest_matches(
            stage, identity["dependency_identity"]
        ):
            raise ValueError(
                f"{stage} manifest does not match active stage identity"
            )
        _atomic_json(
            self.paths.state_root / "stages" / f"{stage}.json",
            {
                "run_id": run_id,
                "stage": stage,
                "completed_at": datetime.now(UTC).isoformat(),
                **identity,
                "outputs": [str(path) for path in outputs],
                "sizes": {str(path): path.stat().st_size for path in outputs if path.exists()},
            },
        )

    def _stage_marker_identity(
        self, stage: str, run_id: str
    ) -> dict[str, Any]:
        """Build one deterministic marker identity for the active stage."""

        provider = getattr(self.operations, "stage_dependency_identity", None)
        raw = provider(stage) if callable(provider) else {}
        if not isinstance(raw, dict):
            raise TypeError("stage_dependency_identity must return a mapping")
        dependency_identity = _identity_value(raw)
        fingerprint = _fingerprint(
            {
                "run_id": run_id,
                "stage": stage,
                "dependency_identity": dependency_identity,
            }
        )
        return {
            "dependency_identity": dependency_identity,
            "dependency_fingerprint": fingerprint,
        }

    def _stage_manifest_matches(
        self, stage: str, dependency_identity: Any
    ) -> bool:
        """Compare stable manifest fields with the active policy identity."""

        if not isinstance(dependency_identity, dict):
            return False
        expected = dependency_identity.get("manifest")
        if not expected:
            return True
        if not isinstance(expected, dict):
            return False
        manifest_paths = [
            path
            for path in self._stage_outputs(stage)
            if path.name == "manifest.json"
        ]
        if len(manifest_paths) != 1 or not manifest_paths[0].is_file():
            return False
        try:
            actual = read_json(manifest_paths[0])
        except Exception:
            return False
        return _identity_subset(actual, expected)

    def _stage_outputs(self, stage: str) -> tuple[Path, ...]:
        caption_outputs = (
            self.paths.caption_root / "captions.parquet",
            self.paths.caption_root / "failures.json",
            self.paths.caption_root / "frame_enrichment.parquet",
            self.paths.caption_root / "manifest.json",
        )
        ocr_outputs = (
            self.paths.ocr_root / "frames.parquet",
            self.paths.ocr_root / "regions.parquet",
            self.paths.ocr_root / "failures.json",
            self.paths.ocr_root / "frame_enrichment.parquet",
            self.paths.ocr_root / "ocr_report.json",
            self.paths.ocr_root / "manifest.json",
        )
        outputs = {
            "frame_store": (
                self.paths.frames_path,
                self.paths.frame_store_root / "manifest.json",
            ),
            "asr": (
                self.paths.transcripts_root,
                self.paths.asr_enrichment_path,
            ),
            "caption": caption_outputs,
            "ocr": ocr_outputs,
            "objects": (
                self.paths.object_root / "frames.parquet",
                self.paths.object_root / "detections.parquet",
                self.paths.object_root / "manifest.json",
            ),
            "frame_context": (
                self.paths.context_root / "frame_context_v1.parquet",
                self.paths.context_root / "manifest.json",
            ),
            "visual_index": (
                self.paths.visual_embeddings_path,
                self.paths.visual_mapping_path,
                self.paths.visual_index_root / "dense.index",
            ),
            **{
                f"{source.value}_index": (
                    self.paths.text_embeddings_path(source),
                    self.paths.text_mapping_path(source),
                    self.paths.index_root(source) / "dense.index",
                )
                for source in _TEXT_SOURCES
            },
        }
        return outputs[stage]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda staging: write_json(value, staging))


def _identity_value(value: Any) -> Any:
    """Normalize dataclass/config identity into deterministic JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported stage identity value: {type(value).__name__}")


def _fingerprint(value: dict[str, Any]) -> str:
    """Hash one normalized stage identity with stable JSON encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_subset(actual: Any, expected: Any) -> bool:
    """Return whether every expected manifest field matches recursively."""

    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _identity_subset(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _identity_subset(item, wanted)
            for item, wanted in zip(actual, expected, strict=True)
        )
    return type(actual) is type(expected) and actual == expected
