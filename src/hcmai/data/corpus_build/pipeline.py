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
from collections.abc import Sequence
from tqdm import tqdm

logger = logging.getLogger(__name__)
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from hcmai.data.preprocessing.s3 import S3Publication

from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.io import atomic_write, read_json, write_json
from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
from hcmai.data.preprocessing import FramePreparationSession
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
    visual_index_root: Path
    caption_index_root: Path
    ocr_index_root: Path
    asr_index_root: Path

    @classmethod
    def from_config(
        cls,
        config: S3CorpusPreparationConfig,
        limit: int | None,
    ) -> PreparationPaths:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero")
        if limit is None:
            artifacts = config.artifacts_root
            frame_store = artifacts / "frame_store"
            state = config.work_root / ".preparation"
        else:
            artifacts = config.work_root / f"artifacts.limit-{limit}"
            frame_store = artifacts / "frame_store"
            state = config.work_root / ".preparation" / f"limit-{limit}"
        return cls(
            artifacts_root=artifacts,
            state_root=state,
            frame_store_root=frame_store,
            transcripts_root=artifacts / "enrichment/transcripts",
            asr_root=artifacts / "enrichment/asr",
            caption_root=artifacts / "enrichment/caption",
            ocr_root=artifacts / "enrichment/ocr",
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


class PreparationOperations(Protocol):
    """Existing stage services adapted to the shared S3 source lifecycle."""

    def prepare_frame(self, video: Path, source: S3VideoObject) -> Any: ...

    def finalize_frames(
        self,
        prepared: Sequence[Any],
        sources: Sequence[S3VideoObject],
    ) -> Path: ...

    def prepare_transcript(self, video: Path) -> Path: ...

    def materialize_asr(self) -> Path: ...

    def generate_caption(self) -> Path: ...

    def generate_ocr(self) -> Path: ...

    def build_visual_index(self) -> Path: ...

    def build_text_index(self, source: RetrievalSource) -> Path: ...


class DefaultPreparationOperations:
    """Production adapter over the repository's current offline services."""

    def __init__(
        self,
        config: S3CorpusPreparationConfig,
        paths: PreparationPaths,
        *,
        resume: bool,
        limit: int | None,
        enrichment_config: str | Path = "configs/enrichment.yaml",
        model_config: str | Path = "llm/config.yaml",
        retrieval_config: str | Path = "configs/baseline.yaml",
    ) -> None:
        from hcmai.common.config import TranscriptJobConfig
        from hcmai.data.enrichment.caption.config import CaptionJobConfig
        from hcmai.llm.config import LLMServiceConfig

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
        self.caption_job = CaptionJobConfig.from_yaml(self.enrichment_config)
        self.transcript_job = TranscriptJobConfig.from_yaml(self.enrichment_config)
        self.model_config = LLMServiceConfig.from_yaml(self.model_config_path)
        
        self._validate_model_pins()
        self._frames: FramePreparationSession | None = None
        self._transcripts: Any | None = None
        self._text_encoder: Any | None = None

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

    def _frame_session(self) -> FramePreparationSession:
        """Khởi tạo FramePreparationSession (Xử lý video) nếu chưa có."""
        if self._frames is None:
            # Copy cấu hình để tránh ảnh hưởng đến config gốc
            preprocessing = self.config.preprocessing.model_copy()
            # Set output về thư mục FrameStore
            preprocessing.output_root = self.paths.frame_store_root
            self._frames = FramePreparationSession(
                preprocessing,
                resume=self.resume,
            )
        return self._frames

    def _transcript_service(self) -> Any:
        """Khởi tạo TranscriptService (Xử lý phụ đề/phân đoạn) nếu chưa có."""
        if self._transcripts is None:
            from hcmai.data.enrichment.transcripts.pipeline import TranscriptService

            self._transcripts = TranscriptService.from_job_config(
                self.transcript_job
            )
        return self._transcripts

    def prepare_frame(self, video: Path, source: S3VideoObject) -> Any:
        """Chuẩn bị dữ liệu video (tạo frame, embedding)."""
        return self._frame_session().prepare_video(
            video,
            source_version=source.source_version,
        )

    def finalize_frames(
        self,
        prepared: Sequence[Any],
        sources: Sequence[S3VideoObject],
    ) -> Path:
        inventory = self.paths.frame_store_root / "source-manifest.json"
        _atomic_json(
            inventory,
            {
                "bucket": self.storage.bucket,
                "videos_prefix": self.storage.videos_prefix,
                "objects": [asdict(source) for source in sources],
            },
        )
        return self._frame_session().finalize(
            list(prepared),
            limited_run=self.limit is not None,
            source={
                "type": "s3",
                "bucket": self.storage.bucket,
                "videos_prefix": self.storage.videos_prefix,
                "object_count": len(sources),
                "inventory": "source-manifest.json",
            },
        )

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
            frame_store_id=getattr(self, "_current_run_id", None),
        )

    def generate_caption(self) -> Path:
        from hcmai.data.enrichment.pipeline import EnrichmentService

        caption = replace(
            self.caption_job.caption,
            dataset_version=self.config.corpus_revision,
        )
        adapter = EnrichmentService.create_caption_adapter(caption)
        try:
            EnrichmentService.generate_captions(
                self.paths.frames_path,
                self.paths.caption_root,
                caption,
                adapter=adapter,
                dataset_root=self.paths.frame_store_root,
                frame_store_id=getattr(self, "_current_run_id", None),
            )
        finally:
            del adapter
        return self.paths.caption_root / "frame_enrichment.parquet"

    def generate_ocr(self) -> Path:
        from hcmai.data.enrichment.ocr.config import OCRConfig
        from hcmai.data.enrichment.pipeline import EnrichmentService

        pin = self.config.models.ocr
        ocr = OCRConfig(
            checkpoint=pin.model_name,
            revision=pin.revision,
            device=self.config.preprocessing.device,
            dataset_version=self.config.corpus_revision,
        )
        adapter = EnrichmentService.create_ocr_adapter(ocr)
        try:
            EnrichmentService.generate_ocr(
                self.paths.frames_path,
                self.paths.ocr_root,
                ocr,
                adapter=adapter,
                dataset_root=self.paths.frame_store_root,
                frame_store_id=getattr(self, "_current_run_id", None),
            )
        finally:
            del adapter
        return self.paths.ocr_root / "frame_enrichment.parquet"

    def build_visual_index(self) -> Path:
        import numpy as np
        import pandas as pd

        from hcmai.retrieval.embedding.pipeline import EmbeddingService
        from hcmai.retrieval.retriever.pipeline import RetrievalService

        run = EmbeddingService.build_visual_artifacts(
            self.paths.frames_path,
            self.paths.frame_store_root,
            self.paths.artifacts_root,
            self.model_config.visual_embedding,
            self.config.corpus_revision,
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

    def build_text_index(self, source: RetrievalSource) -> Path:
        from hcmai.retrieval.embedding.pipeline import EmbeddingService
        from hcmai.retrieval.retriever.pipeline import RetrievalService

        if self._text_encoder is None:
            self._text_encoder = EmbeddingService.create_text_adapter(
                self.model_config.caption_embedding
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
        enrichment_config: str | Path = "configs/enrichment.yaml",
        model_config: str | Path = "llm/config.yaml",
        retrieval_config: str | Path = "configs/baseline.yaml",
    ) -> None:
        storage = config.preprocessing.s3
        if storage is None:
            raise ValueError("S3 corpus preparation requires S3 storage")
        self.config = config
        self.storage = storage
        self.paths = PreparationPaths.from_config(config, limit)
        self.paths.state_root.mkdir(parents=True, exist_ok=True)
        self.client = client if client is not None else create_s3_client(storage)
        self.resume = resume
        self.limit = limit
        self.operations = operations or DefaultPreparationOperations(
            config,
            self.paths,
            resume=resume,
            limit=limit,
            enrichment_config=enrichment_config,
            model_config=model_config,
            retrieval_config=retrieval_config,
        )

    def run(self) -> PreparationRun:
        """Thực thi chuỗi Pipeline Preparation qua từng Stage.
        
        Lifecycle:
        1. Kéo danh sách S3 Video (Inventory).
        2. Tính toán Fingerprint (run_id) của lượt chạy.
        3. Resume/Skip những stage đã hoàn thành (nhờ marker .json).
        4. Chạy Preprocessing & ASR Extraction (song song hoặc đồng bộ).
        5. Lần lượt chạy các Enrichment (Caption, OCR, Indexing).
        """
        sources = list_video_objects(
            self.client,
            self.storage,
            limit=self.limit,
        )
        logger.info(f"Loaded {len(sources)} videos from S3 inventory.")

        # Tính toán mã băm Run ID để bảo toàn Data Lineage
        run_id, inventory = self._record_inventory(sources)
        completed: list[str] = []
        skipped: list[str] = []

        # Kiểm tra xem có cần xử lý FrameStore hay ASR từ Video gốc không
        frame_pending = self._pending(
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
        )
        prepared: list[Any] = []
        if frame_pending or asr_pending:
            logger.info("Starting Video Frame & ASR Preparation Stage...")
            for i, source in enumerate(tqdm(sources, desc="Videos Processed", unit="video")):
                logger.info(f"Processing video {i + 1}/{len(sources)}: {source.video_id}")
                with staged_video(self.client, self.storage, source) as video:
                    if frame_pending:
                        prepared.append(
                            self.operations.prepare_frame(video, source)
                        )
                    if asr_pending:
                        self.operations.prepare_transcript(video)
            if frame_pending:
                self.operations.finalize_frames(prepared, sources)
                self._complete_stage("frame_store", run_id)
                completed.append("frame_store")
            logger.info("Completed Video Frame & ASR Preparation Stage.")

        # Các stage đơn giản (tuyến tính) chạy tiếp theo (Caption, OCR, Indexing)
        simple_stages = (
            ("caption", self.config.stages.caption, self.operations.generate_caption),
            ("ocr", self.config.stages.ocr, self.operations.generate_ocr),
            ("asr", self.config.stages.asr, self.operations.materialize_asr),
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
        for stage_name, condition, executor in simple_stages:
            if condition and self._pending(
                stage_name, run_id, self._stage_outputs(stage_name), skipped
            ):
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
            if value.get("run_id") == run_id:
                sizes = value.get("sizes", {})
                if all(path.exists() and path.stat().st_size == sizes.get(str(path), -1) for path in outputs):
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
        _atomic_json(
            self.paths.state_root / "stages" / f"{stage}.json",
            {
                "run_id": run_id,
                "stage": stage,
                "completed_at": datetime.now(UTC).isoformat(),
                "outputs": [str(path) for path in outputs],
                "sizes": {str(path): path.stat().st_size for path in outputs if path.exists()},
            },
        )

    def _stage_outputs(self, stage: str) -> tuple[Path, ...]:
        outputs = {
            "frame_store": (
                self.paths.frames_path,
                self.paths.frame_store_root / "manifest.json",
            ),
            "asr": (
                self.paths.transcripts_root,
                self.paths.asr_enrichment_path,
            ),
            "caption": (
                self.paths.caption_root / "frame_enrichment.parquet",
                self.paths.caption_root / "manifest.json",
            ),
            "ocr": (
                self.paths.ocr_root / "frame_enrichment.parquet",
                self.paths.ocr_root / "manifest.json",
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
