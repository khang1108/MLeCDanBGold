"""Local-input, group-scoped corpus preparation and immutable publication."""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from tqdm import tqdm

from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.io import read_json
from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
from hcmai.data.corpus_build.pipeline import (
    DefaultPreparationOperations,
    PreparationOperations,
    PreparationPaths,
    PreparationRun,
    S3CorpusPreparationService,
    _atomic_json,
)
from hcmai.data.corpus_build.publish import publish_group_artifacts
from hcmai.data.s3 import S3VideoObject, VIDEO_EXTENSIONS, create_s3_client

_GROUP_PIPELINE_VERSION = "local-group-preparation-v1"


class GroupSourceObject(BaseModel):
    """One immutable S3 object expected in the local group directory."""

    key: str = Field(min_length=1)
    size: int = Field(gt=0)
    etag: str = Field(min_length=1)
    last_modified_ns: int = Field(ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("key")
    @classmethod
    def require_video_key(cls, value: str) -> str:
        normalized = value.strip().lstrip("/")
        if Path(normalized).suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("source inventory contains a non-video key")
        return normalized

    def as_s3_object(self) -> S3VideoObject:
        return S3VideoObject(
            key=self.key,
            size=self.size,
            etag=self.etag,
            last_modified_ns=self.last_modified_ns,
        )


class GroupSourceInventory(BaseModel):
    """Bản ghi danh sách các file video thuộc về một Group cụ thể.
    Được sử dụng để đảm bảo tính nhất quán (không thiếu/thừa file) khi chạy pipeline phân tán.
    """

    group_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
    bucket: str = Field(min_length=3)
    prefix: str = Field(min_length=1)
    objects: list[GroupSourceObject] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> GroupSourceInventory:
        keys = [item.key for item in self.objects]
        video_ids = [Path(key).stem for key in keys]
        if len(set(keys)) != len(keys) or len(set(video_ids)) != len(video_ids):
            raise ValueError("group source keys and video IDs must be unique")
        prefix = self.prefix.strip("/") + "/"
        if any(not key.startswith(prefix) for key in keys):
            raise ValueError("group source object is outside the inventory prefix")
        self.prefix = self.prefix.strip("/")
        self.objects.sort(key=lambda item: item.key)
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> GroupSourceInventory:
        return cls.model_validate(read_json(Path(path)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_group(
    videos_root: str | Path,
    inventory: GroupSourceInventory,
) -> dict[str, Path]:
    """Kiểm tra và xác thực thư mục chứa video local so với danh sách (inventory).
    Đảm bảo không bị thiếu file, dư file lạ, hoặc sai kích thước/checksum.
    Trả về mapping từ video_id sang đường dẫn file tuyệt đối.
    """

    root = Path(videos_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"local group videos directory is missing: {root}")
    expected: dict[str, Path] = {}
    for item in inventory.objects:
        path = (root / Path(item.key).name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"local source is missing: {Path(item.key).name}")
        if path.stat().st_size != item.size:
            raise ValueError(f"local source size mismatch: {path.name}")
        if item.sha256 is not None and _sha256(path) != item.sha256:
            raise ValueError(f"local source checksum mismatch: {path.name}")
        expected[path.stem] = path
    actual = {
        path.stem
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }
    if actual != set(expected):
        raise ValueError("local group video set differs from source inventory")
    return expected


class GroupPreparationService(S3CorpusPreparationService):
    """Thực thi pipeline chuẩn bị dữ liệu (Data Preparation Pipeline) cho riêng MỘT Group.
    Đọc video từ ổ cứng local, gọi các hàm trích xuất đặc trưng (OCR, ASR, Embedding...),
    và xuất (publish) thẳng kết quả lên S3 để Reducer tổng hợp sau.
    """

    def __init__(
        self,
        config: S3CorpusPreparationConfig,
        videos_root: str | Path,
        inventory: GroupSourceInventory | str | Path,
        *,
        client: Any | None = None,
        operations: PreparationOperations | None = None,
        resume: bool = True,
        cleanup_raw: bool = False,
        cleanup_artifacts: bool = False,
        enrichment_config: str | Path = "configs/enrichment.yaml",
        model_config: str | Path = "llm/config.yaml",
        retrieval_config: str | Path = "configs/baseline.yaml",
    ) -> None:
        """
        Khởi tạo GroupPreparationService cho việc xử lý một Group.

        Args:
            config (S3CorpusPreparationConfig): Cấu hình pipeline chuẩn bị dữ liệu.
            videos_root (str | Path): Đường dẫn đến thư mục chứa video local.
            inventory (GroupSourceInventory | str | Path): Thông tin về các file video thuộc group.
            client (Any | None, optional): Client S3. Defaults to None.
            operations (PreparationOperations | None, optional): Các phép toán chuẩn bị dữ liệu. Defaults to None.
            resume (bool, optional): Có Resume pipeline hay không. Defaults to True.
            cleanup_raw (bool, optional): Có dọn dẹp video gốc sau khi chuẩn bị xong hay không. Defaults to False.
            cleanup_artifacts (bool, optional): Có dọn dẹp các file tạm (audio, frames) sau khi chuẩn bị xong hay không. Defaults to False.
            enrichment_config (str | Path, optional): Đường dẫn đến file cấu hình enrichment. Defaults to "configs/enrichment.yaml".
            model_config (str | Path, optional): Đường dẫn đến file cấu hình model. Defaults to "llm/config.yaml".
            retrieval_config (str | Path, optional): Đường dẫn đến file cấu hình retrieval. Defaults to "configs/baseline.yaml".
        """
        value = (
            GroupSourceInventory.from_json(inventory)
            if isinstance(inventory, (str, Path))
            else inventory
        )

        storage = config.preprocessing.s3
        if storage is None or value.bucket != storage.bucket:
            raise ValueError("group inventory bucket differs from preparation config")

        self.group_inventory = value
        self.local_videos = verify_local_group(videos_root, value)
        self.cleanup_raw = cleanup_raw
        self.cleanup_artifacts = cleanup_artifacts

        paths = PreparationPaths.for_group(config, value.group_id)
        actual_client = client if client is not None else create_s3_client(storage)

        actual_operations = operations or DefaultPreparationOperations(
            config,
            paths,
            resume=resume,
            limit=None,
            enrichment_config=enrichment_config,
            model_config=model_config,
            retrieval_config=retrieval_config,
            s3_client=actual_client,
        )
        super().__init__(
            config,
            client=actual_client,
            operations=actual_operations,
            resume=resume,
            limit=None,
            enrichment_config=enrichment_config,
            model_config=model_config,
            retrieval_config=retrieval_config,
            paths=paths,
        )

    def _sources_and_inventory(self):
        """
        Xác định danh sách các nguồn (videos) và thông tin inventory cho Group này.

        Returns:
            tuple: Tuple chứa list các S3VideoObject và dictionary thông tin identity.
        """
        sources = [item.as_s3_object() for item in self.group_inventory.objects]
        identity = {
            "pipeline_version": _GROUP_PIPELINE_VERSION,
            "corpus_revision": self.config.corpus_revision,
            "group": self.group_inventory.model_dump(mode="json"),
            "configuration": self.config.model_dump(mode="json"),
        }

        encoded = json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()

        run_id = hashlib.sha256(encoded).hexdigest()
        path = self.paths.state_root / "run.json"
        payload = {**identity, "run_id": run_id, "source_count": len(sources)}

        if path.exists() and read_json(path).get("run_id") != run_id:
            raise RuntimeError("group source or configuration changed in run directory")

        if not path.exists():
            _atomic_json(path, payload)
        return sources, run_id, path

    @contextmanager
    def _source_video(self, source: S3VideoObject):
        """
        Context manager để truy cập video nguồn cho một S3VideoObject.

        Args:
            source (S3VideoObject): Object đại diện cho video nguồn.

        Yields:
            Path: Đường dẫn tuyệt đối đến file video.
        """
        yield self.local_videos[source.video_id]

    def run(self) -> PreparationRun:
        """
        Thực thi pipeline chuẩn bị dữ liệu cho Group này.

        Returns:
            PreparationRun: Kết quả của quá trình chuẩn bị.
        """
        sources, run_id, inventory_path = self._sources_and_inventory()
        setattr(self.operations, "_current_run_id", run_id)

        completed: list[str] = []
        skipped: list[str] = []
        
        frame_pending = self._pending(
            "frame_store", run_id, self._stage_outputs("frame_store"), skipped
        )
        transcript_pending = self.config.stages.asr and self._pending(
            "transcripts",
            run_id,
            self._stage_outputs("transcripts"),
            skipped,
            record_skip=False,
        )

        prepared: list[Any] = []
        if frame_pending or transcript_pending:
            for source in tqdm(sources, desc=f"Preparing {self.group_inventory.group_id}"):
                video = self.local_videos[source.video_id]

                if frame_pending:
                    prepared.append(self.operations.prepare_frame(video, source))
                if transcript_pending:
                    self.operations.prepare_transcript(video)
            
            if frame_pending:
                self.operations.finalize_frames(prepared, sources)
                self._complete_stage("frame_store", run_id)
                completed.append("frame_store")
            
            if transcript_pending:
                self._complete_stage("transcripts", run_id)
                completed.append("transcripts")

        stages = (
            ("caption", self.config.stages.caption, self.operations.generate_caption),
            ("ocr", self.config.stages.ocr, self.operations.generate_ocr),
            ("asr", self.config.stages.asr, self.operations.materialize_asr),
            (
                "visual_embeddings",
                self.config.stages.visual_index,
                self.operations.build_visual_artifacts,
            ),
            *(
                (
                    f"{source.value}_embeddings",
                    getattr(self.config.stages, f"{source.value}_index"),
                    lambda source=source: self.operations.build_text_embeddings(source),
                )
                for source in (
                    RetrievalSource.CAPTION,
                    RetrievalSource.OCR,
                    RetrievalSource.ASR,
                )
            ),
        )

        for stage, enabled, execute in stages:
            if not enabled:
                continue
            
            if stage == "asr":
                pending = transcript_pending
                if not pending:
                    skipped.append(stage)
            else:
                pending = self._pending(
                    stage, run_id, self._stage_outputs(stage), skipped
                )
            
            if pending:
                execute()
                self._complete_stage(stage, run_id)
                completed.append(stage)

        publication = publish_group_artifacts(
            self.client,
            self.paths,
            self.config,
            group_id=self.group_inventory.group_id,
            run_id=run_id,
            source_manifest=self.group_inventory.model_dump(mode="json"),
        )
        
        references = getattr(self.operations, "_audio_references", None)
        
        if references is not None:
            references.cleanup()
        
        if self.cleanup_raw:
            for path in self.local_videos.values():
                path.unlink()
        
        if self.cleanup_artifacts:
            shutil.rmtree(self.paths.artifacts_root)
        
        return PreparationRun(
            run_id=run_id,
            inventory_path=inventory_path,
            artifacts_root=self.paths.artifacts_root,
            source_count=len(sources),
            completed_stages=tuple(completed),
            skipped_stages=tuple(skipped),
            publication=publication,
        )

    def _stage_outputs(self, stage: str) -> tuple[Path, ...]:
        outputs = {
            "frame_store": (
                self.paths.frames_path,
                self.paths.frame_store_root / "manifest.json",
            ),
            "transcripts": (self.paths.transcripts_root,),
            "asr": (self.paths.transcripts_root, self.paths.asr_enrichment_path),
            "caption": (
                self.paths.caption_root / "frame_enrichment.parquet",
                self.paths.caption_root / "manifest.json",
            ),
            "ocr": (
                self.paths.ocr_root / "frame_enrichment.parquet",
                self.paths.ocr_root / "manifest.json",
            ),
            "visual_embeddings": (
                self.paths.visual_embeddings_path,
                self.paths.visual_mapping_path,
            ),
            **{
                f"{source.value}_embeddings": (
                    self.paths.text_embeddings_path(source),
                    self.paths.text_mapping_path(source),
                )
                for source in (
                    RetrievalSource.CAPTION,
                    RetrievalSource.OCR,
                    RetrievalSource.ASR,
                )
            },
        }
        return outputs[stage]
