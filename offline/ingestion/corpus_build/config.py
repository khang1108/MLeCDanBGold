"""Cấu hình cho quá trình xây dựng Corpus.

Chứa các định nghĩa về thông số và đường dẫn cần thiết để chạy pipeline corpus.

Các tính năng chính:
1. Paths Configuration: Định nghĩa đường dẫn thư mục đầu vào (frames, evidence) và đầu ra (corpus đích).
2. Tham số chạy: Các tuỳ chọn về số luồng (threads), batch size khi build dữ liệu.
3. Versioning: Cấu hình tiền tố (prefix) hoặc phiên bản cho các lần build corpus khác nhau."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_LOCAL_ROOTS = (
    (PROJECT_ROOT / "data").resolve(),
    (PROJECT_ROOT / "artifacts").resolve(),
)
_MOVING_REVISIONS = {"current", "head", "latest", "main", "master", "newest"}
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _s3_prefix(value: str) -> str:
    """Normalize one bucket-relative S3 prefix."""

    normalized = value.strip().strip("/")
    if not normalized or value.strip().startswith("s3://"):
        raise ValueError("S3 prefixes must be non-empty bucket-relative keys")
    if "\\" in normalized or any(
        part in {"", ".", ".."} for part in normalized.split("/")
    ):
        raise ValueError("S3 prefixes must not contain path traversal")
    return normalized


class S3PreparationConfig(BaseModel):
    """S3 transport settings shared by BTC-native preparation stages."""

    bucket: str = Field(min_length=3)
    videos_prefix: str = "videos"
    artifacts_prefix: str = "artifacts"
    smoke_artifacts_prefix: str = "artifacts/smoke"
    region: str | None = None
    endpoint_url: str | None = None
    staging_root: Path | None = None
    cache_root: Path | None = None
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    read_timeout_seconds: float = Field(default=300.0, gt=0)
    max_attempts: int = Field(default=4, ge=1, le=10)

    @field_validator("bucket")
    @classmethod
    def normalize_bucket(cls, value: str) -> str:
        """Keep bucket names plain and independent from URI syntax."""

        bucket = value.strip()
        if len(bucket) < 3 or bucket.startswith("s3://") or "/" in bucket:
            raise ValueError("bucket must be a plain S3 bucket name")
        return bucket

    @field_validator(
        "videos_prefix", "artifacts_prefix", "smoke_artifacts_prefix"
    )
    @classmethod
    def normalize_prefix(cls, value: str) -> str:
        """Reject absolute and traversal-like S3 keys."""

        return _s3_prefix(value)

    def artifacts_prefix_for_run(self, limit: int | None) -> str:
        """Keep smoke-test publication pointers outside the full corpus."""

        if limit is None:
            return self.artifacts_prefix
        return f"{self.smoke_artifacts_prefix}/limit-{limit}"


class PreparationStorageConfig(BaseModel):
    """Storage section retained after removing custom video preprocessing."""

    s3: S3PreparationConfig | None = None


def _resolved_absolute(path: Path, label: str) -> Path:
    """Giải quyết đường dẫn tuyệt đối và kiểm tra tính cô lập.

    Args:
        path: Đường dẫn cần xử lý (có thể tương đối hoặc tuyệt đối).
        label: Tên tham số (để in lỗi).

    Returns:
        Path tuyệt đối, đã được resolve() và không nằm trong các thư mục hệ thống (data/, artifacts/ legacy).
    """
    value = path.expanduser()
    if not value.is_absolute():
        raise ValueError(f"{label} must be an absolute isolated path")
    return value.resolve()


def _inside(path: Path, root: Path) -> bool:
    """Kiểm tra xem một đường dẫn có nằm bên trong (hoặc bằng) một thư mục gốc hay không."""
    return path == root or path.is_relative_to(root)


def _reject_legacy_local(path: Path, label: str) -> None:
    """Kiểm tra và loại trừ các đường dẫn nằm trong thư mục 'data/' hoặc 'artifacts/' cũ của repo."""
    if any(_inside(path, root) for root in LEGACY_LOCAL_ROOTS):
        raise ValueError(
            f"{label} must not use the repository's legacy local data/ or "
            "artifacts/ roots"
        )


class PreparationStagesConfig(BaseModel):
    """Thiết lập các stage được kích hoạt cho một lượt chuẩn bị dữ liệu."""

    frame_store: bool = True
    caption: bool = True
    ocr: bool = True
    objects: bool = False
    asr: bool = True
    frame_context: bool = False

    visual_index: bool = True
    caption_index: bool = True
    ocr_index: bool = True
    asr_index: bool = True

    @model_validator(mode="after")
    def validate_dependencies(self) -> PreparationStagesConfig:
        values = self.model_dump()
        if not any(values.values()):
            raise ValueError("at least one preparation stage must be enabled")
        if not self.frame_store and any(
            (
                self.caption,
                self.ocr,
                self.objects,
                self.asr,
                self.frame_context,
                self.visual_index,
                self.caption_index,
                self.ocr_index,
                self.asr_index,
            )
        ):
            raise ValueError("enabled preparation stages require frame_store")
        dependencies = {
            "caption_index": self.caption,
            "ocr_index": self.ocr,
            "asr_index": self.asr,
        }
        missing = [
            name
            for name, available in dependencies.items()
            if not available and getattr(self, name)
        ]
        if missing:
            raise ValueError(
                "text indexes require their enrichment stage: "
                + ", ".join(sorted(missing))
            )
        return self


class PinnedModelConfig(BaseModel):
    """One model name coupled to an immutable Hugging Face commit."""

    model_name: str = Field(min_length=1)
    revision: str

    @field_validator("model_name")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model_name must not be empty")
        return normalized

    @field_validator("revision")
    @classmethod
    def require_commit(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _COMMIT_PATTERN.fullmatch(normalized):
            raise ValueError(
                "revision must be a 40-character lowercase hexadecimal commit"
            )
        return normalized


class PreparationModelPins(BaseModel):
    """Immutable model identities required by every preparation stage."""

    caption: PinnedModelConfig
    ocr: PinnedModelConfig
    asr: PinnedModelConfig
    diarization: PinnedModelConfig
    visual_embedding: PinnedModelConfig
    text_embedding: PinnedModelConfig


class PreparationExecutionConfig(BaseModel):
    """Resource policy for one offline preparation run."""

    overlap_frame_asr: bool = False
    minimum_free_gib_after_cache: float = Field(default=10.0, ge=0)
    cache_download_workers: int = Field(default=4, ge=1, le=32)


class RemoteEndpointPoolConfig(BaseModel):
    """Cấu hình pool kết nối cho một nhóm remote GPU worker.
    Quản lý danh sách URLs endpoint, timeout, retry và giới hạn kết nối đồng thời.
    Giúp phân tán tải (load balancing) cho các tính năng nặng về GPU.
    """

    urls: list[str] = Field(min_length=1)
    timeout_seconds: float = Field(default=120, gt=0, le=600)
    connect_timeout_seconds: float = Field(default=5, gt=0, le=120)
    read_timeout_seconds: float = Field(default=120, gt=0, le=600)
    write_timeout_seconds: float = Field(default=30, gt=0, le=600)
    pool_timeout_seconds: float = Field(default=5, gt=0, le=120)
    max_attempts: int = Field(default=3, ge=1, le=10)
    max_concurrency: int = Field(default=2, ge=1, le=32)
    batch_size: int = Field(default=32, ge=1, le=256)

    @field_validator("urls")
    @classmethod
    def normalize_urls(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().rstrip("/") for value in values]
        if any(not value for value in normalized):
            raise ValueError("remote endpoint URLs must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("remote endpoint URLs must be unique")
        if any(
            not value.startswith("https://")
            and not value.startswith("http://127.0.0.1")
            and not value.startswith("http://localhost")
            for value in normalized
        ):
            raise ValueError("remote endpoints require HTTPS except localhost")
        return normalized


class RemoteInferencePoolsConfig(BaseModel):
    """Khai báo toàn bộ các remote workers phục vụ cho Data Pipeline.
    Nếu trường nào bị bỏ trống (None), pipeline sẽ chạy mô hình đó tại máy local.
    """

    caption: RemoteEndpointPoolConfig | None = None
    ocr: RemoteEndpointPoolConfig | None = None
    visual_embedding: RemoteEndpointPoolConfig | None = None
    text_embedding: RemoteEndpointPoolConfig | None = None
    transcript: RemoteEndpointPoolConfig | None = None

    @property
    def enabled(self) -> bool:
        return any(value is not None for value in self.__dict__.values())


class S3CorpusPreparationConfig(BaseModel):
    """Authoritative production contract for the newest S3 video corpus."""

    corpus_revision: str = Field(min_length=3, max_length=128)
    work_root: Path
    frame_store_source: Literal["btc_keyframes"] = "btc_keyframes"
    stages: PreparationStagesConfig = Field(
        default_factory=PreparationStagesConfig
    )
    models: PreparationModelPins
    preprocessing: PreparationStorageConfig
    execution: PreparationExecutionConfig = Field(
        default_factory=PreparationExecutionConfig
    )
    remote_inference: RemoteInferencePoolsConfig = Field(
        default_factory=RemoteInferencePoolsConfig
    )

    @field_validator("corpus_revision")
    @classmethod
    def require_corpus_revision(cls, value: str) -> str:
        normalized = value.strip()
        if (
            normalized.casefold() in _MOVING_REVISIONS
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", normalized)
        ):
            raise ValueError(
                "production preparation requires an immutable corpus revision"
            )
        return normalized

    @field_validator("work_root")
    @classmethod
    def normalize_work_root(cls, value: Path) -> Path:
        root = _resolved_absolute(value, "work_root")
        _reject_legacy_local(root, "work_root")
        return root

    @model_validator(mode="after")
    def validate_production_boundaries(self) -> S3CorpusPreparationConfig:
        storage = self.preprocessing.s3
        if storage is None:
            raise ValueError("production corpus preparation requires S3 storage")

        staging = storage.staging_root
        if staging is None:
            raise ValueError("production S3 preparation requires staging_root")
        staging_root = _resolved_absolute(staging, "s3.staging_root")
        _reject_legacy_local(staging_root, "s3.staging_root")
        if not _inside(staging_root, self.work_root):
            raise ValueError("s3.staging_root must be inside work_root")
        # Keep the normalized value on the nested config object. Runtime
        # consumers use ``storage.staging_root`` directly when creating
        # temporary directories, so leaving ``~`` here would produce a
        # literal path such as ``repo/~/MLeCDanBGold/...``.
        storage.staging_root = staging_root

        full = storage.artifacts_prefix
        smoke = storage.smoke_artifacts_prefix
        if (
            full == smoke
            or full.startswith(f"{smoke}/")
            or smoke.startswith(f"{full}/")
        ):
            raise ValueError(
                "smoke and full artifact prefixes must be separate namespaces"
            )

        cache_root = storage.cache_root
        if cache_root is not None:
            resolved_cache = _resolved_absolute(cache_root, "s3.cache_root")
            _reject_legacy_local(resolved_cache, "s3.cache_root")
            if not _inside(resolved_cache, self.work_root):
                raise ValueError("s3.cache_root must be inside work_root")
            storage.cache_root = resolved_cache
        if self.execution.overlap_frame_asr and cache_root is None:
            raise ValueError("overlap_frame_asr requires a persistent source cache")
        return self

    @property
    def artifacts_root(self) -> Path:
        """Return the isolated root that later stages must write below."""

        return self.work_root / "artifacts"

    @property
    def full_artifacts_prefix(self) -> str:
        storage = self.preprocessing.s3
        assert storage is not None
        return storage.artifacts_prefix

    @property
    def smoke_artifacts_prefix(self) -> str:
        storage = self.preprocessing.s3
        assert storage is not None
        return storage.smoke_artifacts_prefix

    @classmethod
    def from_yaml(cls, path: str | Path) -> S3CorpusPreparationConfig:
        """Load the top-level ``preparation`` mapping."""

        with Path(path).open(encoding="utf-8") as handle:
            values: dict[str, Any] = yaml.safe_load(handle) or {}
        preparation = values.get("preparation", values)
        if not isinstance(preparation, dict):
            raise ValueError("preparation YAML requires a mapping")
        preparation = dict(preparation)
        preprocessing = dict(preparation.get("preprocessing", {}))
        storage = dict(preprocessing.get("s3", {}))
        bucket = os.getenv("HCMAI_S3_BUCKET")
        region = os.getenv("HCMAI_S3_REGION")
        if bucket:
            storage["bucket"] = bucket
        if region:
            storage["region"] = region
        preprocessing["s3"] = storage
        preparation["preprocessing"] = preprocessing
        return cls.model_validate(preparation)
