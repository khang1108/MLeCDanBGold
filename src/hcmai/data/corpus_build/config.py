"""Cấu hình cho quá trình xây dựng Corpus.

Chứa các định nghĩa về thông số và đường dẫn cần thiết để chạy pipeline corpus.

Các tính năng chính:
1. Paths Configuration: Định nghĩa đường dẫn thư mục đầu vào (frames, evidence) và đầu ra (corpus đích).
2. Tham số chạy: Các tuỳ chọn về số luồng (threads), batch size khi build dữ liệu.
3. Versioning: Cấu hình tiền tố (prefix) hoặc phiên bản cho các lần build corpus khác nhau."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from hcmai.data.preprocessing.config import PreprocessingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]
LEGACY_LOCAL_ROOTS = (
    (PROJECT_ROOT / "data").resolve(),
    (PROJECT_ROOT / "artifacts").resolve(),
)
_MOVING_REVISIONS = {"current", "head", "latest", "main", "master", "newest"}
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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
    asr: bool = True
    
    visual_index: bool = True
    caption_index: bool = True
    ocr_index: bool = True
    asr_index: bool = True

    @model_validator(mode="after")
    def validate_dependencies(self) -> PreparationStagesConfig:
        values = self.model_dump()
        if not any(values.values()):
            raise ValueError("at least one preparation stage must be enabled")
        if any(
            enabled for name, enabled in values.items() if name != "frame_store"
        ) and not self.frame_store:
            raise ValueError("downstream stages require frame_store")
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

    dino: PinnedModelConfig
    caption: PinnedModelConfig
    ocr: PinnedModelConfig
    asr: PinnedModelConfig
    diarization: PinnedModelConfig
    visual_embedding: PinnedModelConfig
    text_embedding: PinnedModelConfig


class S3CorpusPreparationConfig(BaseModel):
    """Authoritative production contract for the newest S3 video corpus."""

    corpus_revision: str = Field(min_length=3, max_length=128)
    work_root: Path
    stages: PreparationStagesConfig = Field(
        default_factory=PreparationStagesConfig
    )
    models: PreparationModelPins
    preprocessing: PreprocessingConfig

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
        preprocessing = self.preprocessing
        if preprocessing.s3 is None or preprocessing.videos_root is not None:
            raise ValueError(
                "production corpus preparation is S3-only and cannot use "
                "videos_root"
            )


        staging = preprocessing.s3.staging_root
        if staging is None:
            raise ValueError("production S3 preparation requires staging_root")
        staging_root = _resolved_absolute(staging, "s3.staging_root")
        _reject_legacy_local(staging_root, "s3.staging_root")
        if not _inside(staging_root, self.work_root):
            raise ValueError("s3.staging_root must be inside work_root")

        full = preprocessing.s3.artifacts_prefix
        smoke = preprocessing.s3.smoke_artifacts_prefix
        if (
            full == smoke
            or full.startswith(f"{smoke}/")
            or smoke.startswith(f"{full}/")
        ):
            raise ValueError(
                "smoke and full artifact prefixes must be separate namespaces"
            )

        dino = self.models.dino
        if preprocessing.dino_revision is None:
            raise ValueError("production preprocessing requires a DINO revision")
        if (
            preprocessing.dino_model != dino.model_name
            or preprocessing.dino_revision != dino.revision
        ):
            raise ValueError(
                "preprocessing DINO model pin must match preparation.models.dino"
            )
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
        return cls.model_validate(preparation)
