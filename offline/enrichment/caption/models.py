"""Caption contracts and configuration models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

from pydantic import Field, model_validator

from hcmai.common.utils.io import read_yaml_section
from offline.contracts import ContractModel, NonEmptyString
from offline.enrichment.dataset_cli import merge_dataset_values
from offline.enrichment.models import ProcessingStatus

ENRICHMENT_VERSION = "enrichment_version"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENRICHMENT_CONFIG = PROJECT_ROOT / "configs" / "prepare.yaml"


class CaptionAdapter(Protocol):
    """Protocol for frame captioning backends."""

    resolved_revision: str | None

    def resolve_revision(self) -> str: ...

    def caption_batch(self, images: Sequence[Any]) -> list[Any]: ...


@dataclass(frozen=True)
class CaptionConfig:
    """Settings identifying one reproducible caption enrichment."""

    model_checkpoint: str
    revision: str | None
    prompt: str
    decoding: dict[str, Any]
    device: str
    precision: str
    dtype: str
    image_size: int
    batch_size: int
    enrichment_version: str
    write_interval: int
    dataset_version: str

    def __post_init__(self) -> None:
        if min(self.batch_size, self.image_size, self.write_interval) < 1:
            raise ValueError("batch_size, image_size, and write_interval must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptionConfig:
        """Validate caption settings loaded from YAML."""
        values = dict(data)
        if "name" in values:
            values["model_checkpoint"] = values.pop("name")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        missing = sorted(known - set(values))
        if unknown:
            raise ValueError(f"Unknown caption configuration: {', '.join(unknown)}")
        if missing:
            raise ValueError(f"Missing caption configuration: {', '.join(missing)}")
        return cls(**values)


@dataclass(frozen=True)
class CaptionJobConfig:
    """Paths, lineage, and model settings for one caption job."""

    caption: CaptionConfig
    dataset_root: Path
    frames_path: Path
    output_dir: Path
    frame_store_id: str | None = None

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = DEFAULT_ENRICHMENT_CONFIG,
        *,
        dataset: Mapping[str, Any] | None = None,
    ) -> CaptionJobConfig:
        """Load a complete caption job from the preparation config."""
        config_path = Path(path).expanduser().resolve()
        raw = read_yaml_section(config_path, "enrichment")
        dataset_values = merge_dataset_values(
            raw,
            dict(dataset) if dataset else None,
        )
        dataset, caption = dataset_values, raw.get("caption")
        if not isinstance(dataset, dict) or not isinstance(caption, dict):
            raise ValueError("Enrichment YAML requires dataset and caption mappings")

        values = dict(caption)
        output_dir = values.pop("output_dir", None)
        dataset_root = dataset.get("data_root", dataset.get("root"))
        missing = sorted({"version", "frames_path"} - set(dataset))
        if not isinstance(dataset_root, (str, Path)):
            missing.append("data_root")
        if missing or output_dir is None:
            fields = missing + ([] if output_dir is not None else ["caption.output_dir"])
            raise ValueError(f"Missing enrichment configuration: {', '.join(fields)}")
        assert isinstance(dataset_root, (str, Path))

        values["dataset_version"] = dataset["version"]
        return cls(
            caption=CaptionConfig.from_dict(values),
            dataset_root=_project_path(dataset_root),
            frames_path=_project_path(dataset["frames_path"]),
            output_dir=_project_path(output_dir),
            frame_store_id=dataset.get("frame_store_id"),
        )


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class CaptionEvidence(ContractModel):
    """One frame-aligned caption with model and frame-store provenance."""

    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None
    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    text: str | None = None
    frame_store_id: NonEmptyString | None = None
    artifact_version: NonEmptyString
    model_name: NonEmptyString
    model_revision: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_failure_details(self) -> Self:
        """Require diagnostics when caption generation failed."""
        if self.status is ProcessingStatus.FAILED and (
            self.error_code is None or self.error_message is None
        ):
            raise ValueError("failed evidence requires error_code and error_message")
        return self


def usable_completed_text(row: CaptionEvidence) -> str | None:
    """Return usable completed caption text without inventing empty evidence."""
    if row.status is not ProcessingStatus.COMPLETED:
        return None
    return row.text if row.text is not None and row.text.strip() else None


__all__ = [
    "CaptionAdapter",
    "CaptionConfig",
    "CaptionEvidence",
    "CaptionJobConfig",
    "DEFAULT_ENRICHMENT_CONFIG",
    "ENRICHMENT_VERSION",
    "usable_completed_text",
]
