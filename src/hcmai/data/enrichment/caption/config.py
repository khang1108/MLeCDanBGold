"""Cấu hình cho quá trình Captioning (Mô tả ảnh).

Chứa các thiết lập (ví dụ: batch size, tên mô hình, đường dẫn) để điều khiển quá trình tạo caption cho các frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcmai.common.utils.io import read_yaml

ENRICHMENT_VERSION = "enrichment_version"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_ENRICHMENT_CONFIG = PROJECT_ROOT / "configs" / "enrichment.yaml"


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
    """Paths and model settings for one caption job."""

    caption: CaptionConfig
    dataset_root: Path
    frames_path: Path
    output_dir: Path

    @classmethod
    def from_yaml(
        cls, path: str | Path = DEFAULT_ENRICHMENT_CONFIG
    ) -> CaptionJobConfig:
        """Load a complete caption job from YAML."""
        config_path = Path(path).expanduser().resolve()
        raw = read_yaml(config_path)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping in {config_path}")
        dataset, caption = raw.get("dataset"), raw.get("caption")
        if not isinstance(dataset, dict) or not isinstance(caption, dict):
            raise ValueError("Enrichment YAML requires dataset and caption mappings")

        values = dict(caption)
        output_dir = values.pop("output_dir", None)
        missing = sorted({"version", "root", "frames_path"} - set(dataset))
        if missing or output_dir is None:
            fields = missing + ([] if output_dir is not None else ["caption.output_dir"])
            raise ValueError(f"Missing enrichment configuration: {', '.join(fields)}")

        values["dataset_version"] = dataset["version"]
        return cls(
            caption=CaptionConfig.from_dict(values),
            dataset_root=_project_path(dataset["root"]),
            frames_path=_project_path(dataset["frames_path"]),
            output_dir=_project_path(output_dir),
        )


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
