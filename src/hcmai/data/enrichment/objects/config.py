"""Configuration for deterministic BTC object artifact import."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata


def normalize_lineage(value: str | None, name: str) -> str | None:
    """NFC-normalize and trim an optional non-empty lineage identifier."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True)
class ObjectConfig:
    """Paths, lineage, and summary policy for one object import."""

    objects_root: Path
    output_dir: Path
    artifact_version: str = "object-v1"
    summary_min_confidence: float = 0.25
    max_summary_labels: int = 20

    def __post_init__(self) -> None:
        artifact_version = normalize_lineage(
            self.artifact_version, "artifact_version"
        )
        assert artifact_version is not None
        object.__setattr__(self, "artifact_version", artifact_version)
        if not 0.0 <= self.summary_min_confidence <= 1.0:
            raise ValueError("summary_min_confidence must be in [0, 1]")
        if self.max_summary_labels < 1:
            raise ValueError("max_summary_labels must be positive")
