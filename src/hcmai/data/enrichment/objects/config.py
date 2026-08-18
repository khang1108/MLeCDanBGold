"""Configuration for deterministic BTC object artifact import."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ObjectConfig:
    """Paths, lineage, and summary policy for one object import."""

    objects_root: Path
    output_dir: Path
    artifact_version: str = "object-v1"
    summary_min_confidence: float = 0.25
    max_summary_labels: int = 20

    def __post_init__(self) -> None:
        if not str(self.artifact_version).strip():
            raise ValueError("artifact_version must not be empty")
        if not 0.0 <= self.summary_min_confidence <= 1.0:
            raise ValueError("summary_min_confidence must be in [0, 1]")
        if self.max_summary_labels < 1:
            raise ValueError("max_summary_labels must be positive")
