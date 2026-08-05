"""Value objects returned by embedding artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hcmai.embedding.models.metadata import EmbeddingMetadata


@dataclass(frozen=True)
class EmbeddingRun:
    metadata: EmbeddingMetadata
    embeddings_file: Path
    mapping_file: Path
    generated_count: int
