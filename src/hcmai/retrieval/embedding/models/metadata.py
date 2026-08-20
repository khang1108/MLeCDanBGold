"""Metadata contracts for the embedding pipeline.

Kept separate from ``embedding.py`` so the corpus provenance record lives apart
from the source code that reads frames, builds embeddings, and writes artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Optional


@dataclass
class EmbeddingMetadata:
    """Metadata for a generated embedding corpus."""

    dataset_version: str
    model_name: str
    model_checkpoint: Optional[str]
    preprocessing_size: int
    dtype: str
    embedding_dimension: int
    total_frames: int
    successful_frames: int
    failed_frames: int
    normalization: str
    generated_at: str
    device: str
    batch_size: int
    processing_time_sec: float
    model_revision: str | None = None
    source_fingerprint: str | None = None
    schema_version: str = "visual-embedding-v2"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingMetadata:
        """Create from persisted metadata while accepting older mappings.

        The new defaulted provenance fields keep existing visual artifacts
        readable, and unknown newer fields are ignored for rollback safety.
        """
        known_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known_fields})
