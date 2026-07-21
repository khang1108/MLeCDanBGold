"""Provenance metadata (artifact descriptors) for the retriever pipeline.

Metadata records describe an artifact that was produced and are serialized
next to it, kept separate from configuration inputs (``config.py``) and runtime
stats (``stats.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class IndexMetadata:
    """Provenance and shape metadata describing a serialized FAISS index."""

    dataset_version: str
    model_name: str
    index_type: str
    metric: str
    normalization: str
    embedding_dim: int
    vector_count: int
    build_time_sec: float
    index_size_bytes: int
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexMetadata:
        """Create metadata from a dictionary loaded from JSON."""
        return cls(**data)
