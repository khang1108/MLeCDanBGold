"""Provenance metadata for dense retrieval artifacts.

Metadata records describe an artifact that was produced and are serialized
next to it, kept separate from configuration inputs (``config.py``) and runtime
stats (``stats.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
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
    schema_version: str = "dense-index-v1"
    entity_kind: str = "frame"
    retrieval_source: str | None = None
    model_revision: str | None = None
    source_fingerprint: str | None = None
    config_fingerprint: str | None = None
    checksums: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexMetadata:
        """Create metadata from JSON, ignoring unknown future-version fields.

        The defaulted v2 provenance fields deliberately keep v1 metadata
        readable, which permits rollback to existing offline artifacts.
        """
        known_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known_fields})
