"""Persist frame-native text embedding artifacts offline without an index.

The generated vector and mapping filenames remain compatible with existing
corpus-build consumers.  Runtime query encoders never call this writer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hcmai.retrieval.models import RetrievalSource
from hcmai.retrieval.embedding.pipeline import TextEmbeddingAdapter
from offline.indexes.text import (
    _TEXT_SOURCES,
    _FrameLookup,
    _TextEvidenceLookup,
    _embedding_artifact_name,
    _encode_texts,
    _normalized,
    _text_corpus,
)


def build_text_embedding_artifacts(
    frames: _FrameLookup,
    evidence: _TextEvidenceLookup,
    encoder: TextEmbeddingAdapter,
    source: RetrievalSource,
    output_dir: str | Path,
    *,
    embeddings_filename: str,
) -> tuple[Path, Path]:
    """Persist deterministic frame-aligned text vectors without an index."""

    if source not in _TEXT_SOURCES:
        raise ValueError(f"Unsupported text evidence source {source.value!r}")
    artifact_name = _embedding_artifact_name(embeddings_filename)
    texts, mapping = _text_corpus(frames, evidence, source)
    vectors = _normalized(_encode_texts(texts, encoder, source))
    if len(vectors) != len(mapping):
        raise ValueError("text embedding count does not match mapping rows")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vectors_path = output / artifact_name
    mapping_path = output / "frame_mapping.parquet"
    vectors_partial = vectors_path.with_suffix(f"{vectors_path.suffix}.partial")
    mapping_partial = mapping_path.with_suffix(f"{mapping_path.suffix}.partial")
    try:
        with vectors_partial.open("wb") as handle:
            np.save(handle, vectors)
        mapping.to_parquet(mapping_partial, index=False)
        vectors_partial.replace(vectors_path)
        mapping_partial.replace(mapping_path)
    finally:
        vectors_partial.unlink(missing_ok=True)
        mapping_partial.unlink(missing_ok=True)
    return vectors_path, mapping_path


__all__ = ["build_text_embedding_artifacts"]
