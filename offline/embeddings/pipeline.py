"""Offline entry points for constructing visual embedding artifacts.

This module composes the offline artifact writer with runtime-compatible
encoder adapters. It does not participate in online retrieval serving.
"""

from __future__ import annotations

from pathlib import Path

from hcmai.common.config import EncoderConfig
from hcmai.retrieval.embedding.models.contracts import ImageEmbeddingAdapter
from offline.embeddings.artifacts import EmbeddingArtifactBuilder
from offline.embeddings.models.artifacts import EmbeddingRun


def build_visual_artifacts(
    frames_path: Path | str,
    dataset_root: Path | str,
    output_dir: Path | str,
    encoder_config: EncoderConfig,
    dataset_version: str = "hcmai2026",
    encoder: ImageEmbeddingAdapter | None = None,
) -> EmbeddingRun:
    """Build visual vectors and mapping artifacts without changing their layout.

    The returned paths preserve the established ``embeddings/`` location under
    ``output_dir`` and the canonical frame-to-vector mapping semantics.
    """
    builder = EmbeddingArtifactBuilder(
        frames_path,
        dataset_root,
        output_dir,
        encoder_config,
        dataset_version,
        encoder,
    )
    metadata = builder.run()
    return EmbeddingRun(
        metadata=metadata,
        embeddings_file=builder.embeddings_file,
        mapping_file=builder.mapping_file,
        generated_count=len(builder.frame_mapping),
    )
