"""Runtime query encoding through configured visual and text adapters.

This module owns adapter construction and query-time encoding for loaded
retrieval indexes. Offline embedding artifact construction is owned by
``offline.embeddings``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.retrieval.embedding.models.contracts import (
    ImageEmbeddingAdapter,
    TextEmbeddingAdapter,
)
from hcmai.retrieval.embedding.models.stats import EncodingStats

__all__ = [
    "EmbeddingService",
    "EncodingStats",
    "ImageEmbeddingAdapter",
    "TextEmbeddingAdapter",
]


class EmbeddingService:
    """Create adapters and encode runtime visual or evidence queries.

    The service does not build or publish corpus embedding artifacts.
    """

    def __init__(
        self,
        visual: ImageEmbeddingAdapter | None = None,
        visual_query: TextEmbeddingAdapter | None = None,
        evidence_query: TextEmbeddingAdapter | None = None,
    ) -> None:
        """Configure optional adapters for the loaded runtime indexes."""
        self.visual = visual
        self.visual_query = visual_query or (
            cast(TextEmbeddingAdapter, visual)
            if hasattr(visual, "encode_text")
            else None
        )
        self.evidence_query = evidence_query

    @staticmethod
    def create_text_adapter(config: EncoderConfig) -> TextEmbeddingAdapter:
        """Create the explicitly configured local text adapter."""

        if config.backend == "bge_m3":
            from hcmai.retrieval.embedding.adapters.bge import BGEAdapter

            return BGEAdapter(config)
        from hcmai.retrieval.embedding.adapters.siglip import SigLIPAdapter

        return SigLIPAdapter(config)

    @staticmethod
    def create_visual_adapter(config: EncoderConfig) -> ImageEmbeddingAdapter:
        """Create the configured local visual adapter for image encoding."""
        from hcmai.retrieval.embedding.adapters.siglip import SigLIPAdapter

        return SigLIPAdapter(config)

    @staticmethod
    def create_remote_adapter(
        client: object,
        config: EncoderConfig,
        embedding_dim: int,
        source: str = "visual",
    ) -> TextEmbeddingAdapter:
        """Create a remote text adapter for evidence or visual queries."""
        from hcmai.retrieval.embedding.adapters.remote import (
            EmbeddingClient,
            RemoteEmbeddingAdapter,
        )

        return RemoteEmbeddingAdapter(
            cast(EmbeddingClient, client),
            config,
            embedding_dim,
            source,
        )

    @staticmethod
    def create_remote_visual_adapter(
        client: object,
        config: EncoderConfig,
        embedding_dim: int = 0,
    ) -> ImageEmbeddingAdapter:
        """Create a remote visual adapter for image encoding."""
        from hcmai.retrieval.embedding.adapters.remote import (
            ImageEmbeddingClient,
            RemoteImageEmbeddingAdapter,
        )

        return RemoteImageEmbeddingAdapter(
            cast(ImageEmbeddingClient, client),
            config,
            embedding_dim,
        )

    def encode_visual_images(
        self,
        images: list[Image.Image],
        stats: EncodingStats | None = None,
    ) -> np.ndarray:
        """Encode visual images without persisting vectors or mappings."""
        if self.visual is None:
            raise RuntimeError("Visual embedding adapter is not configured")
        return self.visual.encode_images(images, stats)

    def encode_visual_query(
        self,
        texts: list[str],
        stats: EncodingStats | None = None,
    ) -> np.ndarray:
        """Encode text into the visual index's query embedding space."""
        if self.visual_query is None:
            raise RuntimeError("Visual-query embedding adapter is not configured")
        return self.visual_query.encode_text(texts, stats)

    def encode_evidence_query(
        self,
        texts: list[str],
        stats: EncodingStats | None = None,
    ) -> np.ndarray:
        """Encode text into an evidence index's query embedding space."""
        if self.evidence_query is None:
            raise RuntimeError("Evidence-query embedding adapter is not configured")
        return self.evidence_query.encode_text(texts, stats)
