"""Public service boundary for visual and text embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.retrieval.embedding.artifacts import EmbeddingArtifactBuilder
from hcmai.retrieval.embedding.models.contracts import (
    ImageEmbeddingAdapter,
    TextEmbeddingAdapter,
)
from hcmai.retrieval.embedding.models.artifacts import EmbeddingRun
from hcmai.retrieval.embedding.models.stats import EncodingStats

__all__ = [
    "EmbeddingRun",
    "EmbeddingService",
    "EncodingStats",
    "ImageEmbeddingAdapter",
    "TextEmbeddingAdapter",
]


class EmbeddingService:
    """Quản lý các adapter mã hóa (encoder) cho hình ảnh và văn bản.
    Hỗ trợ cả adapter chạy local và remote (thông qua InferenceClientPool)
    để phục vụ việc sinh embeddings cho pipeline chuẩn bị dữ liệu (Data Preparation).
    """

    def __init__(
        self,
        visual: ImageEmbeddingAdapter | None = None,
        visual_query: TextEmbeddingAdapter | None = None,
        evidence_query: TextEmbeddingAdapter | None = None,
    ) -> None:
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
        from hcmai.retrieval.embedding.adapters.siglip import SigLIPAdapter

        return SigLIPAdapter(config)

    @staticmethod
    def create_remote_adapter(
        client: object,
        config: EncoderConfig,
        embedding_dim: int,
        source: str = "visual",
    ) -> TextEmbeddingAdapter:
        """Khởi tạo Remote Adapter để mã hóa văn bản (Text) trên Kaggle."""
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
        """Khởi tạo Remote Adapter để mã hóa hình ảnh (Visual) trên Kaggle."""
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
        if self.visual is None:
            raise RuntimeError("Visual embedding adapter is not configured")
        return self.visual.encode_images(images, stats)

    def encode_visual_query(
        self,
        texts: list[str],
        stats: EncodingStats | None = None,
    ) -> np.ndarray:
        if self.visual_query is None:
            raise RuntimeError("Visual-query embedding adapter is not configured")
        return self.visual_query.encode_text(texts, stats)

    def encode_evidence_query(
        self,
        texts: list[str],
        stats: EncodingStats | None = None,
    ) -> np.ndarray:
        if self.evidence_query is None:
            raise RuntimeError("Evidence-query embedding adapter is not configured")
        return self.evidence_query.encode_text(texts, stats)

    @staticmethod
    def build_visual_artifacts(
        frames_path: Path | str,
        dataset_root: Path | str,
        output_dir: Path | str,
        encoder_config: EncoderConfig,
        dataset_version: str = "hcmai2026",
        encoder: ImageEmbeddingAdapter | None = None,
    ) -> EmbeddingRun:
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
