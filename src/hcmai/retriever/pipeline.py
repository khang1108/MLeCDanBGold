"""Public service boundary for multimodal retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from hcmai.common.config import FusionConfig
from hcmai.common.schemas import RetrievalResult, RetrievalSource, TaskType
from hcmai.common.schemas.search import SearchFilters
from hcmai.embedding.pipeline import TextEmbeddingAdapter
from hcmai.retriever.dense.index import INDEX_FILENAME, DenseIndex
from hcmai.retriever.dense.retriever import DenseRetriever
from hcmai.retriever.fusion.rrf import RRFFusionRetriever
from hcmai.retriever.models.contracts import Retriever
from hcmai.retriever.models.metadata import IndexMetadata
from hcmai.retriever.query_batch import QueryEmbeddingBatch, SourceFamily
from hcmai.retriever.text.artifacts import build_text_artifacts
from hcmai.retriever.text.retriever import (
    ASRRetriever,
    CaptionRetriever,
    OCRRetriever,
)

_TEXT_RETRIEVERS = {
    RetrievalSource.CAPTION: CaptionRetriever,
    RetrievalSource.OCR: OCRRetriever,
    RetrievalSource.ASR: ASRRetriever,
}


class RetrievalService:
    """Own visual/text retrievers, fusion, and offline index operations."""

    INDEX_FILENAME = INDEX_FILENAME

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    @classmethod
    def from_index(
        cls,
        index: DenseIndex,
        encoder: TextEmbeddingAdapter,
        source: RetrievalSource = RetrievalSource.VISUAL,
    ) -> "RetrievalService":
        return cls(DenseRetriever(encoder, index, source))

    @classmethod
    def from_indexes(
        cls,
        visual_index: DenseIndex,
        visual_encoder: TextEmbeddingAdapter,
        text_indexes: Mapping[RetrievalSource, DenseIndex],
        text_encoder: TextEmbeddingAdapter,
        fusion: FusionConfig,
    ) -> "RetrievalService":
        retrievers: list[Retriever] = [
            DenseRetriever(visual_encoder, visual_index)
        ]
        retrievers.extend(
            _TEXT_RETRIEVERS[source](text_encoder, index)
            for source, index in text_indexes.items()
        )
        if len(retrievers) == 1:
            return cls(retrievers[0])
        return cls(RRFFusionRetriever(retrievers, fusion))

    @property
    def index_metadata(self) -> IndexMetadata:
        """Expose provenance for single-index benchmark reporting."""

        index = getattr(self._retriever, "index", None)
        if index is None:
            raise RuntimeError(
                "Index metadata is unavailable for fused retrieval"
            )
        return index.metadata

    @property
    def visual_index(self) -> DenseIndex:
        """Expose the visual index for full-frame rescoring such as TRAKE."""

        retrievers = getattr(self._retriever, "retrievers", (self._retriever,))
        for retriever in retrievers:
            if getattr(retriever, "source_family", None) == "visual":
                return cast(DenseIndex, retriever.index)
        raise RuntimeError("No visual index is configured for retrieval")

    def search(
        self,
        query: str,
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> RetrievalResult:
        return self._retriever.search(query, top_k, filters, query_type)

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalResult]:
        """Retrieve multiple ordered queries with batched encoder calls."""

        return self._retriever.search_batch(queries, top_k, filters, query_type)

    def encode_text_batch(
        self,
        texts: list[str],
        source_family: SourceFamily = "text",
    ) -> QueryEmbeddingBatch:
        """Expose a reusable query batch for task pipelines such as TRAKE."""

        retrievers = getattr(self._retriever, "retrievers", (self._retriever,))
        for retriever in retrievers:
            if getattr(retriever, "source_family", None) == source_family:
                return cast(Any, retriever).encode(texts)
        raise RuntimeError(
            f"No {source_family!r} query encoder is configured for retrieval"
        )

    @staticmethod
    def load_index(index_dir: str | Path) -> DenseIndex:
        return DenseIndex.load(index_dir)

    @staticmethod
    def build_index(
        embeddings: np.ndarray,
        mapping: pd.DataFrame,
        *,
        dataset_version: str,
        model_name: str,
        index_type: str = "flat_ip",
        show_progress: bool = False,
    ) -> DenseIndex:
        return DenseIndex.build(
            embeddings,
            mapping,
            dataset_version=dataset_version,
            model_name=model_name,
            index_type=index_type,
            show_progress=show_progress,
        )

    @staticmethod
    def build_text_artifacts(
        config_path: str | Path = "configs/baseline.yaml",
        model_config_path: str | Path = "llm/config.yaml",
        *,
        source: RetrievalSource = RetrievalSource.CAPTION,
        enrichment_path: str | Path | None = None,
        frames_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        encoder: TextEmbeddingAdapter | None = None,
    ) -> DenseIndex:
        return build_text_artifacts(
            config_path,
            model_config_path,
            source=source,
            enrichment_path=enrichment_path,
            frames_path=frames_path,
            output_dir=output_dir,
            encoder=encoder,
        )
