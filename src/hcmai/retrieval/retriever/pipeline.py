"""Public service boundary for multimodal retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hcmai.common.config import FusionConfig, RetrievalCacheConfig
from hcmai.common.schemas import RetrievalResult, RetrievalSource
from hcmai.corpus import Corpus
from hcmai.retrieval.embedding.pipeline import TextEmbeddingAdapter
from hcmai.retrieval.retriever.dense.index import INDEX_FILENAME, DenseIndex
from hcmai.retrieval.retriever.cache import EmbeddingCache
from hcmai.retrieval.retriever.dense.retriever import DenseRetriever
from hcmai.retrieval.retriever.fusion.rrf import RRFFusionRetriever
from hcmai.retrieval.retriever.models.contracts import Retriever, VectorRetriever
from hcmai.retrieval.retriever.models.metadata import IndexMetadata
from hcmai.retrieval.retriever.query_batch import SourceFamily
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
from hcmai.retrieval.retriever.segment.retriever import ASRSegmentRetriever
from hcmai.retrieval.retriever.text.retriever import ContextRetriever
from hcmai.retrieval.retriever.video_scores import (
    VideoEventScores,
    score_all_videos,
)


class RetrievalService:
    """Quản lý các module tìm kiếm (retriever) cho hình ảnh và văn bản.
    Hỗ trợ cấu hình tìm kiếm đơn luồng (một retriever) hoặc đa luồng (Fusion Retriever)
    với cơ chế cache (embedding cache) để tăng tốc độ truy vấn.
    """

    INDEX_FILENAME = INDEX_FILENAME

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    @classmethod
    def from_index(
        cls,
        index: DenseIndex,
        encoder: TextEmbeddingAdapter,
        source: RetrievalSource = RetrievalSource.VISUAL,
        cache_config: RetrievalCacheConfig | None = None,
    ) -> "RetrievalService":
        cache = _embedding_cache(cache_config)
        prompt_version = (
            cache_config.prompt_version if cache_config is not None else "query-v1"
        )
        return cls(
            DenseRetriever(
                encoder,
                index,
                source,
                embedding_cache=cache,
                prompt_version=prompt_version,
            )
        )

    @classmethod
    def from_fast_track_indexes(
        cls,
        *,
        visual_index: DenseIndex,
        visual_encoder: TextEmbeddingAdapter,
        context_index: DenseIndex | None,
        asr_segment_index: SegmentDenseIndex | None,
        text_encoder: TextEmbeddingAdapter | None,
        corpus: Corpus,
        fusion: FusionConfig,
        cache_config: RetrievalCacheConfig | None = None,
        max_projection_gap_ms: int = 5_000,
    ) -> "RetrievalService":
        """Compose the explicit Visual, Context, and segment-ASR fast track.

        Context and ASR are optional, but either one requires the shared text
        query encoder.  ASR candidates are projected to canonical frame IDs by
        ``ASRSegmentRetriever`` before reciprocal-rank fusion sees them.
        """

        has_text_index = context_index is not None or asr_segment_index is not None
        if has_text_index and text_encoder is None:
            raise ValueError(
                "text_encoder is required when Context or ASR segment indexes "
                "are configured"
            )

        cache = _embedding_cache(cache_config)
        prompt_version = (
            cache_config.prompt_version if cache_config is not None else "query-v1"
        )
        retrievers: list[VectorRetriever] = [
            DenseRetriever(
                visual_encoder,
                visual_index,
                RetrievalSource.VISUAL,
                embedding_cache=cache,
                prompt_version=prompt_version,
            )
        ]
        if context_index is not None:
            assert text_encoder is not None
            retrievers.append(
                ContextRetriever(
                    text_encoder,
                    context_index,
                    cache,
                    prompt_version,
                )
            )
        if asr_segment_index is not None:
            assert text_encoder is not None
            frames = tuple(
                corpus.frames(
                    [str(frame_id) for frame_id in visual_index.mapping["frame_id"]]
                )
            )
            retrievers.append(
                ASRSegmentRetriever(
                    text_encoder,
                    asr_segment_index,
                    frames,
                    cache,
                    prompt_version,
                    max_projection_gap_ms,
                )
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

    def score_event_videos(
        self,
        events: Sequence[str],
        *,
        chunk_size: int = 65_536,
    ) -> list[VideoEventScores]:
        """Score every visual-index frame for each ordered event.

        The Phase A temporal baseline deliberately uses only the visual
        retriever. It encodes all events in one batch and supplies full-corpus
        scores to the monotonic decoder without shortlist or filter gating.
        """

        if not events:
            raise ValueError("events must not be empty")
        visual = self._retriever_for("visual")
        batch = visual.encode(list(events))
        return score_all_videos(
            visual.index,
            batch.vectors,
            chunk_size,
        )

    @property
    def active_sources(self) -> tuple[RetrievalSource, ...]:
        """Report configured modality indexes in deterministic order."""

        retrievers = getattr(self._retriever, "retrievers", (self._retriever,))
        active = {getattr(retriever, "source") for retriever in retrievers}
        return tuple(source for source in RetrievalSource if source in active)

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> RetrievalResult:
        """Retrieve one query globally through the configured detached stack."""

        return self._retriever.search(query, top_k)

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 100,
    ) -> list[RetrievalResult]:
        """Retrieve multiple ordered queries with batched encoder calls."""

        return self._retriever.search_batch(queries, top_k)

    def _retriever_for(self, source_family: SourceFamily) -> Any:
        """Return the one configured retriever owning an embedding family."""

        retrievers = getattr(self._retriever, "retrievers", (self._retriever,))
        for retriever in retrievers:
            if getattr(retriever, "source_family", None) == source_family:
                return retriever
        raise RuntimeError(
            f"No {source_family!r} retriever is configured for retrieval"
        )

    @staticmethod
    def load_index(
        index_dir: str | Path,
    ) -> DenseIndex:
        return DenseIndex.load(index_dir)

def _embedding_cache(
    config: RetrievalCacheConfig | None,
) -> EmbeddingCache | None:
    if config is None or not config.enabled:
        return None
    return EmbeddingCache(
        max_entries=config.embedding_max_entries,
        max_bytes=config.embedding_max_bytes,
        ttl_seconds=config.embedding_ttl_seconds,
    )
