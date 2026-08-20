"""Retrieve segment-native ASR hits and materialize canonical frame candidates.

This adapter encodes BGE text queries, searches ``SegmentDenseIndex``, and
projects returned timeline segments through ``FrameStore`` before frame-ID RRF.
It does not fabricate identity, fuse modalities, or apply frame-time filtering
after projection.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from hcmai.common.observability import PipelineStage
from hcmai.common.observability.tracing import StageTimer
from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    RetrievalTrace,
    TaskType,
)
from hcmai.common.schemas.search import SearchFilters
from hcmai.data.stores.frame import FrameStore
from hcmai.retrieval.embedding.pipeline import TextEmbeddingAdapter
from hcmai.retrieval.retriever.cache import EmbeddingCache
from hcmai.retrieval.retriever.query_batch import (
    QueryEmbeddingBatch,
    SourceFamily,
    encode_query_batch,
)
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
from hcmai.retrieval.retriever.segment.projector import (
    SegmentFrameProjection,
    SegmentFrameProjector,
)


class ASRSegmentRetriever:
    """Adapt segment-native ASR vector hits into canonical frame candidates."""

    source = RetrievalSource.ASR

    def __init__(
        self,
        encoder: TextEmbeddingAdapter,
        index: SegmentDenseIndex,
        frame_store: FrameStore,
        embedding_cache: EmbeddingCache | None = None,
        prompt_version: str = "query-v1",
        max_projection_gap_ms: int = 5_000,
    ) -> None:
        """Bind compatible text/index artifacts and canonical frame projection."""

        if index.metadata.model_name != encoder.config.model_name:
            raise ValueError(
                f"Encoder/index model mismatch: encoder={encoder.config.model_name!r}, "
                f"index={index.metadata.model_name!r}"
            )
        if encoder.embedding_dim and index.metadata.embedding_dim != encoder.embedding_dim:
            raise ValueError(
                f"Encoder/index dimension mismatch: encoder={encoder.embedding_dim}, "
                f"index={index.metadata.embedding_dim}"
            )
        self.encoder = encoder
        self.index = index
        self.frame_store = frame_store
        self.embedding_cache = embedding_cache
        self.prompt_version = prompt_version
        self.projector = SegmentFrameProjector(
            frame_store,
            max_projection_gap_ms=max_projection_gap_ms,
        )

    @property
    def source_family(self) -> SourceFamily:
        """Share the generic BGE text-family query batch with Context retrieval."""

        return "text"

    def encode(self, query_texts: list[str]) -> QueryEmbeddingBatch:
        """Encode a non-empty text query batch with cache-compatible provenance."""

        return encode_query_batch(
            query_texts,
            self.encoder,
            self.source_family,
            self.embedding_cache,
            self.prompt_version,
        )

    def search_vectors(
        self,
        query_batch: QueryEmbeddingBatch,
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalResult]:
        """Search segments, project hits, deduplicate frames, and assign ASR ranks."""

        del query_type
        self._validate_query_batch(query_batch)
        timer = StageTimer(PipelineStage.SEARCH.value)
        scores, positions = self.index.search_filtered(
            query_batch.vectors,
            top_k,
            filters,
        )
        search_trace = timer.finish(
            input_count=len(query_batch.embeddings),
            output_count=sum(position >= 0 for row in positions for position in row),
            backend="faiss_or_exact_subset",
        )
        return [
            RetrievalResult(
                candidates=self._materialize(
                    scores[row_index], positions[row_index], top_k, filters
                ),
                trace=RetrievalTrace(stages={search_trace.stage: search_trace}),
            )
            for row_index in range(len(query_batch.embeddings))
        ]

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalResult]:
        """Encode and retrieve an ordered query batch without per-query encoding."""

        if not queries:
            return []
        started = perf_counter()
        batch = self.encode(queries)
        results = self.search_vectors(batch, top_k, filters, query_type)
        first_candidate_ms = (perf_counter() - started) * 1_000
        return [
            result.model_copy(
                update={
                    "trace": RetrievalTrace(
                        stages={
                            batch.encoding_trace.stage: batch.encoding_trace,
                            **result.trace.stages,
                        }
                    ),
                    "time_to_first_candidate_ms": first_candidate_ms,
                }
            )
            for result in results
        ]

    def search(
        self,
        query: str,
        top_k: int = 100,
        filters: SearchFilters | None = None,
        query_type: TaskType = TaskType.KIS,
    ) -> RetrievalResult:
        """Preserve the single-query convenience boundary."""

        return self.search_batch([query], top_k, filters, query_type)[0]

    def _validate_query_batch(self, batch: QueryEmbeddingBatch) -> None:
        """Require query vectors compatible with the immutable segment bundle."""

        if batch.model_name != self.index.metadata.model_name:
            raise ValueError("query batch and index model names differ")
        if batch.dimension != self.index.metadata.embedding_dim:
            raise ValueError("query batch and index dimensions differ")
        if batch.source_family != self.source_family:
            raise ValueError("query batch source family is incompatible with index")
        if self.index.metadata.normalization == "l2":
            norms = np.linalg.norm(batch.vectors, axis=1)
            if not batch.normalized or not np.allclose(
                norms, 1.0, rtol=1e-3, atol=1e-4
            ):
                raise ValueError("query batch must contain L2-normalized vectors")

    def _materialize(
        self,
        scores: Any,
        positions: Any,
        top_k: int,
        filters: SearchFilters | None,
    ) -> list[RetrievalCandidate]:
        """Project segment rows and retain one strongest hit per canonical frame."""

        minimum = filters.min_score if filters is not None else None
        selected: dict[
            str,
            tuple[tuple[float, str, int, int, int], RetrievalCandidate],
        ] = {}
        for raw_score, raw_position in zip(scores, positions):
            position = int(raw_position)
            if position < 0:
                continue
            score = float(raw_score)
            if minimum is not None and score < minimum:
                continue
            row = self.index.mapping.iloc[position]
            projection = self.projector.project(
                str(row["video_id"]),
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
            )
            if projection is None:
                continue
            segment_id = str(row["segment_id"])
            selection_key = (
                -score,
                segment_id,
                int(row["start_ms"]),
                int(row["end_ms"]),
                position,
            )
            candidate = _candidate(projection, row, score)
            existing = selected.get(projection.frame_id)
            if existing is None or selection_key < existing[0]:
                selected[projection.frame_id] = (selection_key, candidate)

        ordered = sorted(
            (candidate for _, candidate in selected.values()),
            key=lambda candidate: (
                -candidate.source_scores[RetrievalSource.ASR],
                candidate.frame_id,
            ),
        )[:top_k]
        return [
            candidate.model_copy(
                update={"source_ranks": {RetrievalSource.ASR: rank}}
            )
            for rank, candidate in enumerate(ordered, start=1)
        ]


def _candidate(
    projection: SegmentFrameProjection,
    row: Any,
    score: float,
) -> RetrievalCandidate:
    """Materialize one projected ASR hit without altering canonical identity."""

    return RetrievalCandidate(
        frame_id=projection.frame_id,
        source_scores={RetrievalSource.ASR: score},
        source_ranks={RetrievalSource.ASR: 1},
        metadata={
            "frame": {
                "frame_id": projection.frame_id,
                "video_id": projection.video_id,
                "frame_idx": projection.frame_idx,
                "timestamp_ms": projection.timestamp_ms,
            },
            "asr_segment": {
                "segment_id": str(row["segment_id"]),
                "start_ms": int(row["start_ms"]),
                "end_ms": int(row["end_ms"]),
                "projection_kind": projection.kind,
                "projection_distance_ms": projection.distance_ms,
                "segment_score": score,
            },
        },
    )


__all__ = ["ASRSegmentRetriever"]
