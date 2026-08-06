"""Dense query encoding and vector-based frame retrieval."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    RetrievalTrace,
    TaskType,
)
from hcmai.common.schemas.search import SearchFilters
from hcmai.common.utils.logging import get_logger
from hcmai.embedding.pipeline import TextEmbeddingAdapter
from hcmai.observability.tracing import StageTimer
from hcmai.retriever.dense.index import DenseIndex
from hcmai.retriever.query_batch import (
    QueryEmbeddingBatch,
    SourceFamily,
    encode_query_batch,
)

logger = get_logger(__name__)


class DenseRetriever:
    """Encode query batches once and search one compatible dense index."""

    def __init__(
        self,
        encoder: TextEmbeddingAdapter,
        index: DenseIndex,
        source: RetrievalSource = RetrievalSource.VISUAL,
    ) -> None:
        if index.metadata.model_name != encoder.config.model_name:
            raise ValueError(
                f"Encoder/index model mismatch: encoder={encoder.config.model_name!r}, "
                f"index={index.metadata.model_name!r}"
            )
        if (
            encoder.embedding_dim
            and index.metadata.embedding_dim != encoder.embedding_dim
        ):
            raise ValueError(
                f"Encoder/index dimension mismatch: encoder={encoder.embedding_dim}, "
                f"index={index.metadata.embedding_dim}"
            )
        self.encoder = encoder
        self.index = index
        self.source = source

    @property
    def source_family(self) -> SourceFamily:
        return "visual" if self.source is RetrievalSource.VISUAL else "text"

    def encode(self, query_texts: list[str]) -> QueryEmbeddingBatch:
        """Encode a non-empty query batch once with full provenance."""

        return encode_query_batch(query_texts, self.encoder, self.source_family)

    def search_vectors(
        self,
        query_batch: QueryEmbeddingBatch,
        top_k: int = 100,
        filters: Optional[SearchFilters] = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalResult]:
        """Search pre-encoded queries with one FAISS batch call."""

        del query_type
        self._validate_query_batch(query_batch)
        mapping = self.index.mapping
        allowed_positions = _allowed_positions(mapping, filters)
        search_k = self.index.index.ntotal if allowed_positions is not None else top_k
        logger.info(
            "FAISS batch search started queries=%d search_k=%d source=%s",
            len(query_batch.embeddings),
            search_k,
            self.source.value,
        )
        timer = StageTimer("index_search")
        scores, positions = self.index.search(query_batch.vectors, search_k)
        search_trace = timer.finish()
        return [
            RetrievalResult(
                candidates=self._materialize(
                    scores[row_index],
                    positions[row_index],
                    top_k,
                    filters,
                    allowed_positions,
                ),
                trace=RetrievalTrace(
                    stages={search_trace.stage: search_trace}
                ),
            )
            for row_index in range(len(query_batch.embeddings))
        ]

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 100,
        filters: Optional[SearchFilters] = None,
        query_type: TaskType = TaskType.KIS,
    ) -> list[RetrievalResult]:
        """Encode and retrieve an ordered batch without per-query calls."""

        if not queries:
            return []
        batch = self.encode(queries)
        results = self.search_vectors(batch, top_k, filters, query_type)
        return [
            result.model_copy(
                update={
                    "trace": RetrievalTrace(
                        stages={
                            batch.encoding_trace.stage: batch.encoding_trace,
                            **result.trace.stages,
                        }
                    )
                }
            )
            for result in results
        ]

    def search(
        self,
        query: str,
        top_k: int = 100,
        filters: Optional[SearchFilters] = None,
        query_type: TaskType = TaskType.KIS,
    ) -> RetrievalResult:
        """Preserve the single-query convenience boundary."""

        return self.search_batch([query], top_k, filters, query_type)[0]

    def _validate_query_batch(self, batch: QueryEmbeddingBatch) -> None:
        if batch.model_name != self.index.metadata.model_name:
            raise ValueError("query batch and index model names differ")
        if batch.dimension != self.index.metadata.embedding_dim:
            raise ValueError("query batch and index dimensions differ")
        if batch.source_family != self.source_family:
            raise ValueError("query batch source family is incompatible with index")
        if self.index.metadata.normalization == "l2":
            norms = np.linalg.norm(batch.vectors, axis=1)
            if not batch.normalized or not np.allclose(
                norms,
                1.0,
                rtol=1e-3,
                atol=1e-4,
            ):
                raise ValueError("query batch must contain L2-normalized vectors")

    def _materialize(
        self,
        scores,
        positions,
        top_k: int,
        filters: SearchFilters | None,
        allowed_positions: set[int] | None,
    ) -> list[RetrievalCandidate]:
        minimum = filters.min_score if filters is not None else None
        candidates: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for raw_score, raw_position in zip(scores, positions):
            position = int(raw_position)
            if position < 0:
                continue
            if allowed_positions is not None and position not in allowed_positions:
                continue
            score = float(raw_score)
            if minimum is not None and score < minimum:
                continue
            row = self.index.mapping.iloc[position]
            frame_id = str(row["frame_id"])
            if frame_id in seen:
                continue
            seen.add(frame_id)
            candidates.append(_candidate(frame_id, row, self.source, score, len(seen)))
            if len(candidates) >= top_k:
                break
        return candidates


def _allowed_positions(
    mapping: pd.DataFrame,
    filters: SearchFilters | None,
) -> set[int] | None:
    if filters is None or not (
        filters.video_ids
        or filters.start_time_ms is not None
        or filters.end_time_ms is not None
    ):
        return None
    mask = pd.Series(True, index=mapping.index)
    if filters.video_ids:
        mask &= mapping["video_id"].isin(filters.video_ids)
    if filters.start_time_ms is not None:
        mask &= mapping["timestamp_ms"] >= filters.start_time_ms
    if filters.end_time_ms is not None:
        mask &= mapping["timestamp_ms"] <= filters.end_time_ms
    return set(mapping.loc[mask, "embedding_index"].tolist())


def _candidate(frame_id, row, source, score, rank) -> RetrievalCandidate:
    return RetrievalCandidate(
        frame_id=frame_id,
        source_scores={source: score},
        source_ranks={source: rank},
        metadata={
            "frame": {
                "frame_id": frame_id,
                "video_id": row.get("video_id", ""),
                "frame_idx": int(row.get("frame_idx", 0)),
                "timestamp_ms": int(row.get("timestamp_ms", 0)),
            }
        },
    )
