"""Dense query encoding and vector-based frame retrieval."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from hcmai.common.observability import PipelineStage, RetrievalTrace
from hcmai.retrieval.models import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
)
from hcmai.common.utils.logging import get_logger
from hcmai.retrieval.embedding.pipeline import TextEmbeddingAdapter
from hcmai.common.observability.tracing import StageTimer
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.cache import EmbeddingCache
from hcmai.retrieval.retriever.query_batch import (
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
        embedding_cache: EmbeddingCache | None = None,
        prompt_version: str = "query-v1",
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
        self.embedding_cache = embedding_cache
        self.prompt_version = prompt_version

    @property
    def source_family(self) -> SourceFamily:
        return "visual" if self.source is RetrievalSource.VISUAL else "text"

    def encode(self, query_texts: list[str]) -> QueryEmbeddingBatch:
        """Encode a non-empty query batch once with full provenance."""

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
    ) -> list[RetrievalResult]:
        """Search pre-encoded queries with one FAISS batch call."""

        self._validate_query_batch(query_batch)
        logger.info(
            "FAISS batch search started queries=%d search_k=%d source=%s",
            len(query_batch.embeddings),
            top_k,
            self.source.value,
        )
        timer = StageTimer(PipelineStage.SEARCH.value)
        scores, positions = self.index.search(query_batch.vectors, top_k)
        search_trace = timer.finish(
            input_count=len(query_batch.embeddings),
            output_count=sum(position >= 0 for row in positions for position in row),
            backend="faiss",
        )
        return [
            RetrievalResult(
                candidates=self._materialize(
                    scores[row_index],
                    positions[row_index],
                    top_k,
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
    ) -> list[RetrievalResult]:
        """Encode and retrieve an ordered batch without per-query calls."""

        if not queries:
            return []
        started = perf_counter()
        batch = self.encode(queries)
        results = self.search_vectors(batch, top_k)
        first_candidate_ms = (perf_counter() - started) * 1_000
        return [
            replace(
                result,
                trace=RetrievalTrace(
                    stages={
                        batch.encoding_trace.stage: batch.encoding_trace,
                        **result.trace.stages,
                    }
                ),
                time_to_first_candidate_ms=first_candidate_ms,
            )
            for result in results
        ]

    def search(
        self,
        query: str,
        top_k: int = 100,
    ) -> RetrievalResult:
        """Preserve the single-query convenience boundary."""

        return self.search_batch([query], top_k)[0]

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
    ) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for raw_score, raw_position in zip(scores, positions):
            position = int(raw_position)
            if position < 0:
                continue
            score = float(raw_score)
            row = self.index.mapping.iloc[position]
            frame_id = str(row["frame_id"])
            if frame_id in seen:
                continue
            seen.add(frame_id)
            candidates.append(_candidate(frame_id, row, self.source, score, len(seen)))
            if len(candidates) >= top_k:
                break
        return candidates


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
