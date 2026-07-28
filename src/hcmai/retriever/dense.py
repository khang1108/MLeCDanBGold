"""Online dense retriever implementing the SearchEngine retriever contract."""

from __future__ import annotations

import pandas as pd
from typing import Optional

from hcmai.common.schemas import RetrievalSource
from hcmai.common.schemas.search import SearchFilters
from hcmai.common.schemas.retrieval import RetrievalCandidate
from hcmai.common.utils.logging import get_logger
from hcmai.common.utils.timing import Timer
from hcmai.retriever.encoder import TextEncoder
from hcmai.retriever.index import VisualIndex

logger = get_logger(__name__)


class DenseRetriever:
    """Encode a text query, search the FAISS index, and return candidates.

    The retriever exposes the ``search(query, top_k, filters)`` contract that
    :class:`hcmai.search.SearchEngine` depends on, returning shared
    :class:`RetrievalCandidate` objects rather than raw FAISS tuples.
    """

    def __init__(self, encoder: TextEncoder, index: VisualIndex) -> None:
        """Pair an encoder with an index and reject mismatched artifacts.

        Raises:
            ValueError: If the index was built by a different model or with a
                different embedding dimension than the encoder produces.
        """
        # Coupling model and index versions guards against silently searching
        # an index whose vectors are incompatible with the query encoder.
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
        # Query-encoding and index-search latency are recorded separately so
        # callers can attribute cost to the encoder versus the search.
        self.last_query_encoding_ms = 0.0
        self.last_index_search_ms = 0.0

    def search(
        self, query: str, top_k: int = 100, filters: Optional[SearchFilters] = None
    ) -> list[RetrievalCandidate]:
        """Retrieve the top matching frames for a text query.

        Args:
            query: Natural-language search text.
            top_k: Maximum number of candidates to return.
            filters: Optional video/time/score restrictions.

        Returns:
            Score-sorted, deduplicated candidates with a visual score and rank.
        """

        # Get frames mapping of the FAISS VectorDB
        mapping = self.index.mapping

        # Encode the query to a single normalized vector, timed on its own.
        encode_timer = Timer()
        query_vector = self.encoder.encode_text(
            [query]
        )  # Embed the query to get embedding vector

        self.last_query_encoding_ms = encode_timer.stop()
        # Restrict to positions allowed by video/time filters. When such a
        # filter is active we must scan the whole exact index to guarantee a
        # full top_k after filtering; otherwise top_k neighbours suffice.
        allowed_positions: set[int] | None = None
        if filters is not None and (
            filters.video_ids
            or filters.start_time_ms is not None
            or filters.end_time_ms is not None
        ):
            mask = pd.Series(True, index=mapping.index)
            if filters.video_ids:
                mask &= mapping["video_id"].isin(filters.video_ids)
            if filters.start_time_ms is not None:
                mask &= mapping["timestamp_ms"] >= filters.start_time_ms
            if filters.end_time_ms is not None:
                mask &= mapping["timestamp_ms"] <= filters.end_time_ms
            allowed_positions = set(mapping.loc[mask, "embedding_index"].tolist())

        search_k = self.index.index.ntotal if allowed_positions is not None else top_k

        # Search the index, timed on its own.
        logger.info(
            "FAISS search started search_k=%d filtered_positions=%s",
            search_k,
            len(allowed_positions) if allowed_positions is not None else "all",
        )
        search_timer = Timer()
        scores, positions = self.index.search(query_vector, search_k)
        self.last_index_search_ms = search_timer.stop()

        min_score = filters.min_score if filters is not None else None

        candidates: list[RetrievalCandidate] = []
        seen_frame_ids: set[str] = set()
        for score, position in zip(scores[0], positions[0]):
            if position < 0:  # FAISS pads short result rows with -1.
                continue
            if allowed_positions is not None and position not in allowed_positions:
                continue
            score = float(score)
            if min_score is not None and score < min_score:
                continue

            row = mapping.iloc[int(position)]
            frame_id = row["frame_id"]
            if frame_id in seen_frame_ids:
                continue
            seen_frame_ids.add(frame_id)

            candidates.append(
                RetrievalCandidate(
                    frame_id=frame_id,
                    source_scores={RetrievalSource.VISUAL: score},
                    source_ranks={RetrievalSource.VISUAL: len(candidates) + 1},
                    metadata={
                        "frame": {
                            "frame_id": frame_id,
                            "video_id": row.get("video_id", ""),
                            "frame_idx": int(row.get("frame_idx", 0)),
                            "timestamp_ms": int(row.get("timestamp_ms", 0)),
                        }
                    },
                )
            )
            if len(candidates) >= top_k:
                break

        logger.info(
            f"Retrieved {len(candidates)} candidates for query "
            f"(encode={self.last_query_encoding_ms:.1f}ms, "
            f"search={self.last_index_search_ms:.1f}ms)"
        )
        return candidates
