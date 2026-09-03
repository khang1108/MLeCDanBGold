"""Adapt timestamped ASR segments to canonical frame identity.

This module owns the deterministic segment-to-frame identity projection used
by Dense temporal retrieval. It does not create frame identifiers or mutate
the segment-native ASR artifacts; canonical identity is copied from the
visual :class:`DenseIndex` and timeline selection is delegated to the shared
``SegmentFrameProjector``.
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from hcmai.retrieval.retriever.segment.projector import SegmentFrameProjector

if TYPE_CHECKING:
    from hcmai.retrieval.retriever.dense.index import DenseIndex
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex


class SegmentProjectedASRIndex:
    """Expose segment ASR evidence with canonical frame-shaped identity.

    Every segment receives one precomputed canonical frame position, or ``-1``
    when the shared projector cannot find an eligible frame. The segment
    vectors and mapping remain owned by ``segment_index``; this adapter only
    supplies the canonical identity and projection map required downstream.
    """

    def __init__(
        self,
        *,
        segment_index: SegmentDenseIndex,
        canonical_index: DenseIndex,
        projector: SegmentFrameProjector,
        interval_projection: bool = True,
    ) -> None:
        """Project dense ASR segment vectors to canonical frame coordinates.

        Raises:
            ValueError: If either index has inconsistent identity/vector
                cardinality, duplicate canonical frame IDs, or a projector
                returns a frame ID absent from the canonical index.
        """

        frame_ids = np.asarray(canonical_index.frame_ids)
        video_ids = np.asarray(canonical_index.video_ids)
        frame_idx = np.asarray(canonical_index.frame_idx, dtype=np.int64)
        timestamps = np.asarray(canonical_index.timestamps, dtype=np.int64)
        canonical_lengths = {
            len(frame_ids),
            len(video_ids),
            len(frame_idx),
            len(timestamps),
        }
        if canonical_lengths != {len(frame_ids)}:
            raise ValueError("canonical identity arrays have unequal lengths")
        if not len(frame_ids):
            raise ValueError("canonical identity arrays must be non-empty")

        canonical_frame_ids = [str(frame_id) for frame_id in frame_ids]
        if len(set(canonical_frame_ids)) != len(canonical_frame_ids):
            raise ValueError("canonical frame_id values must be unique")

        mapping = segment_index.mapping
        segment_vectors = segment_index.vectors
        if len(segment_vectors) != len(mapping):
            raise ValueError(
                "segment vector count must equal segment mapping row count"
            )

        self.segment_index = segment_index
        self.frame_ids = frame_ids
        self.video_ids = video_ids
        self.frame_idx = frame_idx
        self.timestamps = timestamps
        self.metadata = segment_index.metadata
        self._segment_vectors = segment_vectors
        self.interval_projection = interval_projection
        self.segment_frame_positions = self._build_segment_frame_positions(
            projector,
            canonical_frame_ids,
        )

        if np.any(
            (self.segment_frame_positions < -1)
            | (self.segment_frame_positions >= len(self.frame_ids))
        ):
            raise ValueError("projected canonical frame positions are out of bounds")

        (
            self.segment_coverage_offsets,
            self.segment_coverage_positions,
            self.coverage_mask,
        ) = self._build_segment_coverage(canonical_index)

    def _build_segment_coverage(
        self,
        canonical_index: DenseIndex,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute canonical frame intervals per segment and total coverage."""

        offsets = [0]
        flattened_positions: list[int] = []
        coverage_mask = np.zeros(len(self.frame_ids), dtype=bool)

        has_video_positions = hasattr(canonical_index, "video_positions") and callable(
            getattr(canonical_index, "video_positions")
        )
        video_positions_cache: dict[str, np.ndarray] = {}

        for segment_position, row in self.segment_index.mapping.iterrows():
            video_id = str(row["video_id"])
            start_ms = int(row["start_ms"])
            end_ms = int(row["end_ms"])

            if video_id in video_positions_cache:
                video_positions = video_positions_cache[video_id]
            else:
                try:
                    if has_video_positions:
                        video_positions = np.asarray(
                            canonical_index.video_positions(video_id),
                            dtype=np.int64,
                        )
                    else:
                        video_positions = np.flatnonzero(self.video_ids == video_id)
                except (KeyError, IndexError):
                    video_positions = np.empty(0, dtype=np.int64)
                video_positions_cache[video_id] = video_positions

            if len(video_positions) > 0:
                video_timestamps = self.timestamps[video_positions]
                inside = video_positions[
                    (video_timestamps >= start_ms) & (video_timestamps <= end_ms)
                ]
            else:
                inside = np.empty(0, dtype=np.int64)

            if self.interval_projection and len(inside) > 0:
                covered_positions = inside.tolist()
            else:
                fallback_pos = int(self.segment_frame_positions[int(segment_position)])
                if fallback_pos >= 0:
                    covered_positions = [fallback_pos]
                else:
                    covered_positions = []

            flattened_positions.extend(covered_positions)
            offsets.append(len(flattened_positions))
            for pos in covered_positions:
                coverage_mask[pos] = True

        return (
            np.asarray(offsets, dtype=np.int64),
            np.asarray(flattened_positions, dtype=np.int64),
            coverage_mask,
        )

    def _build_segment_frame_positions(
        self,
        projector: SegmentFrameProjector,
        canonical_frame_ids: list[str],
    ) -> np.ndarray:
        """Project each segment row to one canonical position or ``-1``."""

        position_by_frame_id = {
            frame_id: position
            for position, frame_id in enumerate(canonical_frame_ids)
        }
        mapped = np.full(len(self.segment_index.mapping), -1, dtype=np.int64)
        for segment_position, row in self.segment_index.mapping.iterrows():
            projection = projector.project(
                str(row["video_id"]),
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
            )
            if projection is None:
                continue
            projected_frame_id = str(projection.frame_id)
            if projected_frame_id not in position_by_frame_id:
                raise ValueError(
                    "projector returned frame_id absent from canonical index: "
                    f"{projected_frame_id!r}"
                )
            canonical_position = position_by_frame_id[projected_frame_id]
            projected_identity = (
                str(projection.video_id),
                int(projection.frame_idx),
                int(projection.timestamp_ms),
            )
            canonical_identity = (
                str(self.video_ids[canonical_position]),
                int(self.frame_idx[canonical_position]),
                int(self.timestamps[canonical_position]),
            )
            if projected_identity != canonical_identity:
                raise ValueError(
                    "projector identity conflicts with canonical index for frame_id "
                    f"{projected_frame_id!r}"
                )
            mapped[int(segment_position)] = canonical_position
        return mapped

    def score_subset(
        self,
        query_vectors: np.ndarray,
        positions: np.ndarray,
        chunk_size: int = 65_536,
    ) -> np.ndarray:
        """Score every ASR segment and scatter maxima onto canonical frames.

        Segment vectors are already L2-normalized, so exact matrix
        multiplication computes cosine similarity. Chunks bound the segment
        vector and score matrices held at once. Coverage across segment
        intervals is scattered with max aggregation; uncovered frames remain 0.0.

        Args:
            query_vectors: One ``[embedding_dim]`` query or a two-dimensional
                ``[event_count, embedding_dim]`` array of finite values.
            positions: One-dimensional integer canonical frame positions to
                return, in the caller's requested order.
            chunk_size: Positive number of segment vectors to score per chunk.

        Returns:
            ``float32`` scores shaped ``[event_count, len(positions)]``.

        Raises:
            ValueError: If queries, positions, or chunk size violate the
                scoring contract.
        """

        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        elif queries.ndim != 2:
            raise ValueError("query_vectors must be one- or two-dimensional")
        expected_dim = int(self.metadata.embedding_dim)
        if queries.shape[1] != expected_dim:
            raise ValueError(
                "query_vectors embedding dimension does not match ASR index: "
                f"expected {expected_dim}, got {queries.shape[1]}"
            )
        if not np.all(np.isfinite(queries)):
            raise ValueError("query_vectors must contain only finite values")
        queries = np.ascontiguousarray(queries, dtype=np.float32)

        raw_positions = np.asarray(positions)
        if raw_positions.ndim != 1:
            raise ValueError("positions must be a one-dimensional integer array")
        if not np.issubdtype(raw_positions.dtype, np.integer):
            raise ValueError("positions must contain integer values")
        positions = np.ascontiguousarray(raw_positions, dtype=np.int64)
        if np.any((positions < 0) | (positions >= len(self.frame_ids))):
            raise ValueError("positions must be within canonical frame bounds")

        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, Integral)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer")
        chunk_size = int(chunk_size)

        frame_scores = np.zeros(
            (len(queries), len(self.frame_ids)),
            dtype=np.float32,
        )
        for start in range(0, len(self._segment_vectors), chunk_size):
            stop = min(start + chunk_size, len(self._segment_vectors))
            vectors = np.asarray(
                self._segment_vectors[start:stop],
                dtype=np.float32,
            )
            chunk_scores = queries @ vectors.T
            for local_segment, global_segment in enumerate(range(start, stop)):
                offset_start = self.segment_coverage_offsets[global_segment]
                offset_stop = self.segment_coverage_offsets[global_segment + 1]
                target_positions = self.segment_coverage_positions[offset_start:offset_stop]
                if not len(target_positions):
                    continue
                for event_index in range(len(queries)):
                    np.maximum.at(
                        frame_scores[event_index],
                        target_positions,
                        chunk_scores[event_index, local_segment],
                    )

        return np.asarray(frame_scores[:, positions], dtype=np.float32)


__all__ = ["SegmentProjectedASRIndex"]
