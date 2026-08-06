"""Exact candidate-local vector search over persisted normalized arrays."""

from __future__ import annotations

import numpy as np


def exact_subset_search(
    query_vectors: np.ndarray,
    vectors: np.ndarray,
    allowed_positions: np.ndarray,
    top_k: int,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-k positions without searching unrelated vectors."""

    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
    positions = np.asarray(allowed_positions, dtype=np.int64)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if positions.size == 0:
        shape = (len(queries), 0)
        return (
            np.empty(shape, dtype=np.float32),
            np.empty(shape, dtype=np.int64),
        )
    keep = min(top_k, len(positions))
    best_scores = np.empty((len(queries), 0), dtype=np.float32)
    best_positions = np.empty((len(queries), 0), dtype=np.int64)
    for start in range(0, len(positions), max(1, chunk_size)):
        current_positions = positions[start : start + chunk_size]
        current_vectors = np.asarray(vectors[current_positions], dtype=np.float32)
        current_scores = queries @ current_vectors.T
        best_scores, best_positions = _merge_top_k(
            best_scores,
            best_positions,
            current_scores,
            current_positions,
            keep,
        )
    return best_scores, best_positions


def _merge_top_k(
    previous_scores: np.ndarray,
    previous_positions: np.ndarray,
    new_scores: np.ndarray,
    new_positions: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    tiled_positions = np.broadcast_to(new_positions, new_scores.shape)
    scores = np.concatenate((previous_scores, new_scores), axis=1)
    positions = np.concatenate((previous_positions, tiled_positions), axis=1)
    if scores.shape[1] > top_k:
        selected = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        scores = np.take_along_axis(scores, selected, axis=1)
        positions = np.take_along_axis(positions, selected, axis=1)
    ordered_scores = np.empty_like(scores)
    ordered_positions = np.empty_like(positions)
    for row in range(len(scores)):
        order = np.lexsort((positions[row], -scores[row]))
        ordered_scores[row] = scores[row, order]
        ordered_positions[row] = positions[row, order]
    return ordered_scores, ordered_positions
