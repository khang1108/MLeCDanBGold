"""Sparse unary candidate-lattice and bounded-order controls."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np

from baseline.contracts import BaselinePath, MethodOptions
from baseline.methods.framewise import (
    _path_from_positions,
    _rank_unique,
    _validate_top_k,
    _validated_scores,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores


def select_event_candidates(
    scores: np.ndarray,
    timestamps_ms: np.ndarray,
    *,
    limit: int,
    min_separation_ms: int,
) -> tuple[np.ndarray, ...]:
    """Select deterministic high-scoring frame positions for every event."""

    matrix = np.asarray(scores)
    timestamps = np.asarray(timestamps_ms)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("scores must be a non-empty two-dimensional matrix")
    if len(timestamps) != matrix.shape[1]:
        raise ValueError("timestamps_ms must match the score columns")
    if not np.isfinite(matrix).all() or not np.isfinite(timestamps).all():
        raise ValueError("scores and timestamps_ms must be finite")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if min_separation_ms < 0:
        raise ValueError("min_separation_ms must be non-negative")

    frame_positions = np.arange(matrix.shape[1], dtype=np.int64)
    selected: list[np.ndarray] = []
    for row in matrix:
        ranked = np.lexsort((frame_positions, -row))
        accepted: list[int] = []
        for raw_position in ranked:
            position = int(raw_position)
            timestamp = float(timestamps[position])
            if any(
                abs(timestamp - float(timestamps[taken])) < min_separation_ms
                for taken in accepted
            ):
                continue
            accepted.append(position)
            if len(accepted) == min(limit, matrix.shape[1]):
                break
        selected.append(np.asarray(sorted(accepted), dtype=np.int64))
    return tuple(selected)


def candidate_orders(
    event_count: int,
    *,
    max_adjacent_swaps: int,
    max_orders: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate permutations by bounded adjacent-swap distance from identity."""

    if event_count <= 0:
        raise ValueError("event_count must be greater than zero")
    if max_adjacent_swaps < 0:
        raise ValueError("max_adjacent_swaps must be non-negative")
    if max_orders <= 0:
        raise ValueError("max_orders must be greater than zero")

    identity = tuple(range(event_count))
    queue: deque[tuple[tuple[int, ...], int]] = deque([(identity, 0)])
    seen = {identity}
    orders: list[tuple[int, ...]] = []
    while queue and len(orders) < max_orders:
        order, distance = queue.popleft()
        orders.append(order)
        if distance == max_adjacent_swaps:
            continue
        for index in range(event_count - 1):
            swapped = list(order)
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            candidate = tuple(swapped)
            if candidate in seen:
                continue
            seen.add(candidate)
            queue.append((candidate, distance + 1))
    return tuple(orders)


class UnaryCandidateLatticeMethod:
    """Decode the strict unary objective over sparse per-event candidates."""

    name = "unary_candidate_lattice"

    def __init__(self, options: MethodOptions) -> None:
        self.options = options

    def rank(
        self,
        videos: Sequence[VideoEventScores],
        *,
        top_k: int,
    ) -> list[BaselinePath]:
        _validate_top_k(top_k)
        paths: list[BaselinePath] = []
        for video in videos:
            scores = _powered_scores(_validated_scores(video), self.options.event_power)
            candidates = select_event_candidates(
                scores,
                video.timestamps_ms,
                limit=self.options.candidates_per_event,
                min_separation_ms=self.options.candidate_min_separation_ms,
            )
            order = tuple(range(scores.shape[0]))
            decoded = _decode_order(
                scores,
                np.asarray(video.timestamps_ms, dtype=np.float64),
                candidates,
                order,
                self.options.lambda_gap,
            )
            if decoded is None:
                continue
            score, chronological_positions = decoded
            paths.append(
                _path_from_positions(
                    video,
                    chronological_positions,
                    score,
                    chronological_event_order=order,
                )
            )
        return _rank_unique(paths, top_k)


class UnaryBoundedOrderMethod:
    """Choose a bounded event permutation, then decode frames chronologically."""

    name = "unary_bounded_order"

    def __init__(self, options: MethodOptions) -> None:
        self.options = options

    def rank(
        self,
        videos: Sequence[VideoEventScores],
        *,
        top_k: int,
    ) -> list[BaselinePath]:
        _validate_top_k(top_k)
        paths: list[BaselinePath] = []
        for video in videos:
            scores = _powered_scores(_validated_scores(video), self.options.event_power)
            candidates = select_event_candidates(
                scores,
                video.timestamps_ms,
                limit=self.options.candidates_per_event,
                min_separation_ms=self.options.candidate_min_separation_ms,
            )
            best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
            for order in candidate_orders(
                scores.shape[0],
                max_adjacent_swaps=self.options.max_adjacent_swaps,
                max_orders=self.options.max_order_candidates,
            ):
                decoded = _decode_order(
                    scores,
                    np.asarray(video.timestamps_ms, dtype=np.float64),
                    candidates,
                    order,
                    self.options.lambda_gap,
                )
                if decoded is None:
                    continue
                raw_score, chronological_positions = decoded
                score = raw_score - self.options.swap_penalty * _inversion_count(order)
                if best is None or score > best[0]:
                    best = score, chronological_positions, order
            if best is None:
                continue
            score, chronological_positions, order = best
            original_positions = [0] * scores.shape[0]
            for position, event in zip(chronological_positions, order, strict=True):
                original_positions[event] = position
            paths.append(
                _path_from_positions(
                    video,
                    tuple(original_positions),
                    score,
                    chronological_event_order=order,
                )
            )
        return _rank_unique(paths, top_k)


def _powered_scores(scores: np.ndarray, event_power: float) -> np.ndarray:
    if event_power == 1.0:
        return scores
    return np.clip(scores, 0.0, None) ** event_power


def _decode_order(
    scores: np.ndarray,
    timestamps_ms: np.ndarray,
    candidates: tuple[np.ndarray, ...],
    order: tuple[int, ...],
    lambda_gap: float,
) -> tuple[float, tuple[int, ...]] | None:
    """Return the exact best chronological path through one sparse lattice."""

    layer_positions = [candidates[event] for event in order]
    first_positions = layer_positions[0]
    current = np.asarray(scores[order[0], first_positions], dtype=np.float64)
    back: list[np.ndarray] = [np.full(len(first_positions), -1, dtype=np.int64)]

    for layer in range(1, len(order)):
        previous_positions = layer_positions[layer - 1]
        positions = layer_positions[layer]
        next_scores = np.full(len(positions), -np.inf, dtype=np.float64)
        predecessors = np.full(len(positions), -1, dtype=np.int64)
        for endpoint_index, endpoint in enumerate(positions):
            valid = np.flatnonzero(previous_positions < endpoint)
            if not len(valid):
                continue
            transition_scores = current[valid] - lambda_gap * (
                timestamps_ms[endpoint] - timestamps_ms[previous_positions[valid]]
            )
            finite = np.isfinite(transition_scores)
            if not finite.any():
                continue
            maximum = float(np.max(transition_scores[finite]))
            tied = valid[np.flatnonzero(transition_scores == maximum)]
            predecessor_index = int(tied[-1])
            predecessors[endpoint_index] = predecessor_index
            next_scores[endpoint_index] = scores[order[layer], endpoint] + maximum
        current = next_scores
        back.append(predecessors)

    if not np.isfinite(current).any():
        return None
    endpoint_index = int(np.argmax(current))
    candidate_indexes = [endpoint_index]
    for layer in range(len(order) - 1, 0, -1):
        endpoint_index = int(back[layer][endpoint_index])
        if endpoint_index < 0:
            return None
        candidate_indexes.append(endpoint_index)
    candidate_indexes.reverse()
    positions = tuple(
        int(layer_positions[layer][candidate_index])
        for layer, candidate_index in enumerate(candidate_indexes)
    )
    return float(np.max(current)), positions


def _inversion_count(order: tuple[int, ...]) -> int:
    return sum(
        left > right
        for index, left in enumerate(order)
        for right in order[index + 1 :]
    )
