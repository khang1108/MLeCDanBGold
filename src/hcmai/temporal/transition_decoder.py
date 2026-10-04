"""Motion-Aware Graph Decoding for ordered event-to-frame alignment.

Extends monotonic temporal dynamic programming with pairwise candidate transition
compatibility matrices :math:`\\psi(s, t; E_{i-1}, E_i)` representing motion consistency,
action flow continuity, or cross-event transition affinity.

Operates over a sparse candidate lattice of top-K candidate frames per event layer,
decoding optimal chronological paths in :math:`O(M \\cdot K^2)` complexity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import (
    DPPath,
    _prepare_dp_inputs,
)


@runtime_checkable
class FrameEmbeddingSource(Protocol):
    """Protocol for visual frame embedding sources."""

    def get_frame_embeddings(
        self,
        video_id: str,
        frame_indices: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        """Retrieve stored visual embeddings for candidate frames of one video.

        Args:
            video_id: Canonical video identifier.
            frame_indices: 0-based column/temporal indices within the video's
                canonical frame order (as used in VideoEventScores and EventCandidateLayer).

        Returns:
            np.ndarray of shape (len(frame_indices), embedding_dim), dtype float32.
        """
        ...


@runtime_checkable
class TextEmbeddingSource(Protocol):
    """Protocol for text embedding encoders."""

    def encode_text(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Encode an ordered sequence of event queries into embeddings.

        Args:
            texts: List of event query strings.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32.
        """
        ...


@dataclass(frozen=True, slots=True)
class EventCandidateLayer:
    """Top-K candidate frame coordinate and unary score layer for one event."""

    event_index: int
    frame_indices: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        idx = np.asarray(self.frame_indices, dtype=np.int64)
        sc = np.asarray(self.scores, dtype=np.float64)
        if idx.ndim != 1 or sc.ndim != 1:
            raise ValueError("frame_indices and scores must be 1D arrays")
        if len(idx) != len(sc):
            raise ValueError("frame_indices and scores must have the same length")
        idx.setflags(write=False)
        sc.setflags(write=False)
        object.__setattr__(self, "frame_indices", idx)
        object.__setattr__(self, "scores", sc)


def select_event_candidates(
    video: VideoEventScores,
    *,
    candidate_k: int,
    allowed: np.ndarray | None = None,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
) -> tuple[EventCandidateLayer, ...]:
    """Select top-K candidate frames per event with causal feasibility.

    Event 0 does not require predecessor reachability (valid if finite).
    Events i > 0 require predecessor reachability (source >= 0) and finite scores.
    """
    if candidate_k <= 0:
        raise ValueError(f"candidate_k must be positive, got {candidate_k}")

    prep = _prepare_dp_inputs(video, allowed, event_power, cluster_delta)
    if prep is None:
        return ()
    scores, frames, starts, source, reachable = prep
    n_events, n_frames = scores.shape

    layers: list[EventCandidateLayer] = []
    for event in range(n_events):
        ev_scores = scores[event]
        if event == 0:
            valid_mask = np.isfinite(ev_scores)
        else:
            valid_mask = reachable & np.isfinite(ev_scores)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return ()
        k = min(candidate_k, len(valid_idx))
        top_k = valid_idx[np.argpartition(-ev_scores[valid_idx], k - 1)[:k]]
        sorted_indices = np.sort(top_k)
        layers.append(
            EventCandidateLayer(
                event_index=event,
                frame_indices=sorted_indices,
                scores=ev_scores[sorted_indices],
            )
        )
    return tuple(layers)


class DictFrameEmbeddingSource:
    """In-memory dictionary visual embedding source conforming to FrameEmbeddingSource."""

    def __init__(self, embeddings: Mapping[str, np.ndarray]) -> None:
        self._embeddings = embeddings

    def get_frame_embeddings(
        self,
        video_id: str,
        frame_indices: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        """Retrieve stored visual embeddings for candidate frames of one video."""
        if video_id not in self._embeddings:
            raise KeyError(f"Video {video_id!r} not found in visual embedding source")
        video_mat = self._embeddings[video_id]
        indices = np.asarray(frame_indices, dtype=np.int64)
        if len(indices) == 0:
            dim = video_mat.shape[1] if video_mat.ndim > 1 else 0
            return np.empty((0, dim), dtype=np.float32)
        if np.any(indices < 0) or np.any(indices >= len(video_mat)):
            raise IndexError(
                f"frame_indices out of bounds for video {video_id!r} with {len(video_mat)} frames"
            )
        return np.asarray(video_mat[indices], dtype=np.float32)

    def __call__(
        self,
        video_id: str,
        frame_indices: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        return self.get_frame_embeddings(video_id, frame_indices)


class FrameEmbeddingAccessor:
    """Provides visual embedding lookup conforming to FrameEmbeddingSource."""

    def __init__(self, source: FrameEmbeddingSource | Mapping[str, np.ndarray] | np.ndarray) -> None:
        if isinstance(source, np.ndarray):
            self._source: FrameEmbeddingSource = DictFrameEmbeddingSource({"": source})
            self._single_mat: np.ndarray | None = source
        elif isinstance(source, Mapping):
            self._source = DictFrameEmbeddingSource(source)
            self._single_mat = None
        else:
            self._source = source
            self._single_mat = None

    def get_frame_embeddings(
        self,
        video_id: str,
        frame_indices: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        if self._single_mat is not None:
            indices = np.asarray(frame_indices, dtype=np.int64)
            return np.asarray(self._single_mat[indices], dtype=np.float32)
        return self._source.get_frame_embeddings(video_id, frame_indices)

    def __call__(
        self,
        video_id: str,
        frame_indices: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        return self.get_frame_embeddings(video_id, frame_indices)


def encode_query_events(
    events: Sequence[str],
    encoder: TextEmbeddingSource | None = None,
) -> np.ndarray:
    """Encode an ordered sequence of event query strings into shared vision-language space.

    Encodes all event texts in a single batch once per query, returning an (M, D) matrix
    of L2-normalized vectors.

    Args:
        events: Sequence of M event text queries.
        encoder: A text encoder implementing TextEmbeddingSource.

    Returns:
        np.ndarray of shape (M, D) with float32 L2-normalized embeddings.
    """
    if len(events) == 0:
        return np.empty((0, 0), dtype=np.float32)

    if encoder is None:
        raise ValueError("encoder must be provided to encode_query_events")

    if not hasattr(encoder, "encode_text"):
        raise TypeError(f"encoder must implement TextEmbeddingSource with encode_text(), got {type(encoder)}")

    embeddings = np.asarray(encoder.encode_text(list(events)), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(events):
        raise ValueError(
            f"Expected encoder to return shape ({len(events)}, D), got {embeddings.shape}"
        )

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


@dataclass(frozen=True, slots=True)
class EmbeddingDeltaTransitionScorer:
    r"""Query-conditioned signed transition scorer based on embedding differences.

    Evaluates directional semantic visual change aligned with the semantic transition
    between consecutive textual events:
    .. math::

        \psi_i(a, b) = \frac{1}{4} (v_b - v_a)^T (q_{i+1} - q_i)

    Computed efficiently in :math:`O((K_a + K_b)D + K_a K_b)` without materializing
    intermediate 3D tensors:
    .. math::

        s_A = A \cdot \Delta q, \quad s_B = B \cdot \Delta q
        \Psi = \frac{1}{4} (s_B[None, :] - s_A[:, None])

    Transition scores are strictly signed (never clamped to non-negative) so that
    reversed semantic transitions receive negative penalties.
    """

    def compute_transition_matrix(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        query_delta: np.ndarray,
    ) -> np.ndarray:
        """Compute (Ka, Kb) transition compatibility matrix.

        Args:
            source_embeddings: Array of shape (Ka, D) of visual embeddings for event i.
            target_embeddings: Array of shape (Kb, D) of visual embeddings for event i+1.
            query_delta: Vector of shape (D,) representing q_{i+1} - q_i.

        Returns:
            np.ndarray of shape (Ka, Kb) with signed transition affinity scores.
        """
        A = np.asarray(source_embeddings, dtype=np.float64)
        B = np.asarray(target_embeddings, dtype=np.float64)
        dQ = np.asarray(query_delta, dtype=np.float64).reshape(-1)

        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"source ({A.shape}) and target ({B.shape}) embeddings must be 2D")
        if A.shape[1] != B.shape[1] or A.shape[1] != len(dQ):
            raise ValueError(
                f"Dimension mismatch: source D={A.shape[1]}, target D={B.shape[1]}, query_delta D={len(dQ)}"
            )

        # Normalize visual vectors if not normalized
        a_norms = np.linalg.norm(A, axis=-1, keepdims=True)
        A_norm = A / np.maximum(a_norms, 1e-12)
        b_norms = np.linalg.norm(B, axis=-1, keepdims=True)
        B_norm = B / np.maximum(b_norms, 1e-12)

        # Efficient O((Ka + Kb)D + Ka * Kb) matrix computation
        s_A = A_norm @ dQ   # shape (Ka,)
        s_B = B_norm @ dQ   # shape (Kb,)

        # psi = 1/4 * (s_B[None, :] - s_A[:, None])
        return 0.25 * (s_B[None, :] - s_A[:, None])

    def score_all_transitions(
        self,
        candidate_embeddings: Sequence[np.ndarray],
        query_embeddings: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute transition matrices for all consecutive event pairs 0..M-2.

        Args:
            candidate_embeddings: Sequence of length M, each of shape (K_i, D).
            query_embeddings: Array of shape (M, D) with SigLIP text embeddings.

        Returns:
            List of M-1 transition matrices of shape (K_{i}, K_{i+1}).
        """
        n_events = len(candidate_embeddings)
        if len(query_embeddings) != n_events:
            raise ValueError(
                f"Query embeddings ({len(query_embeddings)}) must match event count ({n_events})"
            )
        transitions: list[np.ndarray] = []
        for i in range(n_events - 1):
            dQ = query_embeddings[i + 1] - query_embeddings[i]
            mat = self.compute_transition_matrix(
                candidate_embeddings[i],
                candidate_embeddings[i + 1],
                query_delta=dQ,
            )
            transitions.append(mat)
        return transitions


def _validate_and_extract_lattice_edge(
    matrix: np.ndarray | None,
    k_prev: int,
    k_curr: int,
    weight: float,
) -> np.ndarray | None:
    """Validate 2D transition matrix and apply transition weight."""
    if matrix is None:
        return None
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"Transition matrix must be 2D, got shape {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("Transition matrix contains non-finite values (NaN or Inf)")
    if mat.shape != (k_prev, k_curr):
        raise ValueError(
            f"Expected transition matrix of shape ({k_prev}, {k_curr}), got {mat.shape}"
        )
    return mat * weight


def decode_candidate_lattice(
    video: VideoEventScores,
    *,
    candidates: Sequence[EventCandidateLayer],
    transition_scores: Sequence[np.ndarray] | None = None,
    transition_weight: float = 1.0,
    lambda_gap: float = 1e-5,
    paths: int = 1,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
    min_separation_ms: int = 0,
    allowed: np.ndarray | None = None,
) -> list[DPPath]:
    """Decode optimal chronological paths through a candidate lattice per event.

    Reduces computational complexity from :math:`O(M \\cdot F^2)` to :math:`O(M \\cdot K^2)`,
    enabling millisecond-level motion-aware graph decoding over long videos with thousands
    of keyframes.

    Requires explicit candidate layers :math:`C_1 \\to C_2 \\to \\dots \\to C_M` (e.g. from
    ``select_event_candidates``) and optional :math:`(K_{i-1}, K_i)` transition matrices.
    """
    prep = _prepare_dp_inputs(video, allowed, event_power, cluster_delta)
    if prep is None:
        return []
    scores, frames, starts, source, reachable = prep
    n_events, n_frames = scores.shape

    if len(candidates) != n_events:
        raise ValueError(
            f"Number of candidate layers ({len(candidates)}) must match number of events ({n_events})"
        )

    # Ensure every layer has at least one candidate frame
    for layer in candidates:
        if len(layer.frame_indices) == 0:
            return []

    # Initialize DP on first event candidates
    c0 = np.asarray(candidates[0].frame_indices, dtype=np.int64)
    dp_vals = scores[0, c0].copy()
    back_pointers: list[np.ndarray] = []

    for event in range(1, n_events):
        edge_idx = event - 1
        c_prev = np.asarray(candidates[event - 1].frame_indices, dtype=np.int64)
        c_curr = np.asarray(candidates[event].frame_indices, dtype=np.int64)

        # Validate transition matrix with exact (K_{i-1}, K_i) shape
        edge_sub_mat: np.ndarray | None = None
        if transition_scores is not None and edge_idx < len(transition_scores):
            edge_sub_mat = _validate_and_extract_lattice_edge(
                transition_scores[edge_idx],
                k_prev=len(c_prev),
                k_curr=len(c_curr),
                weight=transition_weight,
            )

        # Chronological predecessor condition: frame s strictly precedes frame t
        s_indices = c_prev[:, None]          # shape (K_prev, 1)
        t_sources = source[c_curr][None, :]    # shape (1, K_curr)
        t_reachable = reachable[c_curr][None, :]  # shape (1, K_curr)
        valid_pred = t_reachable & (s_indices <= t_sources)

        # Time gap penalty: lambda_gap * (t_time - s_time)
        s_times = np.asarray(video.timestamps_ms, dtype=np.float64)[c_prev][:, None]
        t_times = np.asarray(video.timestamps_ms, dtype=np.float64)[c_curr][None, :]
        gap_penalty = lambda_gap * (t_times - s_times)

        # Transition affinity
        trans_term = edge_sub_mat if edge_sub_mat is not None else 0.0

        pair_scores = dp_vals[:, None] + trans_term - gap_penalty
        masked_pair = np.where(valid_pred & np.isfinite(dp_vals)[:, None], pair_scores, -np.inf)

        best_s_local = np.argmax(masked_pair, axis=0)
        best_val = np.max(masked_pair, axis=0)

        dp_vals = np.where(np.isfinite(best_val), scores[event, c_curr] + best_val, -np.inf)
        back_pointers.append(best_s_local)

    # Reconstruct ranked paths from final candidate layer
    c_last = np.asarray(candidates[-1].frame_indices, dtype=np.int64)
    timestamps = np.asarray(video.timestamps_ms, dtype=np.int64)
    results: list[DPPath] = []
    accepted: list[int] = []

    for local_endpoint in np.argsort(-dp_vals):
        if len(results) >= paths:
            break
        if not np.isfinite(dp_vals[local_endpoint]):
            break
        global_pos = int(c_last[local_endpoint])
        if any(
            abs(int(timestamps[global_pos]) - taken) < min_separation_ms
            for taken in accepted
        ):
            continue
        accepted.append(int(timestamps[global_pos]))

        path = [global_pos]
        curr_local = int(local_endpoint)
        for event in range(n_events - 1, 0, -1):
            prev_local = int(back_pointers[event - 1][curr_local])
            prev_global = int(candidates[event - 1].frame_indices[prev_local])
            path.append(prev_global)
            curr_local = prev_local

        ordered = tuple(reversed(path))
        results.append(
            DPPath(
                video_id=video.video_id,
                score=float(dp_vals[local_endpoint]),
                frame_idx=tuple(int(video.frame_idx[pos]) for pos in ordered),
                frame_ids=tuple(str(video.frame_ids[pos]) for pos in ordered),
            )
        )

    return results


def decode_motion_graph_video(
    video: VideoEventScores,
    *,
    frame_embeddings: FrameEmbeddingSource | Any,
    event_embeddings: np.ndarray,
    candidate_k: int = 32,
    transition_weight: float = 1.0,
    transition_scorer: EmbeddingDeltaTransitionScorer | None = None,
    lambda_gap: float = 1e-5,
    paths: int = 1,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
    min_separation_ms: int = 0,
    allowed: np.ndarray | None = None,
) -> list[DPPath]:
    """End-to-end motion-aware graph decoding for a single video.

    Executes the 4-stage pipeline:
        1. Select candidate layers C_1, ..., C_M via unary score and causal reachability.
        2. Retrieve stored visual embeddings for all candidate frames.
        3. Score query-conditioned directional transitions psi(s, t) across consecutive layers.
        4. Decode the optimal chronological path through the sparse candidate lattice.

    Args:
        video: VideoEventScores holding per-event frame similarities and timestamps.
        frame_embeddings: Source conforming to FrameEmbeddingSource (DenseIndex,
            DictFrameEmbeddingSource, FrameEmbeddingAccessor, or dict).
        event_embeddings: Array of shape (M, D) with SigLIP query-event embeddings.
        candidate_k: Maximum candidate frames per event layer.
        transition_weight: Scaling weight beta for transition edges.
        transition_scorer: Optional custom transition scorer (defaults to EmbeddingDeltaTransitionScorer()).
        lambda_gap: Linear time gap penalty weight.
        paths: Maximum number of ranked non-overlapping paths.
        event_power: Power scaling applied to unary scores.
        cluster_delta: Score drift threshold for frame clustering.
        min_separation_ms: Minimum timestamp separation between returned paths.
        allowed: Optional boolean admissibility mask.

    Returns:
        Ranked list of DPPath instances.
    """
    n_events, n_frames = video.scores.shape

    # 1. Candidate Selection
    candidates = select_event_candidates(
        video,
        candidate_k=candidate_k,
        allowed=allowed,
        event_power=event_power,
        cluster_delta=cluster_delta,
    )
    if len(candidates) != n_events or any(len(c.frame_indices) == 0 for c in candidates):
        return []

    # 2. Fetch Candidate Embeddings
    if isinstance(frame_embeddings, np.ndarray):
        source: FrameEmbeddingSource = DictFrameEmbeddingSource({video.video_id: frame_embeddings})
    elif isinstance(frame_embeddings, Mapping):
        source = DictFrameEmbeddingSource(frame_embeddings)
    elif hasattr(frame_embeddings, "get_frame_embeddings"):
        source = frame_embeddings
    else:
        raise TypeError(
            f"frame_embeddings must implement FrameEmbeddingSource, got {type(frame_embeddings)}"
        )

    cand_embs = [
        source.get_frame_embeddings(video.video_id, layer.frame_indices)
        for layer in candidates
    ]

    # 3. Score Query-Conditioned Transitions
    transitions: list[np.ndarray] | None = None
    if transition_weight != 0.0:
        scorer = (
            transition_scorer
            if transition_scorer is not None
            else EmbeddingDeltaTransitionScorer()
        )
        transitions = scorer.score_all_transitions(
            cand_embs,
            event_embeddings,
        )

    # 4. Decode Candidate Lattice
    return decode_candidate_lattice(
        video,
        candidates=candidates,
        transition_scores=transitions,
        transition_weight=transition_weight,
        lambda_gap=lambda_gap,
        paths=paths,
        event_power=event_power,
        cluster_delta=cluster_delta,
        min_separation_ms=min_separation_ms,
        allowed=allowed,
    )


def rank_motion_graph_paths(
    videos: Sequence[VideoEventScores],
    *,
    frame_embeddings: FrameEmbeddingSource | Any,
    event_embeddings: np.ndarray,
    candidate_k: int = 32,
    transition_weight: float = 1.0,
    transition_scorer: EmbeddingDeltaTransitionScorer | None = None,
    lambda_gap: float = 1e-5,
    max_rows: int = 100,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
    paths_per_video: int = 1,
    path_min_separation_ms: int = 0,
) -> list[DPPath]:
    """Rank bounded paths using motion-aware graph decoding with video-level diversification.

    Executes decode_motion_graph_video per video and sorts results according to the
    configured paths_per_video and level-wise diversification rule.
    """
    if not videos:
        return []

    import math

    depth = max(paths_per_video, math.ceil(max_rows / len(videos)))
    per_video = [
        decode_motion_graph_video(
            video,
            frame_embeddings=frame_embeddings,
            event_embeddings=event_embeddings,
            candidate_k=candidate_k,
            transition_weight=transition_weight,
            transition_scorer=transition_scorer,
            lambda_gap=lambda_gap,
            paths=depth,
            event_power=event_power,
            cluster_delta=cluster_delta,
            min_separation_ms=path_min_separation_ms,
        )
        for video in videos
    ]

    if paths_per_video > 1:
        return sorted(
            (path for paths in per_video for path in paths),
            key=lambda path: path.score,
            reverse=True,
        )[:max_rows]

    rows: list[DPPath] = []
    for level in range(depth):
        rows.extend(
            sorted(
                (paths[level] for paths in per_video if len(paths) > level),
                key=lambda path: path.score,
                reverse=True,
            )
        )
        if len(rows) >= max_rows:
            break
    return rows[:max_rows]


__all__ = [
    "DPPath",
    "DictFrameEmbeddingSource",
    "EmbeddingDeltaTransitionScorer",
    "EventCandidateLayer",
    "FrameEmbeddingAccessor",
    "FrameEmbeddingSource",
    "TextEmbeddingSource",
    "decode_candidate_lattice",
    "decode_motion_graph_video",
    "encode_query_events",
    "rank_motion_graph_paths",
    "select_event_candidates",
]
