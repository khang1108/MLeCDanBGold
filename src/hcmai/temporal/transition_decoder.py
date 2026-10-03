"""Motion-Aware Graph Decoding for ordered event-to-frame alignment.

Extends monotonic temporal dynamic programming with pairwise candidate transition
compatibility matrices :math:`\\psi(s, t; E_{i-1}, E_i)` representing motion consistency,
action flow continuity, or cross-event transition affinity.

When transition edge weights are absent or zero, this recurrence is mathematically
and numerically equivalent to the frozen baseline dynamic programming in
``hcmai.temporal.dp``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import (
    AlignedPath,
    ConditionedDPPath,
    DPPath,
    _prepare_dp_inputs,
    align_video,
)


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


class FrameEmbeddingAccessor:
    """Provides visual embedding lookup for candidate frames of videos.

    Wraps a DenseIndex, a per-video embedding dictionary, or an arbitrary callable,
    guaranteeing that candidate frame indices always resolve to the exact stored
    visual vectors in canonical order without decoding or re-encoding video.
    """

    def __init__(self, source: Any) -> None:
        self._source = source

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
        if hasattr(self._source, "get_frame_embeddings"):
            return self._source.get_frame_embeddings(video_id, frame_indices)
        if isinstance(self._source, dict):
            if video_id not in self._source:
                raise KeyError(f"Video {video_id!r} not found in visual embedding source")
            video_mat = self._source[video_id]
            indices = np.asarray(frame_indices, dtype=np.int64)
            if len(indices) == 0:
                dim = video_mat.shape[1] if video_mat.ndim > 1 else 0
                return np.empty((0, dim), dtype=np.float32)
            if np.any(indices < 0) or np.any(indices >= len(video_mat)):
                raise IndexError(
                    f"frame_indices out of bounds for video {video_id!r} with {len(video_mat)} frames"
                )
            return np.asarray(video_mat[indices], dtype=np.float32)
        if callable(self._source):
            return self._source(video_id, frame_indices)
        raise TypeError(f"Unsupported visual embedding source type: {type(self._source)}")

    def __call__(
        self,
        video_id: str,
        frame_indices: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        return self.get_frame_embeddings(video_id, frame_indices)


def encode_query_events(
    events: Sequence[str],
    encoder: Any | None = None,
) -> np.ndarray:
    """Encode an ordered sequence of event query strings into shared vision-language space.

    Encodes all event texts in a single batch once per query, returning an (M, D) matrix
    of L2-normalized vectors.

    Args:
        events: Sequence of M event text queries.
        encoder: An optional text encoder. If None, instantiates a SigLIPAdapter.

    Returns:
        np.ndarray of shape (M, D) with float32 L2-normalized embeddings.
    """
    if len(events) == 0:
        return np.empty((0, 0), dtype=np.float32)

    event_list = list(events)
    if encoder is None:
        from hcmai.common.config import EncoderConfig
        from hcmai.retrieval.embedding.adapters.siglip import SigLIPAdapter

        config = EncoderConfig(backend="siglip", model_name="google/siglip2-base-patch16-224")
        encoder = SigLIPAdapter(config)

    if hasattr(encoder, "encode_text"):
        embeddings = encoder.encode_text(event_list)
    elif hasattr(encoder, "encode"):
        embeddings = encoder.encode(event_list)
    elif callable(encoder):
        embeddings = encoder(event_list)
    else:
        raise TypeError(f"Unsupported encoder type: {type(encoder)}")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(events):
        raise ValueError(
            f"Expected encoder to return shape ({len(events)}, D), got {embeddings.shape}"
        )

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


@dataclass(frozen=True, slots=True)
class TransitionEdgeMatrix:
    """Pairwise transition compatibility scores between frames for adjacent events."""

    source_event_index: int
    target_event_index: int
    matrix: np.ndarray
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class MotionCosineTransitionScorer:
    """Pairwise visual/motion transition affinity based on feature cosine similarity.

    Attributes:
        temporal_horizon_ms: Maximum temporal gap (in milliseconds) between events
            beyond which transition affinity is truncated to 0.0.
        similarity_power: Exponent applied to positive cosine similarity to sharpen transitions.
    """

    temporal_horizon_ms: float = 30000.0
    similarity_power: float = 1.0

    def compute_transition_matrix(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_timestamps_ms: np.ndarray,
        target_timestamps_ms: np.ndarray,
    ) -> np.ndarray:
        """Compute (N_source, N_target) transition affinity matrix psi(s, t)."""
        src = np.asarray(source_features, dtype=np.float64)
        tgt = np.asarray(target_features, dtype=np.float64)
        src_norm = src / np.maximum(np.linalg.norm(src, axis=-1, keepdims=True), 1e-12)
        tgt_norm = tgt / np.maximum(np.linalg.norm(tgt, axis=-1, keepdims=True), 1e-12)

        cos_sim = src_norm @ tgt_norm.T

        # Temporal delta matrix: target timestamp minus source timestamp
        src_t = np.asarray(source_timestamps_ms, dtype=np.float64)[:, None]
        tgt_t = np.asarray(target_timestamps_ms, dtype=np.float64)[None, :]
        delta_t = tgt_t - src_t

        # Forward chronological mask and horizon window
        valid_temporal = (delta_t > 0.0) & (delta_t <= self.temporal_horizon_ms)

        # Non-negative normalized similarity
        sim_clamped = np.maximum(0.0, cos_sim)
        if self.similarity_power != 1.0:
            sim_clamped = np.power(sim_clamped, self.similarity_power)

        psi = np.where(valid_temporal, sim_clamped, 0.0)
        return psi


@dataclass(frozen=True, slots=True)
class EmbeddingDeltaTransitionScorer:
    """Query-conditioned signed transition scorer based on embedding differences.

    Evaluates directional semantic visual change aligned with the semantic transition
    between consecutive textual events:
    .. math::

        \\psi_i(a, b) = \\frac{1}{4} (v_b - v_a)^T (q_{i+1} - q_i)

    Computed efficiently in :math:`O((K_a + K_b)D + K_a K_b)` without materializing
    intermediate 3D tensors:
    .. math::

        s_A = A \\cdot \\Delta q, \\quad s_B = B \\cdot \\Delta q
        \\Psi = \\frac{1}{4} (s_B[None, :] - s_A[:, None])

    Transition scores are strictly signed (never clamped to non-negative) so that
    reversed semantic transitions receive negative penalties.
    """

    temporal_horizon_ms: float | None = None

    def compute_transition_matrix(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        query_delta: np.ndarray,
        source_timestamps_ms: np.ndarray | None = None,
        target_timestamps_ms: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute (Ka, Kb) transition compatibility matrix.

        Args:
            source_embeddings: Array of shape (Ka, D) of visual embeddings for event i.
            target_embeddings: Array of shape (Kb, D) of visual embeddings for event i+1.
            query_delta: Vector of shape (D,) representing q_{i+1} - q_i.
            source_timestamps_ms: Optional timestamps of source candidates.
            target_timestamps_ms: Optional timestamps of target candidates.

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
        psi = 0.25 * (s_B[None, :] - s_A[:, None])

        # Optional temporal horizon attenuation
        if (
            self.temporal_horizon_ms is not None
            and self.temporal_horizon_ms > 0
            and source_timestamps_ms is not None
            and target_timestamps_ms is not None
        ):
            src_t = np.asarray(source_timestamps_ms, dtype=np.float64)[:, None]
            tgt_t = np.asarray(target_timestamps_ms, dtype=np.float64)[None, :]
            delta_t = tgt_t - src_t
            in_horizon = (delta_t > 0) & (delta_t <= self.temporal_horizon_ms)
            psi = np.where(in_horizon, psi, 0.0)

        return psi

    def score_all_transitions(
        self,
        candidate_embeddings: Sequence[np.ndarray],
        query_embeddings: np.ndarray,
        candidate_timestamps_ms: Sequence[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Compute transition matrices for all consecutive event pairs 0..M-2.

        Args:
            candidate_embeddings: Sequence of length M, each of shape (K_i, D).
            query_embeddings: Array of shape (M, D) with SigLIP text embeddings.
            candidate_timestamps_ms: Optional sequence of length M with candidate timestamps.

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
            src_t = candidate_timestamps_ms[i] if candidate_timestamps_ms is not None else None
            tgt_t = candidate_timestamps_ms[i + 1] if candidate_timestamps_ms is not None else None
            mat = self.compute_transition_matrix(
                candidate_embeddings[i],
                candidate_embeddings[i + 1],
                query_delta=dQ,
                source_timestamps_ms=src_t,
                target_timestamps_ms=tgt_t,
            )
            transitions.append(mat)
        return transitions


def _validate_and_extract_edge_matrix(
    spec: np.ndarray | TransitionEdgeMatrix | None,
    global_weight: float,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, float]:
    """Validate and extract 2D transition matrix with strict finite and shape checks."""
    if spec is None:
        return None, 1.0

    if isinstance(spec, TransitionEdgeMatrix):
        matrix = spec.matrix
        weight = spec.weight * global_weight
    else:
        matrix = spec
        weight = global_weight

    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"Transition matrix must be 2D, got shape {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("Transition matrix contains non-finite values (NaN or Inf)")
    if expected_shape is not None and mat.shape != expected_shape:
        raise ValueError(
            f"Expected transition matrix of shape {expected_shape}, got {mat.shape}"
        )
    return mat, weight


def decode_transition_graph(
    video: VideoEventScores,
    transitions: Sequence[np.ndarray | TransitionEdgeMatrix] | None = None,
    transition_weight: float = 1.0,
    lambda_gap: float = 1e-5,
    paths: int = 1,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
    min_separation_ms: int = 0,
    *,
    allowed: np.ndarray | None = None,
) -> list[DPPath]:
    """Decode optimal chronological paths through video frames using motion-aware transition edges.

    Recurrence:
    .. math::

        DP_i(t) = U_i(t) + \\max_{s < t} [ DP_{i-1}(s) + \\psi_{i-1, i}(s, t) - \\lambda (t - s) ]

    Args:
        video: VideoEventScores holding per-event frame scores and timestamps.
        transitions: Optional sequence of transition compatibility matrices (length n_events - 1).
        transition_weight: Global scaling weight for transition edge scores.
        lambda_gap: Linear time-gap penalty weight.
        paths: Maximum number of ranked non-overlapping paths to return.
        event_power: Power scaling applied to positive unary scores.
        cluster_delta: Score drift threshold for frame clustering.
        min_separation_ms: Minimum timestamp separation between alternative paths.
        allowed: Optional boolean admissibility mask of shape (n_events, n_frames).

    Returns:
        Ranked list of DPPath instances.
    """
    # Fast-path: when no transition matrices or zero weight, delegate to baseline
    if (transitions is None or transition_weight == 0.0) and cluster_delta == 0.0:
        return align_video(
            video,
            lambda_gap=lambda_gap,
            paths=paths,
            event_power=event_power,
            cluster_delta=cluster_delta,
            min_separation_ms=min_separation_ms,
            allowed=allowed,
        )

    prep = _prepare_dp_inputs(video, allowed, event_power, cluster_delta)
    if prep is None:
        return []
    scores, frames, starts, source, reachable = prep
    n_events, n_frames = scores.shape

    weighted_time = lambda_gap * np.asarray(video.timestamps_ms, dtype=np.float64)
    current = scores[0].copy()
    back = np.zeros((n_events, n_frames), dtype=np.int64)

    # Pre-build predecessor validity mask: valid[s, t] is True iff s <= source[t] and s >= 0
    s_indices = np.arange(n_frames)[:, None]  # shape (n_frames, 1)
    t_sources = source[None, :]               # shape (1, n_frames)
    t_reachable = reachable[None, :]          # shape (1, n_frames)
    valid_predecessors = (s_indices <= t_sources) & t_reachable

    for event in range(1, n_events):
        edge_idx = event - 1
        edge_mat: np.ndarray | None = None
        if transitions is not None and edge_idx < len(transitions):
            raw_mat, weight = _validate_and_extract_edge_matrix(
                transitions[edge_idx],
                transition_weight,
                expected_shape=(n_frames, n_frames),
            )
            if raw_mat is not None:
                edge_mat = raw_mat * weight

        shifted_prev = current + weighted_time

        if edge_mat is None or np.all(edge_mat == 0.0):
            # No edge weight: use O(n) prefix maximum
            running = np.maximum.accumulate(shifted_prev)
            argmax = np.maximum.accumulate(np.where(shifted_prev == running, frames, 0))
            current = np.where(
                reachable, scores[event] - weighted_time + running[source], -np.inf
            )
            back[event] = np.where(reachable, argmax[source], 0)
        else:
            # Motion-aware edge weight: candidate compatibility matrix
            cand_scores = shifted_prev[:, None] + edge_mat
            masked_cand = np.where(
                valid_predecessors & np.isfinite(shifted_prev)[:, None],
                cand_scores,
                -np.inf,
            )
            best_s = np.argmax(masked_cand, axis=0)
            best_val = np.max(masked_cand, axis=0)

            current = np.where(
                reachable & np.isfinite(best_val),
                scores[event] - weighted_time + best_val,
                -np.inf,
            )
            back[event] = np.where(reachable, best_s, 0)

    timestamps = np.asarray(video.timestamps_ms, dtype=np.int64)
    results: list[DPPath] = []
    accepted: list[int] = []

    for endpoint in np.argsort(-current):
        if len(results) >= paths:
            break
        if not np.isfinite(current[endpoint]):
            break
        position = int(endpoint)
        if any(
            abs(int(timestamps[position]) - taken) < min_separation_ms
            for taken in accepted
        ):
            continue
        accepted.append(int(timestamps[position]))
        path = [position]

        for event in range(n_events - 1, 0, -1):
            position = int(back[event, position])
            path.append(position)
        ordered = tuple(reversed(path))
        results.append(
            DPPath(
                video_id=video.video_id,
                score=float(current[endpoint]),
                frame_idx=tuple(int(video.frame_idx[pos]) for pos in ordered),
                frame_ids=tuple(str(video.frame_ids[pos]) for pos in ordered),
            )
        )
    return results


def decode_candidate_lattice(
    video: VideoEventScores,
    candidate_k: int = 32,
    transitions: Sequence[np.ndarray | TransitionEdgeMatrix] | None = None,
    transition_weight: float = 1.0,
    lambda_gap: float = 1e-5,
    paths: int = 1,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
    min_separation_ms: int = 0,
    *,
    candidates: Sequence[EventCandidateLayer] | None = None,
    transition_scores: Sequence[np.ndarray | TransitionEdgeMatrix] | None = None,
    allowed: np.ndarray | None = None,
) -> list[DPPath]:
    """Decode optimal chronological paths through a candidate lattice per event.

    Reduces computational complexity from :math:`O(M \\cdot F^2)` to :math:`O(M \\cdot K^2)`,
    enabling millisecond-level motion-aware graph decoding over long videos with thousands
    of keyframes.

    When candidates are provided externally (e.g. from ``select_event_candidates``),
    they define the layered graph nodes :math:`C_1 \\to C_2 \\to \\dots \\to C_M`.
    Otherwise, candidates are automatically selected via ``select_event_candidates``.

    Transition scores can be provided via ``transition_scores`` or the legacy alias ``transitions``,
    as either full-frame :math:`(F, F)` matrices or candidate-sliced :math:`(K_{i-1}, K_i)` matrices.
    """
    prep = _prepare_dp_inputs(video, allowed, event_power, cluster_delta)
    if prep is None:
        return []
    scores, frames, starts, source, reachable = prep
    n_events, n_frames = scores.shape

    # Resolve candidates: external sequence or top-k selection
    if candidates is None:
        cand_layers = select_event_candidates(
            video,
            candidate_k=candidate_k,
            allowed=allowed,
            event_power=event_power,
            cluster_delta=cluster_delta,
        )
        if len(cand_layers) == 0:
            return []
        candidates_seq = cand_layers
    else:
        if len(candidates) != n_events:
            raise ValueError(
                f"Number of candidate layers ({len(candidates)}) must match number of events ({n_events})"
            )
        candidates_seq = candidates

    # Ensure every layer has at least one candidate frame
    for layer in candidates_seq:
        if len(layer.frame_indices) == 0:
            return []

    effective_transitions = transition_scores if transition_scores is not None else transitions

    # Initialize DP on first event candidates
    c0 = np.asarray(candidates_seq[0].frame_indices, dtype=np.int64)
    dp_vals = scores[0, c0].copy()
    back_pointers: list[np.ndarray] = []

    for event in range(1, n_events):
        edge_idx = event - 1
        c_prev = np.asarray(candidates_seq[event - 1].frame_indices, dtype=np.int64)
        c_curr = np.asarray(candidates_seq[event].frame_indices, dtype=np.int64)

        # Extract and slice transition matrix to candidate lattice
        edge_sub_mat: np.ndarray | None = None
        if effective_transitions is not None and edge_idx < len(effective_transitions):
            raw_mat, weight = _validate_and_extract_edge_matrix(
                effective_transitions[edge_idx],
                transition_weight,
                expected_shape=None,
            )
            if raw_mat is not None:
                if raw_mat.shape == (n_frames, n_frames):
                    edge_sub_mat = raw_mat[np.ix_(c_prev, c_curr)] * weight
                elif raw_mat.shape == (len(c_prev), len(c_curr)):
                    edge_sub_mat = raw_mat * weight
                else:
                    raise ValueError(
                        f"Transition matrix shape {raw_mat.shape} must match either "
                        f"full video ({n_frames}, {n_frames}) or candidate lattice ({len(c_prev)}, {len(c_curr)})"
                    )

        # Chronological predecessor condition: frame s strictly precedes frame t
        # (s <= source[t] requiring reachable[t] == True to prevent frame 0 self-reachability)
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
    c_last = np.asarray(candidates_seq[-1].frame_indices, dtype=np.int64)
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
            prev_global = int(candidates_seq[event - 1].frame_indices[prev_local])
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
    frame_embeddings: FrameEmbeddingAccessor | Any,
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
        frame_embeddings: Source of visual embeddings (FrameEmbeddingAccessor, DenseIndex,
            dict, or video embedding array of shape (N_frames, D)).
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
    cand_embs: list[np.ndarray] = []
    if isinstance(frame_embeddings, np.ndarray) and frame_embeddings.ndim == 2:
        for layer in candidates:
            cand_embs.append(frame_embeddings[layer.frame_indices])
    else:
        accessor = (
            frame_embeddings
            if isinstance(frame_embeddings, FrameEmbeddingAccessor)
            else FrameEmbeddingAccessor(frame_embeddings)
        )
        for layer in candidates:
            cand_embs.append(accessor(video.video_id, layer.frame_indices))

    cand_timestamps = [video.timestamps_ms[layer.frame_indices] for layer in candidates]

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
            candidate_timestamps_ms=cand_timestamps,
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
    frame_embeddings: FrameEmbeddingAccessor | Any,
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
    "AlignedPath",
    "ConditionedDPPath",
    "DPPath",
    "EventCandidateLayer",
    "FrameEmbeddingAccessor",
    "EmbeddingDeltaTransitionScorer",
    "MotionCosineTransitionScorer",
    "TransitionEdgeMatrix",
    "decode_candidate_lattice",
    "decode_motion_graph_video",
    "decode_transition_graph",
    "encode_query_events",
    "rank_motion_graph_paths",
    "select_event_candidates",
]
