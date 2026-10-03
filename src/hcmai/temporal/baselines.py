"""Evaluation baselines for temporal alignment and transition scoring.

Contains non-query-conditioned transition scoring models (such as B4: visual continuity)
used as ablation and benchmark baselines for SOICT experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


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


__all__ = [
    "MotionCosineTransitionScorer",
]
