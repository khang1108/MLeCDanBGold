"""Tests for query-conditioned EmbeddingDeltaTransitionScorer (SP-06, SP-07).

Verifies Phase 4 (Tasks 4.1, 4.2, 4.3):
- Forward transition: psi(A, B) > 0
- Reverse transition: psi_reverse < psi_forward and psi_reverse < 0
- Static pair: psi = 0
- Fast matrix formulation matches exact pairwise loop (1/4 * (v_b - v_a)^T dQ)
- Shuffled transitions degrade score
- Temporal horizon attenuation
"""

import numpy as np
import pytest

from hcmai.temporal.transition_decoder import EmbeddingDeltaTransitionScorer


def test_forward_transition_is_positive():
    """Verify forward semantic transition produces strictly positive score (Task 4.3)."""
    scorer = EmbeddingDeltaTransitionScorer()

    # Query: sit (dim 0) -> stand (dim 1)
    q1 = np.array([1.0, 0.0])
    q2 = np.array([0.0, 1.0])
    dQ = q2 - q1  # [-1, 1]

    # Video: sit (dim 0) -> stand (dim 1)
    vA = np.array([[1.0, 0.0]])
    vB = np.array([[0.0, 1.0]])

    psi = scorer.compute_transition_matrix(vA, vB, query_delta=dQ)

    assert psi.shape == (1, 1)
    # Expected: 1/4 * ((0 - 1)*(-1) + (1 - 0)*(1)) = 1/4 * (1 + 1) = 0.5
    assert np.isclose(psi[0, 0], 0.5)
    assert psi[0, 0] > 0.0


def test_reverse_transition_is_negative_and_penalized():
    """Verify reverse semantic transition produces negative score and < forward (Task 4.3)."""
    scorer = EmbeddingDeltaTransitionScorer()

    # Query: sit -> stand
    dQ = np.array([-1.0, 1.0])

    # Video: stand (dim 1) -> sit (dim 0) (opposite direction)
    vA = np.array([[0.0, 1.0]])
    vB = np.array([[1.0, 0.0]])

    psi_rev = scorer.compute_transition_matrix(vA, vB, query_delta=dQ)

    assert psi_rev.shape == (1, 1)
    # Expected: 1/4 * ((1 - 0)*(-1) + (0 - 1)*(1)) = 1/4 * (-1 - 1) = -0.5
    assert np.isclose(psi_rev[0, 0], -0.5)
    assert psi_rev[0, 0] < 0.0


def test_static_pair_zero_transition_signal():
    """Verify static video state produces zero transition signal (Task 4.3)."""
    scorer = EmbeddingDeltaTransitionScorer()

    dQ = np.array([-1.0, 1.0])
    vA = np.array([[1.0, 0.0]])
    vB = np.array([[1.0, 0.0]])

    psi = scorer.compute_transition_matrix(vA, vB, query_delta=dQ)
    assert np.isclose(psi[0, 0], 0.0)


def test_matrix_computation_matches_exact_loop():
    """Verify O((Ka+Kb)D + Ka*Kb) matrix computation equals explicit pairwise loop."""
    scorer = EmbeddingDeltaTransitionScorer()
    rng = np.random.default_rng(42)

    Ka, Kb, D = 6, 8, 16
    A = rng.normal(size=(Ka, D))
    B = rng.normal(size=(Kb, D))
    dQ = rng.normal(size=D)

    # Fast matrix computation
    psi_matrix = scorer.compute_transition_matrix(A, B, query_delta=dQ)
    assert psi_matrix.shape == (Ka, Kb)

    # Normalize vectors for ground-truth loop
    A_norm = A / np.linalg.norm(A, axis=-1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=-1, keepdims=True)

    # Explicit loop computation
    psi_expected = np.zeros((Ka, Kb))
    for a in range(Ka):
        for b in range(Kb):
            psi_expected[a, b] = 0.25 * np.dot(B_norm[b] - A_norm[a], dQ)

    np.testing.assert_allclose(psi_matrix, psi_expected, atol=1e-10)


def test_shuffled_transition_degradation():
    """Verify that coherent forward transitions score significantly higher than shuffled target embeddings (Gate 5)."""
    scorer = EmbeddingDeltaTransitionScorer()
    rng = np.random.default_rng(101)

    D = 32
    dQ = rng.normal(size=D)
    dQ /= np.linalg.norm(dQ)

    base = rng.normal(size=(10, D))
    base /= np.linalg.norm(base, axis=-1, keepdims=True)

    # True target frames that undergo the semantic transition dQ
    true_targets = base + 0.8 * dQ
    true_targets /= np.linalg.norm(true_targets, axis=-1, keepdims=True)

    # Shuffled/distractor target frames (unaligned with dQ)
    shuffled_targets = rng.normal(size=(10, D))
    shuffled_targets /= np.linalg.norm(shuffled_targets, axis=-1, keepdims=True)

    psi_true = scorer.compute_transition_matrix(base, true_targets, query_delta=dQ)
    psi_shuffled = scorer.compute_transition_matrix(base, shuffled_targets, query_delta=dQ)

    # Coherent true transitions must score higher on average than shuffled/unaligned transitions
    assert np.mean(np.diag(psi_true)) > np.mean(np.diag(psi_shuffled))


def test_score_all_transitions():
    """Verify score_all_transitions computes matrices for all adjacent event pairs."""
    scorer = EmbeddingDeltaTransitionScorer()
    rng = np.random.default_rng(202)

    D = 8
    Q = rng.normal(size=(3, D)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=-1, keepdims=True)

    cand_embs = [
        rng.normal(size=(4, D)).astype(np.float32),
        rng.normal(size=(5, D)).astype(np.float32),
        rng.normal(size=(3, D)).astype(np.float32),
    ]

    matrices = scorer.score_all_transitions(cand_embs, Q)

    assert len(matrices) == 2
    assert matrices[0].shape == (4, 5)
    assert matrices[1].shape == (5, 3)


def test_temporal_horizon_attenuation():
    """Verify transitions outside temporal_horizon_ms are zeroed."""
    scorer = EmbeddingDeltaTransitionScorer(temporal_horizon_ms=5000.0)

    dQ = np.array([-1.0, 1.0])
    vA = np.array([[1.0, 0.0], [1.0, 0.0]])
    vB = np.array([[0.0, 1.0], [0.0, 1.0]])

    # source timestamps: 1000, 2000
    src_t = np.array([1000, 2000])
    # target timestamps: 3000 (within 5s), 10000 (beyond 5s)
    tgt_t = np.array([3000, 10000])

    psi = scorer.compute_transition_matrix(
        vA, vB, query_delta=dQ,
        source_timestamps_ms=src_t,
        target_timestamps_ms=tgt_t,
    )

    # Pair (src=1000, tgt=3000): delta=2000 <= 5000 -> active (> 0)
    assert psi[0, 0] > 0.0
    # Pair (src=1000, tgt=10000): delta=9000 > 5000 -> zeroed
    assert psi[0, 1] == 0.0
    # Pair (src=2000, tgt=10000): delta=8000 > 5000 -> zeroed
    assert psi[1, 1] == 0.0
