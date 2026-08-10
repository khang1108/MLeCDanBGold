"""Label-driven, deterministic KIS fusion calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, Sequence

from hcmai.common.schemas import RetrievalCandidate, RetrievalSource

KIS_CUTOFFS = (1, 5, 20, 50, 100)


@dataclass(frozen=True)
class CalibrationCase:
    candidates: tuple[RetrievalCandidate, ...]
    accepted_frame_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.accepted_frame_ids:
            raise ValueError("accepted_frame_ids must not be empty")


@dataclass(frozen=True)
class CalibrationResult:
    weights: Mapping[RetrievalSource, float]
    mean_top_k_score: float


def evaluate_fusion_weights(
    cases: Sequence[CalibrationCase],
    weights: Mapping[RetrievalSource, float],
    *,
    rrf_k: int = 60,
) -> float:
    """Compute local Mean Top-k hit score without mutating candidates."""
    if rrf_k < 1 or not weights or any(value < 0 for value in weights.values()):
        raise ValueError("rrf_k and non-negative modality weights are required")
    if not cases:
        return 0.0
    query_scores = []
    for case in cases:
        ranked = sorted(
            case.candidates,
            key=lambda candidate: (
                -sum(
                    weights.get(source, 0.0) / (rrf_k + rank)
                    for source, rank in candidate.source_ranks.items()
                ),
                candidate.frame_id,
            ),
        )
        hits = [item.frame_id in case.accepted_frame_ids for item in ranked]
        query_scores.append(
            mean(float(any(hits[:cutoff])) for cutoff in KIS_CUTOFFS)
        )
    return mean(query_scores)


def calibrate_fusion(
    cases: Sequence[CalibrationCase],
    configurations: Iterable[Mapping[RetrievalSource, float]],
    *,
    rrf_k: int = 60,
) -> CalibrationResult:
    """Select the best measured config with deterministic tie-breaking."""
    results = [
        CalibrationResult(
            dict(weights),
            evaluate_fusion_weights(cases, weights, rrf_k=rrf_k),
        )
        for weights in configurations
    ]
    if not results:
        raise ValueError("at least one weight configuration is required")
    return min(
        results,
        key=lambda result: (
            -result.mean_top_k_score,
            tuple(
                (source.value, result.weights.get(source, 0.0))
                for source in RetrievalSource
            ),
        ),
    )
