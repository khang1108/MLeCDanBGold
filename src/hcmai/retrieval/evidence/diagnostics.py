"""Diagnostic dataclasses and telemetry for multimodal temporal evidence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from hcmai.retrieval.evidence.calibration import CalibratedComponent
    from hcmai.retrieval.evidence.components import TemporalScoreBundle


@dataclass(frozen=True, slots=True)
class ComponentEventDebug:
    """Component-level diagnostic summary for one temporal event.

    Attributes:
        component: Name of the evidence component (e.g. ``visual_dense``).
        event_index: 0-based index of the event query.
        raw_max: Maximum raw score across all canonical frames.
        raw_median: Median raw score across all canonical frames.
        calibrated_max: Maximum score in [0, 1] after calibration.
        reliability: Reliability weight in [0, 1] derived from dynamic range.
        coverage_ratio: Fraction of frames where this component has valid coverage.
        top_positions: Indices of canonical frames with highest calibrated scores.
    """

    component: str
    event_index: int
    raw_max: float
    raw_median: float
    calibrated_max: float
    reliability: float
    coverage_ratio: float
    top_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TemporalEvidenceDebugResult:
    """Telemetry bundle pairing full fused scores with component diagnostics."""

    fused_scores: np.ndarray
    rows: tuple[ComponentEventDebug, ...]


def build_evidence_diagnostics(
    bundle: TemporalScoreBundle,
    calibrated: dict[str, CalibratedComponent],
    fused_scores: np.ndarray,
    top_positions: int = 10,
) -> TemporalEvidenceDebugResult:
    """Construct diagnostic telemetry from components and calibrated matrices.

    Args:
        bundle: Bundle of raw score components and optional coverage masks.
        calibrated: Dictionary of calibrated score matrices and reliabilities.
        fused_scores: Final fused score matrix shaped ``[event_count, frame_count]``.
        top_positions: Number of highest-scoring canonical frame positions to record.

    Returns:
        A :class:`TemporalEvidenceDebugResult` containing fused scores and row telemetry.
    """

    rows: list[ComponentEventDebug] = []
    event_count, frame_count = bundle.shape

    for event_index in range(event_count):
        for name, component in bundle.components.items():
            raw = component.raw_scores[event_index]
            raw_max = float(raw.max()) if len(raw) else 0.0
            raw_median = float(np.median(raw)) if len(raw) else 0.0

            cal = calibrated[name]
            cal_scores = cal.scores[event_index]
            calibrated_max = float(cal_scores.max()) if len(cal_scores) else 0.0
            reliability = float(cal.reliability[event_index])

            if component.coverage is not None:
                coverage_ratio = float(np.mean(component.coverage.astype(np.float32)))
            else:
                coverage_ratio = 1.0

            top_k = min(top_positions, len(cal_scores))
            if top_k > 0:
                top_indices = np.argsort(-cal_scores, kind="stable")[:top_k]
                top_positions_tuple = tuple(int(idx) for idx in top_indices)
            else:
                top_positions_tuple = ()

            rows.append(
                ComponentEventDebug(
                    component=name,
                    event_index=event_index,
                    raw_max=raw_max,
                    raw_median=raw_median,
                    calibrated_max=calibrated_max,
                    reliability=reliability,
                    coverage_ratio=coverage_ratio,
                    top_positions=top_positions_tuple,
                )
            )

    return TemporalEvidenceDebugResult(
        fused_scores=fused_scores,
        rows=tuple(rows),
    )


__all__ = [
    "ComponentEventDebug",
    "TemporalEvidenceDebugResult",
    "build_evidence_diagnostics",
]
