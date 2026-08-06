"""KIS-specific query planning and deterministic result shaping."""

from hcmai.kis.ranking import KISRankingConfig, shape_kis_candidates
from hcmai.kis.calibration import (
    CalibrationCase,
    CalibrationResult,
    calibrate_fusion,
    evaluate_fusion_weights,
)
from hcmai.kis.variants import ControlledQueryExpander, QueryVariant, VariantPlan

__all__ = [
    "ControlledQueryExpander",
    "CalibrationCase",
    "CalibrationResult",
    "KISRankingConfig",
    "QueryVariant",
    "VariantPlan",
    "calibrate_fusion",
    "evaluate_fusion_weights",
    "shape_kis_candidates",
]
