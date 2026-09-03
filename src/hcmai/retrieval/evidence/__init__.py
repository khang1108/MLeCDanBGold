"""Full-corpus temporal evidence scoring primitives."""

from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex
from hcmai.retrieval.evidence.bm25 import BM25ArtifactError, BM25TemporalScorer
from hcmai.retrieval.evidence.calibration import CalibratedComponent, calibrate_component
from hcmai.retrieval.evidence.components import TemporalScoreBundle, TemporalScoreComponent
from hcmai.retrieval.evidence.dense import DenseTemporalScorer
from hcmai.retrieval.evidence.fusion import EventModalityRouter, TemporalFusionScorer
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
from hcmai.retrieval.evidence.normalization import minmax_rows

__all__ = [
    "BM25ArtifactError",
    "BM25TemporalScorer",
    "CalibratedComponent",
    "DenseTemporalScorer",
    "EventModalityRouter",
    "SegmentProjectedASRIndex",
    "TemporalEvidenceScorer",
    "TemporalFusionScorer",
    "TemporalScoreBundle",
    "TemporalScoreComponent",
    "calibrate_component",
    "minmax_rows",
]
