"""Evidence-provider implementations for the shared temporal facade."""

from .dense import DenseOrderedEvidenceProvider
from .sparse import ProgressiveEvidenceProvider

__all__ = ["DenseOrderedEvidenceProvider", "ProgressiveEvidenceProvider"]
