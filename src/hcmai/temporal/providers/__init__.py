"""Evidence-provider implementations for the shared temporal facade."""

from .dense import DenseOrderedEvidenceProvider
from .sparse import SparseProgressiveEvidenceProvider

__all__ = ["DenseOrderedEvidenceProvider", "SparseProgressiveEvidenceProvider"]
