"""Canonical frame data capability.

Cross-component code imports :class:`DataService` from ``data.pipeline``.
"""

# Compatibility alias for existing callers. New cross-component code should
# use DataService; the store implementation remains owned by this package.
from hcmai.data.stores.frame import FrameStore

__all__ = ["FrameStore"]
