"""Import BTC-provided detections as structured object evidence.

This package normalizes organizer JSON artifacts. It never runs object
detection or infers spatial relationships.
"""

from .config import ObjectConfig
from .importer import import_objects

__all__ = ["ObjectConfig", "import_objects"]
