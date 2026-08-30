"""Runtime corpus models and readers.

The package currently exposes only the minimal runtime dataclasses.  Offline
artifact-validation schemas remain under :mod:`hcmai.common.schemas` until
the later corpus migration phases.
"""

from .models import Frame, TranscriptSegment, VideoMetadata

__all__ = ["Frame", "TranscriptSegment", "VideoMetadata"]
