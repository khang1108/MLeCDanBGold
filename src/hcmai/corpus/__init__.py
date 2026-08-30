"""Public read-only corpus facade and runtime canonical data models.

Specialist artifact stores and asset resolution remain implementation details.
Offline artifact-validation schemas remain under :mod:`hcmai.common.schemas`
until the later corpus migration phases.
"""

from .corpus import Corpus, CorpusFrameLoadError
from .models import Frame, TranscriptSegment, VideoMetadata

__all__ = [
    "Corpus",
    "CorpusFrameLoadError",
    "Frame",
    "TranscriptSegment",
    "VideoMetadata",
]
