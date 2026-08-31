"""Public read-only corpus facade and runtime canonical data models.

Specialist artifact stores and asset resolution remain implementation details.
Offline artifact-validation models live with their enrichment owners.
"""

from .corpus import Corpus, CorpusFrameLoadError
from .models import Frame, TranscriptSegment, VideoMetadata

__all__ = [
    "Corpus",
    "Frame",
    "TranscriptSegment",
    "VideoMetadata",
]
