"""Frame-caption enrichment package."""

from .models import (
    CaptionAdapter,
    CaptionConfig,
    CaptionEvidence,
    CaptionJobConfig,
    DEFAULT_ENRICHMENT_CONFIG,
    ENRICHMENT_VERSION,
    usable_completed_text,
)
from .adapter import VLLMCaptionAdapter
from .generator import generate_captions

__all__ = [
    "CaptionAdapter",
    "CaptionConfig",
    "CaptionEvidence",
    "CaptionJobConfig",
    "DEFAULT_ENRICHMENT_CONFIG",
    "ENRICHMENT_VERSION",
    "VLLMCaptionAdapter",
    "generate_captions",
    "usable_completed_text",
]
