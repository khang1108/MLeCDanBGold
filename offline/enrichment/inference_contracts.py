"""Compatibility imports for model-server transport contracts.

The canonical HTTP schema is owned by ``llm.contracts``. Offline enrichment
consumes it as a client and keeps only durable artifact schemas locally.
"""
from llm.contracts import (
    AudioReferenceRequest,
    CaptionItem,
    CaptionResponse,
    DiarizationRequest,
    InferenceReadiness,
    OCRItem,
    OCRRegionItem,
    OCRResponse,
    ObjectItem,
    ObjectResponse,
    TranscriptInferenceResponse,
)

__all__ = [
    "AudioReferenceRequest",
    "CaptionItem",
    "CaptionResponse",
    "DiarizationRequest",
    "InferenceReadiness",
    "OCRItem",
    "OCRRegionItem",
    "OCRResponse",
    "ObjectItem",
    "ObjectResponse",
    "TranscriptInferenceResponse",
]
