from __future__ import annotations
from pydantic import Field
from .common import HTTPContract

class ReadinessModel(HTTPContract):
    enabled: bool = True
    loaded: bool
    checkpoint: str | None = None
    revision: str | None = None

class ReadinessCapabilities(HTTPContract):
    embedding: bool = False
    reranking: bool = False
    structured_parsing: bool = False
    shot_detection: bool = False
    event_detection: bool = False
    dino_embedding: bool = False
    image_embedding: bool = False
    caption: bool = False
    ocr: bool = False
    objects: bool = False
    asr: bool = False
    diarization: bool = False

class InferenceReadiness(HTTPContract):
    ready: bool
    models: dict[str, ReadinessModel]
    capabilities: ReadinessCapabilities = Field(default_factory=ReadinessCapabilities)
