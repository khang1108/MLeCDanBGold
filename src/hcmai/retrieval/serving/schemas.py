"""Pydantic schemas for the HTTP retrieval serving API.

These models define the request and response shapes exchanged between the
FastAPI web application and the standalone retrieval service.
"""

from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TextSearchRequestSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: NonBlank
    top_k: int = Field(default=100, ge=1)


class TextCandidateSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_id: NonBlank
    rank: int = Field(ge=1)
    score: float | None = None


class TextSearchCandidatesSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: list[TextCandidateSchema]
    retrieval_ms: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    """Runtime diagnostic capabilities and limits advertised by retrieval service."""

    model_config = ConfigDict(frozen=True)

    ready: bool
    scoring_revision: str | None = None
    active_modalities: list[str] = Field(default_factory=list)
    startup_messages: list[str] = Field(default_factory=list)
    max_temporal_event_count: int = DEFAULT_MAX_TEMPORAL_EVENT_COUNT
    image_max_upload_bytes: int = 10 * 1024 * 1024
    image_max_pixels: int = 25_000_000


class RetrievalEventSchema(BaseModel):
    """Normalized event query representation."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    canonical_text: str | None = None
    dense_text: str | None = None
    bm25_text: str | None = None
    image_asset_ids: list[str] = Field(default_factory=list)


class SearchPlanRequestSchema(BaseModel):
    """Request payload for multi-event multimodal plan search."""

    model_config = ConfigDict(frozen=True)

    events: list[RetrievalEventSchema]
    use_dense: bool = True
    use_bm25: bool = False
    top_k: int = 20


class SearchEventsRequestSchema(BaseModel):
    """Request payload for ordered event queries without explicit plan object."""

    model_config = ConfigDict(frozen=True)

    original_events: list[str]
    retrieval_events: list[str] | None = None
    caption_events: list[str] | None = None
    has_caption_events: bool = False
    use_dense: bool = True
    use_bm25: bool = False
    top_k: int = 20


class AlignedPathSchema(BaseModel):
    """Decoded temporal alignment path."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    score: float
    frame_ids: list[str]
    frame_idxs: list[int]
    timestamps_ms: list[int]


class DecoderConfigSchema(BaseModel):
    """Configuration snapshot for the DP temporal sequence decoder."""

    model_config = ConfigDict(frozen=True)

    lambda_gap: float
    event_power: float
    cluster_delta: float
    path_min_separation_ms: int


class VideoScoresSchema(BaseModel):
    """Event-by-frame score matrix and metadata for one video."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    frame_ids: list[str]
    frame_idx: list[int]
    timestamps_ms: list[int]
    scores: list[list[float]]


class TemporalSearchResultSchema(BaseModel):
    """Temporal paths and timing diagnostics."""

    model_config = ConfigDict(frozen=True)

    paths: list[AlignedPathSchema]
    retrieval_ms: float
    alignment_ms: float


class TemporalSearchArtifactSchema(BaseModel):
    """Complete temporal search output including scored video matrix."""

    model_config = ConfigDict(frozen=True)

    result: TemporalSearchResultSchema
    video_scores: list[VideoScoresSchema]
    decoder_config: DecoderConfigSchema
    scoring_revision: str | None = None


class ScoreVideoRequestSchema(BaseModel):
    """Request payload for scoring a single video under a retrieval plan."""

    model_config = ConfigDict(frozen=True)

    plan: SearchPlanRequestSchema
    video_id: str
    use_dense: bool = True
    use_bm25: bool = False


class SelectedVideoScoreSchema(BaseModel):
    """Scored frames and decoder config for a targeted single video."""

    model_config = ConfigDict(frozen=True)

    video: VideoScoresSchema
    retrieval_ms: float
    decoder_config: DecoderConfigSchema
    scoring_revision: str | None = None


class ImageCandidateSchema(BaseModel):
    """Frame candidate from direct visual retrieval."""

    model_config = ConfigDict(frozen=True)

    video_id: str
    frame_id: str
    frame_idx: int
    timestamp_ms: int
    score: float


class ImageSearchCandidatesSchema(BaseModel):
    """Direct visual search result candidates and timing metrics."""

    model_config = ConfigDict(frozen=True)

    candidates: list[ImageCandidateSchema]
    query_ms: float
    retrieval_ms: float
