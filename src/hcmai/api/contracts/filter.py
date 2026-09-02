"""Strict HTTP contracts for deterministic metadata filtering.

The models validate the public request and complete page response. They do not
own SQLite queries, artifact paths, or semantic retrieval behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hcmai.filtering.normalization import normalize_filter_text


class FilterMetadataFilters(BaseModel):
    """Populated metadata predicates combined with exact AND semantics."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=500)
    asr: str | None = Field(default=None, max_length=500)
    caption: str | None = Field(default=None, max_length=500)
    ocr: str | None = Field(default=None, max_length=500)
    objects: dict[str, int] = Field(default_factory=dict)

    @field_validator("title", "asr", "caption", "ocr", mode="before")
    @classmethod
    def normalize_text_predicate(cls, value: Any) -> str | None:
        """Normalize populated text and omit blank predicates."""

        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("filter text must be a string or null")
        normalized = normalize_filter_text(value)
        return normalized or None

    @field_validator("objects", mode="before")
    @classmethod
    def normalize_object_predicates(cls, value: Any) -> dict[str, int]:
        """Normalize labels without merging ambiguous exact-count entries."""

        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("objects must be a label-to-count mapping")

        normalized_objects: dict[str, int] = {}
        for raw_label, count in value.items():
            if not isinstance(raw_label, str):
                raise ValueError("object labels must be strings")
            label = normalize_filter_text(raw_label)
            if not label:
                raise ValueError("object labels must not be blank")
            if label in normalized_objects:
                raise ValueError("object labels must be unique after normalization")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("object counts must be nonnegative integers")
            normalized_objects[label] = count

        return normalized_objects


class FilterRequest(BaseModel):
    """One exact metadata-filter request using one-based pagination."""

    model_config = ConfigDict(extra="forbid")

    metadata_filters: FilterMetadataFilters = Field(
        default_factory=FilterMetadataFilters
    )
    folder_id: str | None = None
    video_id: str | None = None
    frames_per_pages: int = Field(default=12, ge=1, le=48)
    page_id: int = Field(default=1, ge=1)

    @field_validator("folder_id", "video_id", mode="before")
    @classmethod
    def normalize_scope(cls, value: Any) -> str | None:
        """Trim optional canonical scope without rewriting its identifier."""

        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("filter scope must be a string or null")
        stripped = value.strip()
        return stripped or None


class FilterResult(BaseModel):
    """One canonical frame plus all metadata needed by a Filter result card."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str
    video_id: str
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    folder_id: str
    title: str | None = None
    caption: str | None = None
    ocr: str | None = None
    objects: dict[str, int] = Field(default_factory=dict)
    asr: str | None = None


class FilterResponse(BaseModel):
    """A complete, stably ordered Filter result page and its true totals."""

    model_config = ConfigDict(extra="forbid")

    page_id: int = Field(ge=1)
    frames_per_pages: int = Field(ge=1, le=48)
    total_results: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    results: list[FilterResult] = Field(default_factory=list)

