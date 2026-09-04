"""HTTP contracts for direct literal search over loaded frame evidence."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


_NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
_OptionalScope = Annotated[
    str | None,
    StringConstraints(strip_whitespace=True, min_length=1),
]
_OptionalFilterText = Annotated[
    str | None,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]
_ObjectMinimumCount = Annotated[int, Field(ge=1)]


class FilterMetadataFilters(BaseModel):
    """Independent literal predicates supplied by the source-specific Filter UI.

    Populated text fields are normalized substring predicates over their own
    evidence source. Every requested object key is an exact normalized label
    with a minimum required count.
    """

    model_config = ConfigDict(extra="forbid")

    title: _OptionalFilterText = None
    asr: _OptionalFilterText = None
    caption: _OptionalFilterText = None
    ocr: _OptionalFilterText = None
    objects: dict[_NonBlankString, _ObjectMinimumCount] = Field(default_factory=dict)

    def populated_text(self) -> dict[str, str]:
        """Return only the text predicates that must participate in matching."""

        return {
            source: value
            for source, value in {
                "title": self.title,
                "asr": self.asr,
                "caption": self.caption,
                "ocr": self.ocr,
            }.items()
            if value is not None
        }


class FilterRequest(BaseModel):
    """Request one fixed-size page of source-specific literal frame matches."""

    model_config = ConfigDict(extra="forbid")

    metadata_filters: FilterMetadataFilters = Field(default_factory=FilterMetadataFilters)
    folder_id: _OptionalScope = None
    video_id: _OptionalScope = None
    frames_per_pages: int = Field(default=20, ge=20, le=20)
    page_id: int = Field(default=1, ge=1)


class FilterResult(BaseModel):
    """One canonical frame with complete display metadata and matched text."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    fps: float | None = None
    folder_id: str
    title: str | None = None
    caption: str | None = None
    ocr: str | None = None
    objects: dict[str, int] = Field(default_factory=dict)
    asr: str | None = None
    matches: dict[str, str] = Field(default_factory=dict)


class FilterResponse(BaseModel):
    """Paginated literal matches and evidence sources loaded at startup."""

    model_config = ConfigDict(extra="forbid")

    page_id: int
    frames_per_pages: int
    total_pages: int
    total_results: int
    available_sources: list[str]
    results: list[FilterResult] = Field(default_factory=list)


__all__ = [
    "FilterMetadataFilters",
    "FilterRequest",
    "FilterResponse",
    "FilterResult",
]
