"""Search services and workflows for orchestration."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AlignedPath",
    "AvsSearchService",
    "DecoderConfigSnapshot",
    "ImageQueryTooLargeError",
    "ImageSearchService",
    "InvalidImageQueryError",
    "SelectedVideoScoreResult",
    "TemporalSearchArtifact",
    "TemporalSearchGateway",
    "TemporalSearchResult",
    "TemporalSearchService",
    "decode_video_scores",
]


def __getattr__(name: str) -> Any:
    if name == "AvsSearchService":
        from hcmai.orchestration.workflows.search.avs import AvsSearchService

        return AvsSearchService
    if name in {
        "ImageSearchService",
        "ImageQueryTooLargeError",
        "InvalidImageQueryError",
    }:
        from hcmai.orchestration.workflows.search import image

        return getattr(image, name)
    if name in {
        "AlignedPath",
        "DecoderConfigSnapshot",
        "SelectedVideoScoreResult",
        "TemporalSearchArtifact",
        "TemporalSearchGateway",
        "TemporalSearchResult",
        "TemporalSearchService",
        "decode_video_scores",
    }:
        from hcmai.orchestration.workflows.search import temporal

        return getattr(temporal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
