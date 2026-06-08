from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from pydantic import Field, NonPositiveFloat
from typing import Any


@dataclass(frozen=True)
class FrameRecord:
    """
    One searchable frame/keyframe in the corpus
    """

    row_id: int
    frame_id: str
    video_id: str
    group_id: str
    image_path: Path

    timestamp: float | None = None
    frame_index: int | None = None

    caption: str | None = None
    ocr_text: str | None = Field(
        default=None, description="OCR Information in Frame/Keyframe"
    )
    object_tags: tuple[str, ...] = Field(
        ..., description="Object tags appear in Frame/Keyframe"
    )
    action_tags: tuple[str, ...] = Field(
        ..., description="Action tags appear in Frame/Keyframe"
    )
    scene_tags: tuple[str, ...] = Field(
        ..., description="Scene tags appear in Frame/Keyframe"
    )

    event_id: str | None = None
    prev_frame_id: str | None = None
    next_frame_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)

        return data


@dataclass(frozen=True)
class VideoRecord:
    """ "Metadata for one original video."""

    video_id: str
    video_path: Path

    group_id: str | None = None
    duration: float | None = None
    fps: float | None = None

    width: int | None = None
    height: int | None = None
    num_frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["video_path"] = str(self.video_path)

        return data


@dataclass(frozen=True)
class QueryRecord:
    """One natural-language query

    relevant_frame_ids is optional because during competition/inference we often do not have ground truth
    """

    query_id: str
    query: str
    relevant_frame_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingRecord:
    """
    Mapping between vector index position and frame Metadata.
    """

    vector_id: int
    row_id: int
    frame_id: str
    video_id: str
    image_path: Path

    embedding_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)

        if self.embedding_path is not None:
            data["embedding_path"] = str(self.embedding_path)

        return data


@dataclass(frozen=True)
class SearchResult:
    """
    One retrieved frame returned by the search engine.
    """

    rank: int
    frame_id: str
    video_id: str
    image_path: Path
    score: float

    row_id: int | None = None
    timestamp: float | None = None
    query_id: str | None = None
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        return data


@dataclass(frozen=True)
class DatasetSummary:
    """
    Small summary object for logging and sanity checks.
    """

    num_frames: int
    num_videos: int
    num_groups: int
    num_missing_images: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
