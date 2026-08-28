"""Validate reusable ASR transcripts and vectors for the local pipeline.

The custom pipeline never runs transcription, diarization, or text embedding.
It only validates that persisted per-video transcript manifests/parquet and a
persisted ASR ``SegmentDenseIndex`` already cover the requested video IDs, then
exposes a lineage-fingerprinted bundle so later batch stages can subset by
``video_id`` (Task 6) without touching any model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hcmai.common.utils.io import read_json
from hcmai.common.utils.logging import get_logger
from hcmai.data.enrichment.transcripts.manifest import TranscriptManifest
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

logger = get_logger(__name__)

_REQUIRED_SEGMENT_COLUMNS = {"segment_id", "video_id", "start_ms", "end_ms"}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ASRReuseBundle(BaseModel):
    """Validated, lineage-fingerprinted reusable ASR evidence for a video set."""

    model_config = ConfigDict(frozen=True)

    transcripts_root: str
    index_root: str
    video_ids: tuple[str, ...]
    transcript_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    segment_count: int = Field(ge=0)

    @field_validator("video_ids")
    @classmethod
    def _non_empty_sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("ASRReuseBundle requires at least one video_id")
        if len(set(value)) != len(value):
            raise ValueError("ASRReuseBundle video_ids must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _non_blank_roots(self) -> "ASRReuseBundle":
        if not self.transcripts_root.strip() or not self.index_root.strip():
            raise ValueError("transcripts_root and index_root must not be blank")
        return self


def _video_directory_prefix(video_id: str) -> str:
    """Return the organizer ``Lxx`` directory owning a ``Lxx_Vnnn`` video ID."""

    prefix = video_id.split("_", 1)[0]
    if not prefix:
        raise ValueError(f"cannot derive a transcript directory for video_id: {video_id}")
    return prefix


def _load_transcript_manifest(transcripts_root: Path, video_id: str) -> TranscriptManifest:
    """Load and validate the completed transcript manifest for one video."""

    path = transcripts_root / _video_directory_prefix(video_id) / f"{video_id}.manifest.json"
    if not path.is_file():
        raise ValueError(f"missing transcript manifest for reusable ASR: {video_id} ({path})")
    manifest = TranscriptManifest(**read_json(path))
    if manifest.video_id != video_id:
        raise ValueError(f"transcript manifest video_id mismatch at {path}: {manifest.video_id}")
    if manifest.status != "completed":
        raise ValueError(f"transcript manifest for {video_id} is not completed: {path}")
    return manifest


def _load_segment_table(transcripts_root: Path, video_id: str, manifest: TranscriptManifest) -> pd.DataFrame:
    """Load and validate the segment-native transcript parquet for one video."""

    path = transcripts_root / _video_directory_prefix(video_id) / f"{video_id}.parquet"
    if not path.is_file():
        raise ValueError(f"missing transcript segment parquet for {video_id}: {path}")
    table = pd.read_parquet(path)

    missing_columns = _REQUIRED_SEGMENT_COLUMNS - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"transcript parquet for {video_id} is missing columns: {sorted(missing_columns)}"
        )
    if len(table) != manifest.segment_count:
        raise ValueError(
            f"transcript parquet row count for {video_id} ({len(table)}) disagrees "
            f"with manifest segment_count ({manifest.segment_count})"
        )
    if table["segment_id"].duplicated().any():
        raise ValueError(f"duplicate segment_id in transcript parquet for {video_id}")
    if (table["video_id"] != video_id).any():
        raise ValueError(f"foreign video_id present in transcript parquet for {video_id}")
    if (table["start_ms"] < 0).any() or (table["end_ms"] <= table["start_ms"]).any():
        raise ValueError(f"invalid segment interval in transcript parquet for {video_id}")
    return table


def _validate_index_coverage(
    index: SegmentDenseIndex, video_id: str, segment_table: pd.DataFrame
) -> int:
    """Confirm the persisted ASR index has finite, matching vectors for one video."""

    positions = index.video_positions(video_id)
    if positions.size == 0:
        raise ValueError(f"ASR index has no vectors for reusable video: {video_id}")
    if positions.size != len(segment_table):
        raise ValueError(
            f"ASR index vector count for {video_id} ({positions.size}) disagrees "
            f"with transcript segment count ({len(segment_table)})"
        )

    vectors = np.asarray(index.vectors[positions])
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"ASR index vectors for {video_id} contain non-finite values")

    indexed_segment_ids = set(index.mapping.loc[positions, "segment_id"])
    transcript_segment_ids = set(segment_table["segment_id"])
    if indexed_segment_ids != transcript_segment_ids:
        raise ValueError(
            f"ASR index segment_id set for {video_id} disagrees with the transcript parquet"
        )
    return int(positions.size)


def _compute_transcript_fingerprint(manifests: Sequence[TranscriptManifest]) -> str:
    """Hash the resume-identity lineage of every reused transcript manifest."""

    payload = [
        {
            "video_id": manifest.video_id,
            "config_sha256": manifest.config_sha256,
            "asr_model": manifest.asr_model,
            "asr_revision": manifest.asr_revision,
            "diarization_enabled": manifest.diarization_enabled,
            "diarization_model": manifest.diarization_model,
            "diarization_revision": manifest.diarization_revision,
            "schema_version": manifest.schema_version,
            "pipeline_version": manifest.pipeline_version,
            "segment_count": manifest.segment_count,
        }
        for manifest in sorted(manifests, key=lambda item: item.video_id)
    ]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _compute_index_fingerprint(index: SegmentDenseIndex) -> str:
    """Hash the persisted ASR index's provenance metadata."""

    return hashlib.sha256(
        _canonical_json(index.metadata.to_dict()).encode("utf-8")
    ).hexdigest()


def validate_asr_source(
    transcripts_root: str | Path,
    index_root: str | Path,
    video_ids: Sequence[str],
) -> ASRReuseBundle:
    """Validate reusable transcripts/vectors and return a fingerprinted bundle.

    Raises:
        ValueError: If any manifest, segment table, or indexed vector for
            ``video_ids`` is missing, incomplete, or internally inconsistent.
    """

    if not video_ids:
        raise ValueError("validate_asr_source requires at least one video_id")

    roots = Path(transcripts_root), Path(index_root)
    transcripts_path, index_path = roots
    logger.info(
        "validating reusable ASR for %d video(s) from %s / %s",
        len(video_ids),
        transcripts_path,
        index_path,
    )

    index = SegmentDenseIndex.load(index_path)

    manifests: list[TranscriptManifest] = []
    total_segments = 0
    for video_id in sorted(set(video_ids)):
        manifest = _load_transcript_manifest(transcripts_path, video_id)
        segment_table = _load_segment_table(transcripts_path, video_id, manifest)
        total_segments += _validate_index_coverage(index, video_id, segment_table)
        manifests.append(manifest)

    bundle = ASRReuseBundle(
        transcripts_root=str(transcripts_path),
        index_root=str(index_path),
        video_ids=tuple(sorted(set(video_ids))),
        transcript_fingerprint=_compute_transcript_fingerprint(manifests),
        index_fingerprint=_compute_index_fingerprint(index),
        segment_count=total_segments,
    )
    logger.info(
        "reusable ASR validated: %d videos, %d segments, "
        "transcript_fingerprint=%s index_fingerprint=%s",
        len(bundle.video_ids),
        bundle.segment_count,
        bundle.transcript_fingerprint[:12],
        bundle.index_fingerprint[:12],
    )
    return bundle


def require_asr_video_coverage(bundle: ASRReuseBundle, archive_video_ids: Sequence[str]) -> None:
    """Confirm a validated ASR bundle covers every video in one archive.

    Raises:
        ValueError: If any archive video is absent from the validated bundle.
    """

    missing = sorted(set(archive_video_ids) - set(bundle.video_ids))
    if missing:
        raise ValueError(
            f"reusable ASR bundle is missing archive video(s): {', '.join(missing)}"
        )


__all__ = ["ASRReuseBundle", "require_asr_video_coverage", "validate_asr_source"]
