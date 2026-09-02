"""Build the deterministic disk-backed Filter catalog offline.

The builder joins already-published specialist artifacts to canonical frames,
validates identity, and atomically publishes SQLite. It never performs model
inference and is never called by online serving.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hcmai.filtering.catalog import CatalogAvailability, FilterCatalog
from hcmai.filtering.normalization import normalize_filter_text
from hcmai.filtering.schema import CATALOG_SCHEMA_VERSION, create_catalog_schema
from offline.artifact_readers import (
    CanonicalFrame,
    CaptionArtifactReader,
    FrameArtifactReader,
    ObjectCountsArtifactReader,
    OCRArtifactReader,
    OfflineTranscriptStore,
    VideoMetadataArtifactReader,
)


_FOLDER_PATTERN = re.compile(r"[A-Za-z]\d+")


class FilterCatalogBuildError(RuntimeError):
    """Raised when source evidence cannot produce a valid atomic catalog."""


@dataclass(frozen=True)
class FilterCatalogBuildConfig:
    """Explicit source, output, identity, and batching choices for one build."""

    frames_path: Path
    output_path: Path
    catalog_version: str
    video_metadata_path: Path | None = None
    caption_path: Path | None = None
    ocr_path: Path | None = None
    object_counts_path: Path | None = None
    transcripts_path: Path | None = None
    source_lineage: dict[str, str] = field(default_factory=dict)
    batch_size: int = 2000

    def __post_init__(self) -> None:
        """Normalize paths and reject ambiguous build configuration."""

        if not self.catalog_version.strip():
            raise ValueError("catalog_version must not be blank")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        for name in (
            "frames_path",
            "output_path",
            "video_metadata_path",
            "caption_path",
            "ocr_path",
            "object_counts_path",
            "transcripts_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())


@dataclass(frozen=True)
class FilterCatalogBuildReport:
    """Machine-readable facts recorded after successful publication."""

    output_path: Path
    catalog_version: str
    frame_count: int
    availability: CatalogAvailability
    output_size_bytes: int
    build_seconds: float

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the report without exposing source artifact paths."""

        return {
            "output_path": str(self.output_path),
            "catalog_version": self.catalog_version,
            "frame_count": self.frame_count,
            "availability": asdict(self.availability),
            "output_size_bytes": self.output_size_bytes,
            "build_seconds": self.build_seconds,
        }


@dataclass(frozen=True)
class _Sources:
    """Loaded source stores plus their global modality availability."""

    frames: FrameArtifactReader
    video_metadata: VideoMetadataArtifactReader | None
    captions: CaptionArtifactReader | None
    ocr: OCRArtifactReader | None
    objects: ObjectCountsArtifactReader | None
    transcripts: OfflineTranscriptStore | None
    availability: CatalogAvailability


def build_filter_catalog(
    config: FilterCatalogBuildConfig,
) -> FilterCatalogBuildReport:
    """Validate sources, build a temporary catalog, and atomically publish it."""

    started = time.perf_counter()
    temporary_path: Path | None = None
    try:
        sources = _load_sources(config)
        frames = tuple(sources.frames.iter_frames())
        canonical = {frame.frame_id: frame for frame in frames}
        _validate_source_identity(sources, canonical)
        lineage = _build_source_lineage(config, sources)

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = config.output_path.with_name(
            f".{config.output_path.name}.{uuid.uuid4().hex}.tmp"
        )
        _write_catalog(
            temporary_path,
            frames=frames,
            sources=sources,
            catalog_version=config.catalog_version.strip(),
            source_lineage=lineage,
            batch_size=config.batch_size,
        )
        _validate_temporary_catalog(temporary_path, expected_frames=len(frames))
        os.replace(temporary_path, config.output_path)
        temporary_path = None
    except FilterCatalogBuildError:
        raise
    except Exception as error:
        raise FilterCatalogBuildError(
            f"Filter catalog build failed: {type(error).__name__}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return FilterCatalogBuildReport(
        output_path=config.output_path,
        catalog_version=config.catalog_version.strip(),
        frame_count=len(frames),
        availability=sources.availability,
        output_size_bytes=config.output_path.stat().st_size,
        build_seconds=time.perf_counter() - started,
    )


def _load_sources(config: FilterCatalogBuildConfig) -> _Sources:
    """Load only configured published artifacts through existing readers."""

    return _Sources(
        frames=FrameArtifactReader(config.frames_path),
        video_metadata=(
            VideoMetadataArtifactReader(config.video_metadata_path)
            if config.video_metadata_path is not None
            else None
        ),
        captions=(
            CaptionArtifactReader(config.caption_path)
            if config.caption_path is not None
            else None
        ),
        ocr=(
            OCRArtifactReader(config.ocr_path)
            if config.ocr_path is not None
            else None
        ),
        objects=(
            ObjectCountsArtifactReader(config.object_counts_path)
            if config.object_counts_path is not None
            else None
        ),
        transcripts=(
            OfflineTranscriptStore(config.transcripts_path)
            if config.transcripts_path is not None
            else None
        ),
        availability=CatalogAvailability(
            title=config.video_metadata_path is not None,
            caption=config.caption_path is not None,
            ocr=config.ocr_path is not None,
            objects=config.object_counts_path is not None,
            asr=config.transcripts_path is not None,
        ),
    )


def _validate_source_identity(
    sources: _Sources,
    canonical: dict[str, CanonicalFrame],
) -> None:
    """Reject specialist evidence that invents or changes frame identity."""

    for store in (sources.captions, sources.ocr, sources.objects):
        if store is None:
            continue
        for record in store.iter_records():
            frame = canonical.get(record.frame_id)
            if frame is None:
                raise FilterCatalogBuildError(
                    f"Source evidence references unknown frame_id {record.frame_id!r}"
                )
            identity = (
                record.video_id,
                record.frame_idx,
                record.timestamp_ms,
            )
            expected = (frame.video_id, frame.frame_idx, frame.timestamp_ms)
            if identity != expected:
                raise FilterCatalogBuildError(
                    f"Source evidence changed canonical identity for {record.frame_id!r}"
                )

    if sources.transcripts is not None:
        video_ids = {frame.video_id for frame in canonical.values()}
        for segment in sources.transcripts.iter_records():
            if segment.video_id not in video_ids:
                raise FilterCatalogBuildError(
                    "Transcript evidence references unknown canonical video_id "
                    f"{segment.video_id!r}"
                )


def _write_catalog(
    path: Path,
    *,
    frames: tuple[CanonicalFrame, ...],
    sources: _Sources,
    catalog_version: str,
    source_lineage: dict[str, Any],
    batch_size: int,
) -> None:
    """Write all canonical frame projections to one unpublished SQLite file."""

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        create_catalog_schema(connection)
        connection.execute(
            """
            INSERT INTO catalog_metadata (
                id, schema_version, catalog_version, built_at, frame_count,
                source_lineage_json, title_available, caption_available,
                ocr_available, objects_available, asr_available
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                CATALOG_SCHEMA_VERSION,
                catalog_version,
                datetime.now(timezone.utc).isoformat(),
                len(frames),
                json.dumps(source_lineage, ensure_ascii=False, sort_keys=True),
                int(sources.availability.title),
                int(sources.availability.caption),
                int(sources.availability.ocr),
                int(sources.availability.objects),
                int(sources.availability.asr),
            ),
        )

        caption_ids = _record_ids(sources.captions)
        ocr_ids = _record_ids(sources.ocr)
        object_ids = _record_ids(sources.objects)
        frame_rows: list[tuple[Any, ...]] = []
        object_rows: list[tuple[str, str, int]] = []
        for frame in frames:
            folder_id = _folder_id(frame.video_id)
            metadata = (
                sources.video_metadata.get(frame.video_id)
                if sources.video_metadata is not None
                else None
            )
            title = metadata.title if metadata is not None else None
            caption = (
                sources.captions.get_text(frame.frame_id)
                if sources.captions is not None and frame.frame_id in caption_ids
                else None
            )
            ocr = (
                sources.ocr.get_text(frame.frame_id)
                if sources.ocr is not None and frame.frame_id in ocr_ids
                else None
            )
            asr = _frame_asr(sources.transcripts, frame)
            frame_rows.append(
                (
                    frame.frame_id,
                    frame.video_id,
                    frame.frame_idx,
                    frame.timestamp_ms,
                    folder_id,
                    title,
                    _normalized_or_none(title),
                    caption,
                    _normalized_or_none(caption),
                    ocr,
                    _normalized_or_none(ocr),
                    asr,
                    _normalized_or_none(asr),
                )
            )
            if sources.objects is not None and frame.frame_id in object_ids:
                counts = sources.objects.get_counts(frame.frame_id)
                if counts is not None:
                    normalized_counts: dict[str, int] = {}
                    for label, count in counts.items():
                        normalized = normalize_filter_text(label)
                        if not normalized:
                            raise FilterCatalogBuildError(
                                f"Object label is blank for {frame.frame_id!r}"
                            )
                        normalized_counts[normalized] = (
                            normalized_counts.get(normalized, 0) + count
                        )
                    object_rows.extend(
                        (frame.frame_id, label, count)
                        for label, count in sorted(normalized_counts.items())
                    )

            if len(frame_rows) >= batch_size:
                _insert_batches(connection, frame_rows, object_rows)
                frame_rows.clear()
                object_rows.clear()

        _insert_batches(connection, frame_rows, object_rows)
        connection.commit()
    finally:
        connection.close()


def _record_ids(store: Any | None) -> set[str]:
    """Return exact IDs present in an optional specialist store."""

    if store is None:
        return set()
    return {record.frame_id for record in store.iter_records()}


def _insert_batches(
    connection: sqlite3.Connection,
    frame_rows: Iterable[tuple[Any, ...]],
    object_rows: Iterable[tuple[str, str, int]],
) -> None:
    """Insert one bounded batch while keeping objects linked to frames."""

    connection.executemany(
        """
        INSERT INTO frames (
            frame_id, video_id, frame_idx, timestamp_ms, folder_id,
            title, title_norm, caption, caption_norm, ocr, ocr_norm, asr, asr_norm
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        frame_rows,
    )
    connection.executemany(
        """
        INSERT INTO frame_objects(frame_id, label_norm, object_count)
        VALUES (?, ?, ?)
        """,
        object_rows,
    )


def _folder_id(video_id: str) -> str:
    """Derive the organizer folder prefix without changing canonical video ID."""

    prefix, separator, _ = video_id.partition("_")
    if not separator or _FOLDER_PATTERN.fullmatch(prefix) is None:
        raise FilterCatalogBuildError(
            f"Canonical video_id has no organizer folder prefix: {video_id!r}"
        )
    return prefix


def _frame_asr(
    store: OfflineTranscriptStore | None, frame: CanonicalFrame
) -> str | None:
    """Join only ordered half-open transcript segments containing this frame."""

    if store is None:
        return None
    text = " ".join(
        segment.text.strip()
        for segment in store.get_at(frame.video_id, frame.timestamp_ms)
        if segment.text.strip()
    )
    return text or None


def _normalized_or_none(value: str | None) -> str | None:
    """Normalize usable display text while retaining missing evidence as null."""

    if value is None:
        return None
    normalized = normalize_filter_text(value)
    return normalized or None


def _validate_temporary_catalog(path: Path, *, expected_frames: int) -> None:
    """Run SQLite integrity and runtime invariants before atomic replacement."""

    connection = sqlite3.connect(path)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise FilterCatalogBuildError(
                f"Temporary Filter catalog integrity check failed: {result}"
            )
    finally:
        connection.close()

    catalog = FilterCatalog.open(path, pool_size=1)
    try:
        if catalog.info.frame_count != expected_frames:
            raise FilterCatalogBuildError(
                "Temporary Filter catalog frame count changed during validation"
            )
    finally:
        catalog.close()


def _build_source_lineage(
    config: FilterCatalogBuildConfig,
    sources: _Sources,
) -> dict[str, Any]:
    """Record supplied lineage, typed versions, and deterministic checksums."""

    lineage: dict[str, Any] = dict(config.source_lineage)
    configured_paths = {
        "frames": config.frames_path,
        "video_metadata": config.video_metadata_path,
        "caption": config.caption_path,
        "ocr": config.ocr_path,
        "objects": config.object_counts_path,
        "transcripts": config.transcripts_path,
    }
    lineage["checksums"] = {
        name: _path_checksum(path)
        for name, path in configured_paths.items()
        if path is not None
    }
    if sources.captions is not None:
        lineage["caption_versions"] = sources.captions.version_identity
    if sources.ocr is not None:
        lineage["ocr_versions"] = sources.ocr.version_identity
    return lineage


def _path_checksum(path: Path) -> str:
    """Hash one file or sorted directory tree for reproducible source lineage."""

    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for file_path in files:
        if not file_path.is_file():
            continue
        if path.is_dir():
            digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
