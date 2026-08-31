"""Read offline artifacts without coupling builders to the runtime corpus.

This module owns the small, deterministic projections that offline producers
and index builders need from published frame, specialist-evidence, and
transcript artifacts.  It deliberately does not provide runtime search or
asset-serving APIs; ``hcmai.corpus`` remains the read facade for online use.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hcmai.common.utils.io import read_json
from offline.enrichment.caption.models.evidence import (
    CaptionEvidence,
    usable_completed_text as usable_caption_text,
)
from offline.enrichment.context.models import FrameContext
from offline.enrichment.models import FrameEnrichment, ProcessingStatus
from offline.enrichment.ocr.models.evidence import (
    OCREvidence,
    usable_completed_text as usable_ocr_text,
)
from offline.enrichment.transcripts.artifacts import load_transcript_artifact_records
from offline.enrichment.transcripts.models import TranscriptSegment


@dataclass(frozen=True, slots=True)
class CanonicalFrame:
    """Minimal canonical frame projection used by offline artifact stages.

    ``frame_idx`` is the organizer-facing coordinate and must never be
    substituted with Parquet order or BTC keyframe order.
    """

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    image_path: str
    thumbnail_path: str | None = None


class _CanonicalFrameRow(BaseModel):
    """Validate the stable frame fields without rejecting offline provenance."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    frame_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    image_path: str = Field(min_length=1)
    thumbnail_path: str | None = None


def _nullable_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas null scalars to ``None`` before contract validation."""

    values = table.astype(object).where(table.notna(), None)
    return cast(list[dict[str, Any]], values.to_dict(orient="records"))


def _canonical_string(value: object, field: str) -> str:
    """Require identity text to already be in its stored canonical form."""

    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must use its canonical representation")
    return value


def _strict_int(value: object, field: str) -> int:
    """Reject boolean, string, and floating-point identity coordinates."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _frame_identity(row: dict[str, Any]) -> tuple[str, str, int, int]:
    """Validate one raw artifact identity before Pydantic can normalize it."""

    return (
        _canonical_string(row.get("frame_id"), "frame_id"),
        _canonical_string(row.get("video_id"), "video_id"),
        _strict_int(row.get("frame_idx"), "frame_idx"),
        _strict_int(row.get("timestamp_ms"), "timestamp_ms"),
    )


def _adjacent_manifest(path: Path) -> dict[str, Any] | None:
    """Load an optional sibling manifest and reject malformed lineage metadata."""

    manifest_path = path.with_name("manifest.json")
    if not manifest_path.exists():
        return None
    try:
        manifest = read_json(manifest_path)
    except Exception as error:
        raise ValueError(f"Malformed adjacent manifest for {path}: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Adjacent manifest must contain an object: {manifest_path}")
    return cast(dict[str, Any], manifest)


def _manifest_lineage(path: Path) -> str | None:
    """Return a valid optional frame-store lineage from a sibling manifest."""

    manifest = _adjacent_manifest(path)
    if manifest is None:
        return None
    lineage = manifest.get("frame_store_id")
    if lineage is None:
        return None
    return _canonical_string(lineage, "manifest frame_store_id")


class FrameArtifactReader:
    """Validate and index canonical frame rows for offline construction only."""

    def __init__(self, metadata_path: str | Path) -> None:
        """Load canonical frames while retaining their deterministic file order."""

        self.metadata_path = Path(metadata_path)
        rows = _nullable_rows(pd.read_parquet(self.metadata_path))
        records: list[CanonicalFrame] = []
        for index, row in enumerate(rows):
            frame_id, video_id, frame_idx, timestamp_ms = _frame_identity(row)
            try:
                artifact = _CanonicalFrameRow.model_validate(row)
            except Exception as error:
                raise ValueError(
                    f"Malformed canonical frame row {index} in {self.metadata_path}"
                ) from error
            if (
                artifact.frame_id,
                artifact.video_id,
                artifact.frame_idx,
                artifact.timestamp_ms,
            ) != (frame_id, video_id, frame_idx, timestamp_ms):
                raise ValueError(f"Canonical frame row {index} changed canonical identity")
            records.append(
                CanonicalFrame(
                    frame_id=artifact.frame_id,
                    video_id=artifact.video_id,
                    frame_idx=artifact.frame_idx,
                    timestamp_ms=artifact.timestamp_ms,
                    image_path=artifact.image_path,
                    thumbnail_path=artifact.thumbnail_path,
                )
            )
        self._records = tuple(records)
        self._by_frame_id = {record.frame_id: record for record in self._records}
        if len(self._by_frame_id) != len(self._records):
            raise ValueError(f"Duplicate frame_id values in {self.metadata_path}")
        self.frame_store_id = _manifest_lineage(self.metadata_path)

    @classmethod
    def load(cls, metadata_path: str | Path) -> "FrameArtifactReader":
        """Load canonical frame metadata from its published Parquet artifact."""

        return cls(metadata_path)

    def __len__(self) -> int:
        """Return the number of canonical frame records."""

        return len(self._records)

    def get(self, frame_id: str) -> CanonicalFrame:
        """Return one exact canonical frame or raise a contextual key error."""

        try:
            return self._by_frame_id[frame_id]
        except KeyError:
            raise KeyError(
                f"Unknown frame_id {frame_id!r} in {self.metadata_path}"
            ) from None

    def iter_frames(self) -> Iterator[CanonicalFrame]:
        """Iterate canonical records in their published Parquet order."""

        return iter(self._records)


_EvidenceT = TypeVar(
    "_EvidenceT", CaptionEvidence, OCREvidence, FrameContext
)


def _uniform_lineage(records: Iterable[object], path: Path) -> str | None:
    """Require one artifact-level lineage value across all specialist rows."""

    values = {getattr(record, "frame_store_id") for record in records}
    if len(values) > 1:
        raise ValueError(f"{path} requires uniform frame_store_id lineage")
    return cast(str | None, next(iter(values), None))


class _SpecialistArtifactReader(Generic[_EvidenceT]):
    """Common strict reader for frame-native specialist evidence artifacts."""

    contract: type[_EvidenceT]
    version_fields: tuple[str, ...]

    def __init__(self, artifact_path: str | Path) -> None:
        """Validate all rows, identity fields, versions, and sibling lineage."""

        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.is_file():
            raise FileNotFoundError(
                f"{self.contract.__name__} artifact is not a file: {self.artifact_path}"
            )
        table = pd.read_parquet(self.artifact_path)
        if "frame_id" not in table.columns:
            raise ValueError(f"{self.artifact_path} is missing column: frame_id")
        records: list[_EvidenceT] = []
        for index, row in enumerate(_nullable_rows(table)):
            identity = _frame_identity(row)
            try:
                record = self.contract.model_validate(row)
            except Exception as error:
                raise ValueError(
                    f"Malformed {self.contract.__name__} row {index} in {self.artifact_path}"
                ) from error
            if (
                record.frame_id,
                record.video_id,
                record.frame_idx,
                record.timestamp_ms,
            ) != identity:
                raise ValueError(
                    f"{self.contract.__name__} row {index} changed canonical identity"
                )
            records.append(record)
        self._records = tuple(records)
        self._by_frame_id = {record.frame_id: record for record in self._records}
        if len(self._by_frame_id) != len(self._records):
            raise ValueError(
                f"Duplicate frame_id values after normalization in {self.artifact_path}"
            )
        self.frame_store_id = _uniform_lineage(self._records, self.artifact_path)
        self._validate_manifest()

    def _validate_manifest(self) -> None:
        """Reject sibling manifests that disagree with artifact rows."""

        manifest = _adjacent_manifest(self.artifact_path)
        if manifest is None:
            return
        for field in self.version_fields:
            values = {getattr(record, field) for record in self._records}
            if len(values) > 1:
                raise ValueError(f"{self.artifact_path} requires uniform {field}")
            expected = next(iter(values), None)
            if expected is None:
                value = manifest.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Adjacent manifest {field} is invalid for {self.artifact_path}"
                    )
                continue
            if manifest.get(field) != expected:
                raise ValueError(
                    f"Adjacent manifest {field} does not match rows in {self.artifact_path}"
                )
        manifest_lineage = manifest.get("frame_store_id")
        if manifest_lineage is not None:
            manifest_lineage = _canonical_string(
                manifest_lineage, "manifest frame_store_id"
            )
        if self._records and manifest_lineage != self.frame_store_id:
            raise ValueError(
                "Adjacent manifest frame_store_id does not match rows in "
                f"{self.artifact_path}"
            )

    def get(self, frame_id: str) -> _EvidenceT:
        """Return one typed specialist row by its canonical frame identity."""

        try:
            return self._by_frame_id[frame_id]
        except KeyError:
            raise KeyError(
                f"Unknown frame_id {frame_id!r} in {self.artifact_path}"
            ) from None

    def iter_records(self) -> Iterator[_EvidenceT]:
        """Iterate typed specialist evidence in deterministic artifact order."""

        return iter(self._records)


class CaptionArtifactReader(_SpecialistArtifactReader[CaptionEvidence]):
    """Offline reader for authoritative caption evidence."""

    contract = CaptionEvidence
    version_fields = ("artifact_version",)

    def get_text(self, frame_id: str) -> str | None:
        """Return usable completed caption text without fabricating evidence."""

        return usable_caption_text(self.get(frame_id))


class OCRArtifactReader(_SpecialistArtifactReader[OCREvidence]):
    """Offline reader for authoritative OCR evidence."""

    contract = OCREvidence
    version_fields = ("artifact_version",)

    def get_text(self, frame_id: str) -> str | None:
        """Return usable completed normalized OCR text."""

        return usable_ocr_text(self.get(frame_id))


class FrameContextArtifactReader(_SpecialistArtifactReader[FrameContext]):
    """Offline reader for deterministic frame context evidence."""

    contract = FrameContext
    version_fields = (
        "context_version",
        "caption_version",
        "ocr_version",
        "object_version",
    )

    def get_text(self, frame_id: str) -> str | None:
        """Return a non-empty deterministic context string when present."""

        value = self.get(frame_id).context_text
        return value if value is not None and value.strip() else None


@dataclass(frozen=True, slots=True)
class ASRTextArtifact:
    """Compact completed ASR compatibility text retained for offline indexing."""

    frame_id: str
    text: str | None


class ASRArtifactReader:
    """Read the legacy frame-aligned ASR projection for compatibility indexes."""

    def __init__(self, artifact_path: str | Path) -> None:
        """Validate one compatibility artifact without inventing frame metadata."""

        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.is_file():
            raise FileNotFoundError(f"ASR artifact is not a file: {self.artifact_path}")
        table = pd.read_parquet(self.artifact_path)
        required = {"frame_id", "model_name", "asr_text"}
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(
                f"{self.artifact_path} is missing columns: {', '.join(missing)}"
            )
        records: list[ASRTextArtifact] = []
        for index, row in enumerate(_nullable_rows(table)):
            frame_id = _canonical_string(row.get("frame_id"), "frame_id")
            values = dict(row)
            objects = values.get("objects")
            if hasattr(objects, "tolist"):
                values["objects"] = objects.tolist()
            if values.get("objects") is None:
                values["objects"] = []
            try:
                artifact = FrameEnrichment.model_validate(values)
            except Exception as error:
                raise ValueError(
                    f"Malformed ASR compatibility row {index} in {self.artifact_path}"
                ) from error
            if artifact.frame_id != frame_id:
                raise ValueError(f"ASR row {index} changed canonical frame_id")
            text = (
                artifact.asr_text
                if artifact.status is ProcessingStatus.COMPLETED
                and artifact.error_message is None
                else None
            )
            records.append(ASRTextArtifact(frame_id=artifact.frame_id, text=text))
        self._records = tuple(records)
        self._by_frame_id = {record.frame_id: record for record in self._records}
        if len(self._by_frame_id) != len(self._records):
            raise ValueError(f"Duplicate frame_id values in {self.artifact_path}")

    def get(self, frame_id: str) -> ASRTextArtifact:
        """Return completed ASR text for one exact canonical frame ID."""

        try:
            return self._by_frame_id[frame_id]
        except KeyError:
            raise KeyError(
                f"Unknown frame_id {frame_id!r} in {self.artifact_path}"
            ) from None

    def get_text(self, frame_id: str) -> str | None:
        """Return usable text while retaining failed/no-text rows as absent evidence."""

        return self.get(frame_id).text

    def iter_records(self) -> Iterator[ASRTextArtifact]:
        """Iterate compatibility values in deterministic artifact order."""

        return iter(self._records)


class OfflineTranscriptStore:
    """Provide offline timeline lookup over complete transcript artifacts."""

    def __init__(self, metadata_path: str | Path) -> None:
        """Load completed transcript segments and arrange them by canonical video."""

        self.metadata_path = Path(metadata_path)
        self._records = tuple(
            record
            for record in load_transcript_artifact_records(self.metadata_path)
            if record.status is ProcessingStatus.COMPLETED
        )
        self._by_segment_id = {record.segment_id: record for record in self._records}
        if len(self._by_segment_id) != len(self._records):
            raise ValueError(f"Duplicate segment_id values in {self.metadata_path}")
        by_video: defaultdict[str, list[TranscriptSegment]] = defaultdict(list)
        for record in self._records:
            by_video[record.video_id].append(record)
        self._by_video = {
            video_id: tuple(
                sorted(
                    records,
                    key=lambda record: (
                        record.start_ms,
                        record.end_ms,
                        record.segment_index,
                        record.segment_id,
                    ),
                )
            )
            for video_id, records in by_video.items()
        }

    def iter_records(self) -> Iterator[TranscriptSegment]:
        """Iterate completed transcript artifact rows in deterministic file order."""

        return iter(self._records)

    def get_by_video(self, video_id: str) -> tuple[TranscriptSegment, ...]:
        """Return one video's completed segments in deterministic time order."""

        return self._by_video.get(video_id, ())

    def get_at(self, video_id: str, timestamp_ms: int) -> list[TranscriptSegment]:
        """Return segments containing a timestamp under half-open semantics."""

        return [
            record
            for record in self._by_video.get(video_id, ())
            if record.start_ms <= timestamp_ms < record.end_ms
        ]

    def get_in_range(
        self, video_id: str, start_ms: int, end_ms: int
    ) -> list[TranscriptSegment]:
        """Return completed segments overlapping a half-open time interval."""

        return [
            record
            for record in self._by_video.get(video_id, ())
            if record.start_ms < end_ms and record.end_ms > start_ms
        ]


class FrameAssetError(OSError):
    """A frame image cannot be safely materialized by an offline stage."""


class FrameAssetMissingError(FrameAssetError, FileNotFoundError):
    """A resolved offline frame image is not an existing regular file."""


class FrameAssetOutsideRootError(FrameAssetError, PermissionError):
    """A frame image path escapes the configured offline dataset root."""


class OfflineFrameAssetResolver:
    """Resolve canonical keyframe image paths for offline model execution."""

    def __init__(self, dataset_root: str | Path) -> None:
        """Bind resolution to one dataset root to prevent path traversal."""

        self.dataset_root = Path(dataset_root).expanduser().resolve()

    def resolve_value(self, value: str | Path, *, require_file: bool = True) -> Path:
        """Resolve a relative or legacy absolute keyframe path within the root."""

        path = Path(value).expanduser()
        resolved = (
            self._resolve_absolute(path)
            if path.is_absolute()
            else (self.dataset_root / path).resolve()
        )
        if not resolved.is_relative_to(self.dataset_root):
            raise FrameAssetOutsideRootError(
                "frame asset escapes dataset root: "
                f"source={path} resolved={resolved} dataset_root={self.dataset_root}"
            )
        if require_file and not resolved.is_file():
            raise FrameAssetMissingError(
                "frame asset is not available: "
                f"source={path} resolved={resolved} dataset_root={self.dataset_root}"
            )
        return resolved

    def _resolve_absolute(self, path: Path) -> Path:
        """Rebase an untrusted legacy absolute keyframe path under this root."""

        resolved = path.resolve()
        if resolved.is_relative_to(self.dataset_root):
            return resolved
        try:
            keyframes_index = resolved.parts.index("keyframes")
        except ValueError:
            return resolved
        return (self.dataset_root / Path(*resolved.parts[keyframes_index:])).resolve()


def assert_frame_identity(record: object, frame: object, artifact: str) -> None:
    """Reject an evidence row whose canonical identity differs from its frame."""

    names = ("frame_id", "video_id", "frame_idx", "timestamp_ms")
    actual = tuple(getattr(record, name) for name in names)
    expected = tuple(getattr(frame, name) for name in names)
    if actual != expected:
        raise ValueError(
            f"{artifact} canonical identity mismatch for frame_id {expected[0]!r}: "
            f"artifact={actual!r} frame={expected!r}"
        )


def assert_matching_lineage(frames: object, evidence: object, artifact: str) -> None:
    """Require specialist lineage to match canonical frame-store lineage exactly."""

    expected = getattr(frames, "frame_store_id", None)
    actual = getattr(evidence, "frame_store_id", None)
    if expected != actual:
        raise ValueError(
            f"{artifact} frame_store_id lineage mismatch: expected {expected!r}, got {actual!r}"
        )


__all__ = [
    "ASRArtifactReader",
    "ASRTextArtifact",
    "CanonicalFrame",
    "CaptionArtifactReader",
    "FrameArtifactReader",
    "FrameAssetError",
    "FrameContextArtifactReader",
    "OCRArtifactReader",
    "OfflineFrameAssetResolver",
    "OfflineTranscriptStore",
    "assert_frame_identity",
    "assert_matching_lineage",
]
