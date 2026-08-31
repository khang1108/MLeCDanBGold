"""Load published specialist evidence into deterministic runtime indexes.

Caption, OCR, object, and frame-context stores validate runtime projections of
the offline-owned artifact layouts. The legacy frame-aligned ASR artifact is
projected to a compact runtime value so its compatibility contract does not
escape through corpus reads.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Annotated, Any, ClassVar, Generic, Self, TypeVar, cast

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from hcmai.common.utils.io import read_json
from hcmai.retrieval.models import RetrievalSource


_NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class _ArtifactModel(BaseModel):
    """Strict base for runtime validation of published evidence artifacts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _ProcessingStatus(str, Enum):
    """Artifact status values understood by runtime readers."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class _SpecialistArtifact(_ArtifactModel):
    """Shared failure diagnostics on specialist evidence artifacts."""

    status: _ProcessingStatus = _ProcessingStatus.COMPLETED
    error_code: _NonEmptyString | None = None
    error_message: _NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_failure_details(self) -> Self:
        """Require diagnostics when an offline producer recorded failure."""

        if self.status is _ProcessingStatus.FAILED and (
            self.error_code is None or self.error_message is None
        ):
            raise ValueError("failed evidence requires error_code and error_message")
        return self


class _CaptionArtifact(_SpecialistArtifact):
    """Runtime reader view of one published caption artifact row."""

    frame_id: _NonEmptyString
    video_id: _NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    text: str | None = None
    frame_store_id: _NonEmptyString | None = None
    artifact_version: _NonEmptyString
    model_name: _NonEmptyString
    model_revision: _NonEmptyString | None = None


class _OCRArtifact(_SpecialistArtifact):
    """Runtime reader view of one published OCR artifact row."""

    frame_id: _NonEmptyString
    video_id: _NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    raw_text: str | None = None
    normalized_text: str | None = None
    quality_score: float = Field(default=0.0, ge=0, le=1)
    region_count: int = Field(default=0, ge=0)
    frame_store_id: _NonEmptyString | None = None
    artifact_version: _NonEmptyString
    model_name: _NonEmptyString
    model_revision: _NonEmptyString | None = None


class _FrameContextArtifact(_ArtifactModel):
    """Runtime reader view of deterministic frame context."""

    frame_id: _NonEmptyString
    video_id: _NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    caption_text: str | None = None
    ocr_text: str | None = None
    object_summary: str | None = None
    context_text: str | None = None
    caption_available: bool = False
    ocr_quality: float = Field(default=0.0, ge=0, le=1)
    object_count: int = Field(default=0, ge=0)
    context_version: _NonEmptyString
    caption_version: _NonEmptyString
    ocr_version: _NonEmptyString
    object_version: _NonEmptyString
    frame_store_id: _NonEmptyString | None = None


class _ObjectDetectionArtifact(_ArtifactModel):
    """Runtime reader view of one normalized object detection."""

    label: _NonEmptyString
    confidence: float = Field(ge=0, le=1)
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_box(self) -> Self:
        """Reject inverted normalized object boxes."""

        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("object maximum coordinates must not precede minimums")
        return self


class _ObjectArtifact(_SpecialistArtifact):
    """Runtime reader view of one frame's object evidence."""

    frame_id: _NonEmptyString
    video_id: _NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    detections: list[_ObjectDetectionArtifact] = Field(default_factory=list)
    counts: dict[_NonEmptyString, int] = Field(default_factory=dict)
    summary: str | None = None
    detection_count: int = Field(default=0, ge=0)
    frame_store_id: _NonEmptyString | None = None
    artifact_version: _NonEmptyString

    @model_validator(mode="after")
    def validate_detections(self) -> Self:
        """Preserve raw detection multiplicity and count consistency."""

        raw_counts = Counter(detection.label for detection in self.detections)
        if self.detection_count != len(self.detections):
            raise ValueError("detection_count must equal the number of detections")
        if any(
            count < 0 or count > raw_counts.get(label, 0)
            for label, count in self.counts.items()
        ):
            raise ValueError("counts must not exceed raw detection multiplicity")
        return self


class _FrameEnrichmentArtifact(_ArtifactModel):
    """Runtime reader for the deprecated frame-aligned ASR artifact."""

    frame_id: _NonEmptyString
    frame_store_id: _NonEmptyString | None = None
    caption: _NonEmptyString | None = None
    detailed_caption: _NonEmptyString | None = None
    ocr_text: _NonEmptyString | None = None
    asr_text: _NonEmptyString | None = None
    source_segment_ids: list[_NonEmptyString] = Field(default_factory=list)
    enrichment_version: _NonEmptyString | None = None
    objects: list[_NonEmptyString] = Field(default_factory=list)
    model_name: _NonEmptyString
    status: _ProcessingStatus = _ProcessingStatus.COMPLETED
    error_message: _NonEmptyString | None = None


def _usable_completed_text(
    row: _CaptionArtifact | _OCRArtifact,
) -> str | None:
    """Return usable completed specialist text from a runtime artifact view."""

    if row.status is not _ProcessingStatus.COMPLETED:
        return None
    value = row.text if isinstance(row, _CaptionArtifact) else row.normalized_text
    return value if value is not None and value.strip() else None


_EvidenceT = TypeVar("_EvidenceT", bound=BaseModel)
_NULLABLE_FIELDS = (
    "caption",
    "detailed_caption",
    "ocr_text",
    "asr_text",
    "enrichment_version",
    "error_message",
)


def _require_file(path: str | Path, artifact: str) -> Path:
    """Return an existing artifact file or raise a clear path error."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{artifact} artifact is not a file: {resolved}")
    return resolved


def _nullable_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas null scalars to ``None`` before contract validation."""

    values = table.astype(object).where(table.notna(), None)
    return cast(list[dict[str, Any]], values.to_dict(orient="records"))


def _strict_int(value: object, field: str) -> int:
    """Reject boolean, floating-point, and string integer representations."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _canonical_string(value: object, field: str) -> str:
    """Require identity strings to already use their canonical representation."""

    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must use its canonical representation")
    return value


def _frame_identity(row: dict[str, Any]) -> tuple[str, str, int, int]:
    """Validate the exact stored representation of canonical frame identity."""

    return (
        _canonical_string(row.get("frame_id"), "frame_id"),
        _canonical_string(row.get("video_id"), "video_id"),
        _strict_int(row.get("frame_idx"), "frame_idx"),
        _strict_int(row.get("timestamp_ms"), "timestamp_ms"),
    )


def _index_records(
    records: Iterable[_EvidenceT], artifact_path: Path
) -> dict[str, _EvidenceT]:
    """Index validated canonical IDs without allowing normalized collisions."""

    record_list = tuple(records)
    indexed = {
        cast(str, getattr(record, "frame_id")): record for record in record_list
    }
    if len(indexed) != len(record_list):
        raise ValueError(
            f"Duplicate frame_id values after normalization in {artifact_path}"
        )
    return indexed


def _adjacent_manifest(artifact_path: Path) -> dict[str, Any] | None:
    """Load an adjacent bundle manifest when the producer published one."""

    manifest_path = artifact_path.with_name("manifest.json")
    if not manifest_path.exists():
        return None
    try:
        value = read_json(manifest_path)
    except Exception as error:
        raise ValueError(
            f"Malformed adjacent manifest for {artifact_path}: {manifest_path}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"Adjacent manifest must contain an object: {manifest_path}"
        )
    return cast(dict[str, Any], value)


def _uniform_field(
    records: Iterable[BaseModel],
    field: str,
    artifact_path: Path,
) -> object | None:
    """Return one uniform row value or reject mixed artifact identity."""

    values = {getattr(record, field) for record in records}
    if len(values) > 1:
        category = "lineage" if field == "frame_store_id" else "version"
        raise ValueError(
            f"{artifact_path} requires uniform {field} {category}"
        )
    return next(iter(values), None)


def _validate_artifact_identity(
    records: Iterable[BaseModel],
    artifact_path: Path,
    version_fields: tuple[str, ...],
) -> tuple[str | None, dict[str, str]]:
    """Validate uniform row lineage/version and its adjacent manifest."""

    record_list = tuple(records)
    lineage_value = _uniform_field(
        record_list, "frame_store_id", artifact_path
    )
    lineage = cast(str | None, lineage_value)
    versions = {
        field: cast(str, _uniform_field(record_list, field, artifact_path))
        for field in version_fields
        if record_list
    }
    manifest = _adjacent_manifest(artifact_path)
    if manifest is None:
        return lineage, versions

    for field, expected in versions.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"Adjacent manifest {field} does not match rows in "
                f"{artifact_path}"
            )
    manifest_lineage = manifest.get("frame_store_id")
    if manifest_lineage is not None and (
        not isinstance(manifest_lineage, str)
        or not manifest_lineage
        or manifest_lineage.strip() != manifest_lineage
    ):
        raise ValueError(
            f"Adjacent manifest frame_store_id is invalid for {artifact_path}"
        )
    if record_list and manifest_lineage != lineage:
        raise ValueError(
            "Adjacent manifest frame_store_id does not match rows in "
            f"{artifact_path}"
        )
    if not record_list:
        lineage = cast(str | None, manifest_lineage)
        for field in version_fields:
            value = manifest.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Adjacent manifest {field} is invalid for {artifact_path}"
                )
            versions[field] = value
    return lineage, versions


class _TypedEvidenceStore(Generic[_EvidenceT]):
    """Validate one typed Parquet artifact and index exact frame IDs."""

    def __init__(
        self,
        artifact_path: str | Path,
        contract: type[_EvidenceT],
        version_fields: tuple[str, ...],
    ) -> None:
        self.artifact_path = _require_file(artifact_path, contract.__name__)
        table = pd.read_parquet(self.artifact_path)
        if "frame_id" not in table.columns:
            raise ValueError(f"{self.artifact_path} is missing column: frame_id")
        records: list[_EvidenceT] = []
        for index, row in enumerate(_nullable_rows(table)):
            frame_id, video_id, frame_idx, timestamp_ms = _frame_identity(row)
            try:
                record = contract.model_validate(row)
            except Exception as error:
                raise ValueError(
                    f"Malformed {contract.__name__} row {index} "
                    f"in {self.artifact_path}"
                ) from error
            if (
                getattr(record, "frame_id") != frame_id
                or getattr(record, "video_id") != video_id
                or getattr(record, "frame_idx") != frame_idx
                or getattr(record, "timestamp_ms") != timestamp_ms
            ):
                raise ValueError(
                    f"{contract.__name__} row {index} changed canonical identity"
                )
            records.append(record)
        self._records = tuple(records)
        self._by_frame_id = _index_records(self._records, self.artifact_path)
        self.frame_store_id, self.version_identity = _validate_artifact_identity(
            self._records,
            self.artifact_path,
            version_fields,
        )

    def __len__(self) -> int:
        """Return the number of validated evidence rows."""

        return len(self._records)

    def get(self, frame_id: str) -> _EvidenceT:
        """Return typed evidence for an exact canonical frame ID."""

        try:
            return self._by_frame_id[frame_id]
        except KeyError:
            raise KeyError(
                f"Unknown frame_id {frame_id!r} in {self.artifact_path}"
            ) from None

    def get_many(self, frame_ids: Iterable[str]) -> list[_EvidenceT]:
        """Return typed rows in requested order while preserving duplicates."""

        return [self.get(frame_id) for frame_id in frame_ids]

    def iter_records(self) -> Iterator[_EvidenceT]:
        """Iterate typed rows in deterministic artifact order."""

        return iter(self._records)


class CaptionStore(_TypedEvidenceStore[_CaptionArtifact]):
    """Provide typed, indexed access to authoritative caption evidence."""

    source = RetrievalSource.CAPTION

    def __init__(self, artifact_path: str | Path) -> None:
        """Load and validate a ``CaptionEvidence`` Parquet artifact."""

        super().__init__(artifact_path, _CaptionArtifact, ("artifact_version",))

    def get_text(self, frame_id: str) -> str | None:
        """Return non-empty text from completed caption evidence."""

        return _usable_completed_text(self.get(frame_id))


class OCRStore(_TypedEvidenceStore[_OCRArtifact]):
    """Provide typed, indexed access to authoritative OCR evidence."""

    source = RetrievalSource.OCR

    def __init__(self, artifact_path: str | Path) -> None:
        """Load and validate an ``OCREvidence`` Parquet artifact."""

        super().__init__(artifact_path, _OCRArtifact, ("artifact_version",))

    def get_text(self, frame_id: str) -> str | None:
        """Return non-empty normalized text from completed OCR evidence."""

        return _usable_completed_text(self.get(frame_id))


class FrameContextStore(_TypedEvidenceStore[_FrameContextArtifact]):
    """Provide typed access to deterministic derived frame context."""

    def __init__(self, artifact_path: str | Path) -> None:
        """Load and validate a ``FrameContext`` Parquet artifact."""

        super().__init__(
            artifact_path,
            _FrameContextArtifact,
            (
                "context_version",
                "caption_version",
                "ocr_version",
                "object_version",
            ),
        )

    def get_text(self, frame_id: str) -> str | None:
        """Return non-empty deterministic context text, when available."""

        value = self.get(frame_id).context_text
        return value if value is not None and value.strip() else None


def _object_counts(value: object) -> dict[str, int]:
    """Decode deterministic object counts from the flattened frame row."""

    if not isinstance(value, str):
        raise ValueError("counts_json must be a serialized JSON string")
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("counts_json must contain valid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("counts_json must contain an object")
    counts: dict[str, int] = {}
    for label, count in raw.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("object count labels must be non-empty strings")
        counts[label] = _strict_int(count, f"count for {label!r}")
    return counts


class ObjectStore(_TypedEvidenceStore[_ObjectArtifact]):
    """Reconstruct strict object evidence from frame and detection artifacts.

    The public frame artifact stores counts and summaries, while its sibling
    ``detections.parquet`` retains the real detections. Both are required for
    non-empty evidence because the store never invents detection records.
    """

    def __init__(self, artifact_path: str | Path) -> None:
        """Load a flattened object frame artifact and its sibling detections."""

        self.artifact_path = _require_file(artifact_path, "ObjectEvidence")
        frame_table = pd.read_parquet(self.artifact_path)
        required = {
            "frame_id",
            "video_id",
            "frame_idx",
            "timestamp_ms",
            "counts_json",
            "detection_count",
            "artifact_version",
        }
        missing = sorted(required.difference(frame_table.columns))
        if missing:
            raise ValueError(
                f"{self.artifact_path} is missing columns: {', '.join(missing)}"
            )
        frame_rows = _nullable_rows(frame_table)
        identities = [_frame_identity(row) for row in frame_rows]
        detections = self._load_detections(frame_rows)
        records: list[_ObjectArtifact] = []
        for index, (row, identity) in enumerate(
            zip(frame_rows, identities, strict=True)
        ):
            frame_id, video_id, frame_idx, timestamp_ms = identity
            values = dict(row)
            values["counts"] = _object_counts(values.pop("counts_json", None))
            values["detection_count"] = _strict_int(
                values.get("detection_count"), "detection_count"
            )
            values["detections"] = detections.get(frame_id, [])
            try:
                record = _ObjectArtifact.model_validate(values)
            except Exception as error:
                raise ValueError(
                    f"Malformed ObjectEvidence row {index} in {self.artifact_path}"
                ) from error
            if (
                record.frame_id != frame_id
                or record.video_id != video_id
                or record.frame_idx != frame_idx
                or record.timestamp_ms != timestamp_ms
            ):
                raise ValueError(
                    f"ObjectEvidence row {index} changed canonical identity"
                )
            records.append(record)
        self._records = tuple(records)
        self._by_frame_id = _index_records(self._records, self.artifact_path)
        self.frame_store_id, self.version_identity = _validate_artifact_identity(
            self._records,
            self.artifact_path,
            ("artifact_version",),
        )

    def _load_detections(
        self, frame_rows: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, object | None]]]:
        """Load real detections and validate their frame identity and order."""

        detection_path = self.artifact_path.with_name("detections.parquet")
        expected_count = sum(
            _strict_int(row.get("detection_count"), "detection_count")
            for row in frame_rows
        )
        if not detection_path.is_file():
            if expected_count:
                raise FileNotFoundError(
                    "ObjectEvidence with detections requires sibling artifact: "
                    f"{detection_path}"
                )
            return {}

        table = pd.read_parquet(detection_path)
        required = {
            "frame_id",
            "video_id",
            "frame_idx",
            "timestamp_ms",
            "detection_index",
            *_ObjectDetectionArtifact.model_fields,
        }
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(
                f"{detection_path} is missing columns: {', '.join(missing)}"
            )
        rows = _nullable_rows(table)
        identities = [
            (
                _canonical_string(row.get("frame_id"), "frame_id"),
                _strict_int(row.get("detection_index"), "detection_index"),
            )
            for row in rows
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(
                f"Duplicate object detection identity in {detection_path}"
            )

        frame_identity = {
            frame_id: (video_id, frame_idx, timestamp_ms)
            for frame_id, video_id, frame_idx, timestamp_ms in map(
                _frame_identity, frame_rows
            )
        }
        grouped: defaultdict[
            str, list[tuple[int, dict[str, object | None]]]
        ] = defaultdict(list)
        for row in rows:
            frame_id = _canonical_string(row.get("frame_id"), "frame_id")
            video_id = _canonical_string(row.get("video_id"), "video_id")
            frame_idx = _strict_int(row.get("frame_idx"), "frame_idx")
            timestamp_ms = _strict_int(row.get("timestamp_ms"), "timestamp_ms")
            if frame_id not in frame_identity:
                raise ValueError(
                    f"Object detection references unknown frame_id {frame_id!r}"
                )
            if (video_id, frame_idx, timestamp_ms) != frame_identity[frame_id]:
                raise ValueError(
                    f"Object detection canonical identity mismatch for frame_id {frame_id!r}"
                )
            order = _strict_int(row.get("detection_index"), "detection_index")
            detection = {
                field: row.get(field)
                for field in _ObjectDetectionArtifact.model_fields
            }
            try:
                _ObjectDetectionArtifact.model_validate(detection)
            except Exception as error:
                raise ValueError(
                    f"Malformed object detection for frame_id {frame_id!r}"
                ) from error
            grouped[frame_id].append((order, detection))
        detections: dict[str, list[dict[str, object | None]]] = {}
        for frame_id, items in grouped.items():
            indices = [index for index, _ in items]
            if indices != list(range(len(items))):
                raise ValueError(
                    "Object detections require contiguous detection_index "
                    f"values for frame_id {frame_id!r}"
                )
            ordered = sorted(items, key=lambda pair: pair[0])
            detections[frame_id] = [item for _, item in ordered]
        return detections


@dataclass(frozen=True, slots=True)
class _ASRText:
    """Usable legacy ASR text for one frame without artifact provenance."""

    frame_id: str
    text: str | None


class ASRStore:
    """Indexed access to the temporary frame-aligned ASR compatibility view."""

    source: ClassVar[RetrievalSource] = RetrievalSource.ASR
    text_field: ClassVar[str] = "asr_text"

    def __init__(self, artifact_path: str | Path) -> None:
        """Load and validate the derived frame-aligned ASR artifact."""

        self.artifact_path = _require_file(artifact_path, "ASR")
        table = pd.read_parquet(self.artifact_path)
        required = {"frame_id", "model_name", self.text_field}
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(
                f"{self.artifact_path} is missing columns: {', '.join(missing)}"
            )
        records = tuple(
            _asr_runtime_value(row) for row in table.to_dict(orient="records")
        )
        self._by_frame_id: dict[str, _ASRText] = {}
        for record in records:
            if record.frame_id in self._by_frame_id:
                raise ValueError(
                    f"Duplicate frame_id {record.frame_id!r} in {self.artifact_path}"
                )
            self._by_frame_id[record.frame_id] = record
        self._records = records

    def __len__(self) -> int:
        """Return the number of compatibility rows."""

        return len(self._records)

    def get(self, frame_id: str) -> _ASRText:
        """Return compact usable ASR text for an exact frame ID."""

        try:
            return self._by_frame_id[frame_id]
        except KeyError:
            raise KeyError(
                f"Unknown frame_id {frame_id!r} in {self.artifact_path}"
            ) from None

    def get_many(self, frame_ids: Iterable[str]) -> list[_ASRText]:
        """Return rows in requested order while preserving duplicates."""

        return [self.get(frame_id) for frame_id in frame_ids]

    def get_text(self, frame_id: str) -> str | None:
        """Return usable ASR text, excluding failed compatibility rows."""

        return self.get(frame_id).text

    def iter_records(self) -> Iterator[_ASRText]:
        """Iterate compact ASR values in deterministic artifact order."""

        return iter(self._records)


def _is_null(value: object) -> bool:
    """Return whether one legacy scalar encodes a missing value."""

    if value is None or value is pd.NA:
        return True
    return isinstance(value, Real) and math.isnan(float(value))


def _asr_runtime_value(data: dict[Hashable, Any]) -> _ASRText:
    """Validate one legacy row and project its usable text for runtime reads."""

    values: dict[str, object] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Enrichment column name must be a string: {key!r}")
        values[key] = value
    objects = values.get("objects")
    to_list = getattr(objects, "tolist", None)
    if callable(to_list):
        values["objects"] = to_list()
    elif _is_null(objects):
        values["objects"] = []
    for field in _NULLABLE_FIELDS:
        if _is_null(values.get(field)):
            values[field] = None
    artifact = _FrameEnrichmentArtifact.model_validate(values)
    usable_text = (
        artifact.asr_text
        if artifact.status is _ProcessingStatus.COMPLETED
        and artifact.error_message is None
        else None
    )
    return _ASRText(frame_id=artifact.frame_id, text=usable_text)


__all__ = [
    "ASRStore",
    "CaptionStore",
    "FrameContextStore",
    "ObjectStore",
    "OCRStore",
]
