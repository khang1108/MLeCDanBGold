"""Load specialist evidence artifacts into deterministic typed indexes.

Caption, OCR, object, and frame-context stores expose their authoritative
public contracts. ASR intentionally retains the temporary frame-aligned
``FrameEnrichment`` view until text retrieval migrates to timeline evidence.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast

import pandas as pd
from pydantic import BaseModel

from hcmai.common.schemas import (
    CaptionEvidence,
    FrameContext,
    FrameEnrichment,
    ObjectDetection,
    ObjectEvidence,
    OCREvidence,
    ProcessingStatus,
    RetrievalSource,
    usable_completed_text,
)
from hcmai.common.utils.io import read_json


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


class CaptionStore(_TypedEvidenceStore[CaptionEvidence]):
    """Provide typed, indexed access to authoritative caption evidence."""

    source = RetrievalSource.CAPTION

    def __init__(self, artifact_path: str | Path) -> None:
        """Load and validate a ``CaptionEvidence`` Parquet artifact."""

        super().__init__(artifact_path, CaptionEvidence, ("artifact_version",))

    def get_text(self, frame_id: str) -> str | None:
        """Return non-empty text from completed caption evidence."""

        return usable_completed_text(self.get(frame_id))


class OCRStore(_TypedEvidenceStore[OCREvidence]):
    """Provide typed, indexed access to authoritative OCR evidence."""

    source = RetrievalSource.OCR

    def __init__(self, artifact_path: str | Path) -> None:
        """Load and validate an ``OCREvidence`` Parquet artifact."""

        super().__init__(artifact_path, OCREvidence, ("artifact_version",))

    def get_text(self, frame_id: str) -> str | None:
        """Return non-empty normalized text from completed OCR evidence."""

        return usable_completed_text(self.get(frame_id))


class FrameContextStore(_TypedEvidenceStore[FrameContext]):
    """Provide typed access to deterministic derived frame context."""

    def __init__(self, artifact_path: str | Path) -> None:
        """Load and validate a ``FrameContext`` Parquet artifact."""

        super().__init__(
            artifact_path,
            FrameContext,
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


class ObjectStore(_TypedEvidenceStore[ObjectEvidence]):
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
        records: list[ObjectEvidence] = []
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
                record = ObjectEvidence.model_validate(values)
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
            *ObjectDetection.model_fields,
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
                field: row.get(field) for field in ObjectDetection.model_fields
            }
            try:
                ObjectDetection.model_validate(detection)
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
            _materialize(row) for row in table.to_dict(orient="records")
        )
        self._by_frame_id: dict[str, FrameEnrichment] = {}
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

    def get(self, frame_id: str) -> FrameEnrichment:
        """Return the compatibility row for an exact frame ID."""

        try:
            return self._by_frame_id[frame_id]
        except KeyError:
            raise KeyError(
                f"Unknown frame_id {frame_id!r} in {self.artifact_path}"
            ) from None

    def get_many(self, frame_ids: Iterable[str]) -> list[FrameEnrichment]:
        """Return rows in requested order while preserving duplicates."""

        return [self.get(frame_id) for frame_id in frame_ids]

    def get_text(self, frame_id: str) -> str | None:
        """Return usable ASR text, excluding failed compatibility rows."""

        record = self.get(frame_id)
        if (
            record.status != ProcessingStatus.COMPLETED
            or record.error_message is not None
        ):
            return None
        return record.asr_text

    def iter_records(self) -> Iterator[FrameEnrichment]:
        """Iterate compatibility rows in deterministic artifact order."""

        return iter(self._records)


def _is_null(value: object) -> bool:
    """Return whether one legacy scalar encodes a missing value."""

    if value is None or value is pd.NA:
        return True
    return isinstance(value, Real) and math.isnan(float(value))


def _materialize(data: dict[Hashable, Any]) -> FrameEnrichment:
    """Validate one legacy frame-aligned enrichment row."""

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
    return FrameEnrichment.model_validate(values)


__all__ = [
    "ASRStore",
    "CaptionStore",
    "FrameContextStore",
    "ObjectStore",
    "OCRStore",
]
