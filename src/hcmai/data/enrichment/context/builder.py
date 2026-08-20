"""Build and safely publish deterministic FrameContext V1 artifacts.

The builder joins already-materialized specialist evidence to canonical frame
identity. It performs no model inference and leaves every source artifact
unchanged. The manifest is the final commit marker for the two-file bundle.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
import math
from numbers import Integral
from pathlib import Path
from typing import Any, TypeVar, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from hcmai.common.schemas import (
    CaptionEvidence,
    FrameContext,
    FrameRecord,
    ObjectDetection,
    ObjectEvidence,
    OCREvidence,
    ProcessingStatus,
)
from hcmai.common.utils.io import atomic_write, read_json, write_json
from hcmai.data.enrichment.bundle import publish_staged_bundle

from .config import FrameContextConfig
from .serializer import serialize_frame_context


_EvidenceT = TypeVar("_EvidenceT", CaptionEvidence, OCREvidence, ObjectEvidence)
_BATCH_SIZE = 4_096

_CONTEXT_SCHEMA = pa.schema([
    pa.field("frame_id", pa.string(), nullable=False),
    pa.field("video_id", pa.string(), nullable=False),
    pa.field("frame_idx", pa.int64(), nullable=False),
    pa.field("timestamp_ms", pa.int64(), nullable=False),
    pa.field("caption_text", pa.string()),
    pa.field("ocr_text", pa.string()),
    pa.field("object_summary", pa.string()),
    pa.field("context_text", pa.string()),
    pa.field("caption_available", pa.bool_(), nullable=False),
    pa.field("ocr_quality", pa.float64(), nullable=False),
    pa.field("object_count", pa.int64(), nullable=False),
    pa.field("context_version", pa.string(), nullable=False),
    pa.field("caption_version", pa.string(), nullable=False),
    pa.field("ocr_version", pa.string(), nullable=False),
    pa.field("object_version", pa.string(), nullable=False),
    pa.field("frame_store_id", pa.string()),
])


def _optional(value: object) -> object | None:
    """Translate Parquet null scalars into values accepted by contracts."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _required_manifest(path: Path) -> dict[str, Any]:
    """Load one specialist manifest and require a non-empty artifact version."""

    if not path.is_file():
        raise FileNotFoundError(f"required artifact manifest not found: {path}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"artifact manifest must contain an object: {path}")
    version = value.get("artifact_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"artifact manifest has invalid artifact_version: {path}")
    frame_store_id = value.get("frame_store_id")
    if frame_store_id is not None and (
        not isinstance(frame_store_id, str) or not frame_store_id.strip()
    ):
        raise ValueError(f"artifact manifest has invalid frame_store_id: {path}")
    return cast(dict[str, Any], value)


def _canonical_lineage(frames_path: Path) -> str | None:
    """Read canonical lineage when its adjacent ingestion manifest is present."""

    manifest_path = frames_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    value = read_json(manifest_path)
    if not isinstance(value, dict):
        raise ValueError("canonical frame manifest must contain an object")
    lineage = value.get("frame_store_id")
    if lineage is None:
        return None
    if not isinstance(lineage, str) or not lineage.strip():
        raise ValueError("canonical frame manifest has invalid frame_store_id")
    return lineage


def _resolve_lineage(
    requested: str | None,
    canonical: str | None,
    manifests: tuple[dict[str, Any], ...],
) -> str | None:
    """Require every present artifact lineage to identify the same frame store."""

    manifest_values = [manifest.get("frame_store_id") for manifest in manifests]
    values = [requested, canonical, *manifest_values]
    present = [value for value in values if value is not None]
    if any(not isinstance(value, str) or not value.strip() for value in present):
        raise ValueError("frame_store_id lineage must be a non-empty string")
    distinct = set(cast(list[str], present))
    if len(distinct) > 1:
        raise ValueError(f"frame_store_id lineage mismatch: {sorted(distinct)}")
    resolved = next(iter(distinct), None)
    if resolved is not None and any(
        value != resolved for value in manifest_values
    ):
        raise ValueError("specialist manifest lineage mismatch")
    return resolved


def _iter_parquet_rows(
    path: Path,
    name: str,
    *,
    columns: list[str] | None = None,
    batch_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield Parquet rows in bounded batches without materializing a table."""

    if not path.is_file():
        raise FileNotFoundError(f"required {name} artifact not found: {path}")
    effective_batch_size = batch_size or _BATCH_SIZE
    try:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=effective_batch_size,
            columns=columns,
            use_threads=True,
        ):
            yield from cast(Iterable[dict[str, Any]], batch.to_pylist())
    except Exception as error:
        raise ValueError(f"malformed {name} artifact: {path}") from error


def _read_canonical_frames(path: Path) -> list[FrameRecord]:
    """Validate canonical frames while avoiding FrameStore's extra indexes."""

    columns = list(FrameRecord.model_fields)
    frames: list[FrameRecord] = []
    seen: set[str] = set()
    for raw in _iter_parquet_rows(path, "canonical frames", columns=columns):
        values = {
            name: _optional(raw.get(name))
            for name in columns
            if name in raw
        }
        try:
            frame = FrameRecord.model_validate(values)
        except Exception as error:
            raise ValueError("malformed canonical frame row") from error
        if frame.frame_id in seen:
            raise ValueError("canonical frame store contains duplicate frame_id values")
        seen.add(frame.frame_id)
        frames.append(frame)
    return frames


def _strict_integer(value: object, field: str) -> int:
    """Normalize true integer scalars without coercing strings, floats, or bools."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _object_values(raw: dict[str, Any]) -> dict[str, Any]:
    """Adapt the public flattened Object frame artifact to its source contract."""

    values = {key: _optional(value) for key, value in raw.items()}
    if "counts" not in values:
        encoded = values.pop("counts_json", None)
        try:
            counts = json.loads(encoded) if isinstance(encoded, str) else {}
        except json.JSONDecodeError as error:
            raise ValueError("object counts_json must contain valid JSON") from error
        if not isinstance(counts, dict):
            raise ValueError("object counts_json must contain an object")
        values["counts"] = counts

    counts_value = values.get("counts")
    if not isinstance(counts_value, dict):
        raise ValueError("object counts must contain an object")
    counts = {
        label: _strict_integer(total, "object count")
        for label, total in counts_value.items()
    }
    values["counts"] = counts

    count = _strict_integer(
        values.get("detection_count", 0), "object detection_count"
    )
    values["detection_count"] = count
    if "detections" not in values:
        labels = [label for label, total in counts.items() for _ in range(total)]
        if len(labels) > count:
            raise ValueError("object counts exceed detection_count")
        labels.extend("__unlisted__" for _ in range(count - len(labels)))
        values["detections"] = [
            ObjectDetection(
                label=label,
                confidence=0.0,
                x_min=0.0,
                y_min=0.0,
                x_max=0.0,
                y_max=0.0,
            ).model_dump(mode="json")
            for label in labels
        ]
    return values


def _validated_rows(
    raw_rows: Iterable[dict[str, Any]],
    name: str,
    contract: type[_EvidenceT],
    version: str,
    canonical: dict[str, tuple[str, int, int]],
    lineage: str | None,
) -> dict[str, _EvidenceT]:
    """Validate specialist rows, uniqueness, canonical identity, and lineage."""

    rows: dict[str, _EvidenceT] = {}
    for raw in raw_rows:
        raw_frame_id = raw.get("frame_id")
        raw_video_id = raw.get("video_id")
        if "frame_id" not in raw:
            raise ValueError(f"{name} artifact is missing frame_id")
        raw_frame_idx = _strict_integer(raw.get("frame_idx"), "frame_idx")
        raw_timestamp_ms = _strict_integer(
            raw.get("timestamp_ms"), "timestamp_ms"
        )
        if (
            not isinstance(raw_frame_id, str)
            or not raw_frame_id
            or raw_frame_id.strip() != raw_frame_id
            or not isinstance(raw_video_id, str)
            or not raw_video_id
            or raw_video_id.strip() != raw_video_id
        ):
            raise ValueError(f"{name} row has non-canonical string identity")
        if raw_frame_id in rows:
            raise ValueError(f"{name} artifact contains duplicate frame_id values")
        values = (
            _object_values(raw)
            if contract is ObjectEvidence
            else {key: _optional(value) for key, value in raw.items()}
        )
        try:
            row = contract.model_validate(values)
        except Exception as error:
            raise ValueError(f"malformed {name} evidence row") from error
        if row.frame_id not in canonical:
            raise ValueError(
                f"{name} artifact contains foreign frame_id: {row.frame_id}"
            )
        video_id, frame_idx, timestamp_ms = canonical[row.frame_id]
        if (
            row.frame_id != raw_frame_id
            or row.video_id != raw_video_id
            or row.frame_idx != raw_frame_idx
            or row.timestamp_ms != raw_timestamp_ms
            or row.video_id != video_id
            or row.frame_idx != frame_idx
            or row.timestamp_ms != timestamp_ms
        ):
            raise ValueError(
                f"{name} row does not match canonical identity: {row.frame_id}"
            )
        if row.artifact_version != version:
            raise ValueError(f"{name} row artifact version does not match manifest")
        if row.frame_store_id != lineage:
            raise ValueError(f"{name} row lineage mismatch: {row.frame_id}")
        rows[row.frame_id] = row
    return rows


def _usable_caption(row: CaptionEvidence | None) -> str | None:
    """Return completed error-free caption text, if present."""

    if row is None or row.status != ProcessingStatus.COMPLETED:
        return None
    if row.error_code is not None or row.error_message is not None:
        return None
    return row.text if row.text is not None and row.text.strip() else None


def _usable_ocr(row: OCREvidence | None, minimum: float) -> str | None:
    """Return completed normalized OCR text meeting the V1 quality threshold."""

    if row is None or row.status != ProcessingStatus.COMPLETED:
        return None
    if row.quality_score < minimum:
        return None
    text = row.normalized_text
    return text if text is not None and text.strip() else None


def _usable_objects(row: ObjectEvidence | None) -> str | None:
    """Return completed non-empty object summary text, if present."""

    if row is None or row.status != ProcessingStatus.COMPLETED:
        return None
    return row.summary if row.summary is not None and row.summary.strip() else None


def _serializer_identity(config: FrameContextConfig) -> dict[str, int | float]:
    """Return the exact serializer policy recorded in dependency identity."""

    return {
        "caption_token_budget": config.caption_token_budget,
        "ocr_token_budget": config.ocr_token_budget,
        "object_token_budget": config.object_token_budget,
        "min_ocr_quality": config.min_ocr_quality,
    }


def _build_context_rows(
    frames: list[FrameRecord],
    caption_rows: dict[str, CaptionEvidence],
    ocr_rows: dict[str, OCREvidence],
    object_rows: dict[str, ObjectEvidence],
    config: FrameContextConfig,
    *,
    caption_version: str,
    ocr_version: str,
    object_version: str,
    frame_store_id: str | None,
) -> list[FrameContext]:
    """Derive the exact expected context rows from validated source evidence.

    ``ocr_quality`` and ``object_count`` retain raw specialist diagnostics even
    when quality/status rules omit OCR text or the Object summary.
    """

    contexts: list[FrameContext] = []
    for frame in frames:
        caption = _usable_caption(caption_rows.get(frame.frame_id))
        ocr = _usable_ocr(ocr_rows.get(frame.frame_id), config.min_ocr_quality)
        objects = _usable_objects(object_rows.get(frame.frame_id))
        contexts.append(
            FrameContext(
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                frame_idx=frame.frame_idx,
                timestamp_ms=frame.timestamp_ms,
                caption_text=caption,
                ocr_text=ocr,
                object_summary=objects,
                context_text=serialize_frame_context(
                    caption=caption, ocr=ocr, objects=objects, config=config
                ),
                caption_available=caption is not None,
                ocr_quality=(
                    ocr_rows[frame.frame_id].quality_score
                    if frame.frame_id in ocr_rows
                    else 0.0
                ),
                object_count=(
                    object_rows[frame.frame_id].detection_count
                    if frame.frame_id in object_rows
                    else 0
                ),
                context_version=config.context_version,
                caption_version=caption_version,
                ocr_version=ocr_version,
                object_version=object_version,
                frame_store_id=frame_store_id,
            )
        )
    return contexts


def _context_batches(
    frames: list[FrameRecord],
    caption_rows: dict[str, CaptionEvidence],
    ocr_rows: dict[str, OCREvidence],
    object_rows: dict[str, ObjectEvidence],
    config: FrameContextConfig,
    *,
    caption_version: str,
    ocr_version: str,
    object_version: str,
    frame_store_id: str | None,
    batch_size: int | None = None,
) -> Iterator[list[FrameContext]]:
    """Yield derived context rows in bounded batches for writing/validation."""

    effective_batch_size = batch_size or _BATCH_SIZE
    for start in range(0, len(frames), effective_batch_size):
        yield _build_context_rows(
            frames[start : start + effective_batch_size],
            caption_rows,
            ocr_rows,
            object_rows,
            config,
            caption_version=caption_version,
            ocr_version=ocr_version,
            object_version=object_version,
            frame_store_id=frame_store_id,
        )


def _valid_existing_bundle(
    context_path: Path,
    manifest_path: Path,
    identity: dict[str, Any],
    expected_batches: Iterable[list[FrameContext]],
) -> bool:
    """Accept resume only when identity and every serialized row field match."""

    if not context_path.is_file() or not manifest_path.is_file():
        return False
    try:
        if read_json(manifest_path) != identity:
            return False
        parquet = pq.ParquetFile(context_path)
        if parquet.schema.names != list(FrameContext.model_fields):
            return False
    except Exception:
        return False

    try:
        actual_rows = _iter_parquet_rows(context_path, "context")
        expected_rows = (
            row
            for batch in expected_batches
            for row in batch
        )
        sentinel = object()
        while True:
            actual = next(actual_rows, sentinel)
            expected = next(expected_rows, sentinel)
            if actual is sentinel or expected is sentinel:
                return actual is sentinel and expected is sentinel
            raw = {
                key: _optional(value)
                for key, value in cast(dict[str, Any], actual).items()
            }
            expected_row = cast(FrameContext, expected).model_dump(mode="json")
            if list(raw) != list(expected_row):
                return False
            FrameContext.model_validate(raw)
            # Contract validation above protects field semantics. This raw
            # comparison independently prevents Pydantic stripping/coercion
            # from hiding corruption.
            for key, expected_value in expected_row.items():
                actual_value = raw[key]
                if (
                    type(actual_value) is not type(expected_value)
                    or actual_value != expected_value
                ):
                    return False
    except Exception:
        return False


def _publish_staged_bundle(
    staged: tuple[Path, Path], published: tuple[Path, Path]
) -> None:
    """Publish context data then manifest, restoring the prior bundle on error."""

    publish_staged_bundle(staged, published)


def _write_bundle(
    output: Path,
    batches_factory: Callable[[], Iterable[list[FrameContext]]],
    identity: dict[str, Any],
) -> Path:
    """Stage, validate, and atomically publish a bounded context bundle."""

    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "frame_context_v1.parquet"
    manifest_path = output / "manifest.json"
    staged = (
        output / ".frame_context_v1.parquet.staged",
        output / ".manifest.json.staged",
    )
    writer: pq.ParquetWriter | None = None
    try:
        for batch in batches_factory():
            if not batch:
                continue
            table = pa.Table.from_pylist(
                [row.model_dump(mode="json") for row in batch],
                schema=_CONTEXT_SCHEMA,
            )
            if writer is None:
                writer = pq.ParquetWriter(str(staged[0]), _CONTEXT_SCHEMA)
            writer.write_table(table)
        if writer is None:
            raise ValueError("cannot publish an empty FrameContext bundle")
        writer.close()
        writer = None
        atomic_write(staged[1], lambda path: write_json(identity, path))
        if not _valid_existing_bundle(
            staged[0], staged[1], identity, batches_factory()
        ):
            raise ValueError("staged FrameContext bundle failed validation")
        _publish_staged_bundle(staged, (context_path, manifest_path))
    finally:
        if writer is not None:
            writer.close()
        for path in staged:
            path.unlink(missing_ok=True)
    return context_path


def build_frame_context(
    frames_path: str | Path,
    caption_path: str | Path,
    ocr_frames_path: str | Path,
    object_frames_path: str | Path,
    output_dir: str | Path,
    config: FrameContextConfig,
    *,
    frame_store_id: str | None = None,
) -> Path:
    """Join specialist artifacts and publish one context row per canonical frame."""

    paths = tuple(
        Path(path)
        for path in (frames_path, caption_path, ocr_frames_path, object_frames_path)
    )
    frames_file, caption_file, ocr_file, object_file = paths

    # Validate every prerequisite before creating or replacing context output.
    # The canonical reader uses bounded Arrow batches and keeps only the small
    # identity list needed for specialist joins; it does not build FrameStore's
    # runtime indexes for this offline materialization step.
    frames = _read_canonical_frames(frames_file)
    if not frames:
        raise ValueError("canonical frame store must contain at least one frame")
    canonical_order = [frame.frame_id for frame in frames]
    canonical = {
        frame.frame_id: (frame.video_id, frame.frame_idx, frame.timestamp_ms)
        for frame in frames
    }
    if len(canonical) != len(canonical_order):
        raise ValueError("canonical frame store contains duplicate frame_id values")

    caption_manifest = _required_manifest(caption_file.parent / "manifest.json")
    ocr_manifest = _required_manifest(ocr_file.parent / "manifest.json")
    object_manifest = _required_manifest(object_file.parent / "manifest.json")
    lineage = _resolve_lineage(
        frame_store_id,
        _canonical_lineage(frames_file),
        (caption_manifest, ocr_manifest, object_manifest),
    )
    caption_version = cast(str, caption_manifest["artifact_version"])
    ocr_version = cast(str, ocr_manifest["artifact_version"])
    object_version = cast(str, object_manifest["artifact_version"])

    caption_rows = _validated_rows(
        _iter_parquet_rows(caption_file, "caption"),
        "caption",
        CaptionEvidence,
        caption_version,
        canonical,
        lineage,
    )
    ocr_rows = _validated_rows(
        _iter_parquet_rows(ocr_file, "OCR"),
        "OCR",
        OCREvidence,
        ocr_version,
        canonical,
        lineage,
    )
    object_rows = _validated_rows(
        _iter_parquet_rows(object_file, "object"),
        "object",
        ObjectEvidence,
        object_version,
        canonical,
        lineage,
    )

    identity: dict[str, Any] = {
        "context_version": config.context_version,
        "caption_version": caption_version,
        "ocr_version": ocr_version,
        "object_version": object_version,
        "frame_store_id": lineage,
        "serializer_config": _serializer_identity(config),
    }
    batches_factory = lambda: _context_batches(
        frames,
        caption_rows,
        ocr_rows,
        object_rows,
        config,
        caption_version=caption_version,
        ocr_version=ocr_version,
        object_version=object_version,
        frame_store_id=lineage,
    )
    output = Path(output_dir)
    context_path = output / "frame_context_v1.parquet"
    manifest_path = output / "manifest.json"
    if _valid_existing_bundle(
        context_path, manifest_path, identity, batches_factory()
    ):
        return context_path
    return _write_bundle(output, batches_factory, identity)


__all__ = ["build_frame_context"]
