"""Build and safely publish deterministic FrameContext V1 artifacts.

The builder joins already-materialized specialist evidence to canonical frame
identity. It performs no model inference and leaves every source artifact
unchanged. The manifest is the final commit marker for the two-file bundle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd

from hcmai.common.schemas import (
    CaptionEvidence,
    FrameContext,
    ObjectDetection,
    ObjectEvidence,
    OCREvidence,
    ProcessingStatus,
)
from hcmai.common.utils.io import atomic_write, read_json, write_json, write_parquet
from hcmai.data.stores.frame import FrameStore

from .config import FrameContextConfig
from .serializer import serialize_frame_context


_EvidenceT = TypeVar("_EvidenceT", CaptionEvidence, OCREvidence, ObjectEvidence)


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

    values = [requested, canonical]
    values.extend(manifest.get("frame_store_id") for manifest in manifests)
    present = [value for value in values if value is not None]
    if any(not isinstance(value, str) or not value.strip() for value in present):
        raise ValueError("frame_store_id lineage must be a non-empty string")
    distinct = set(cast(list[str], present))
    if len(distinct) > 1:
        raise ValueError(f"frame_store_id lineage mismatch: {sorted(distinct)}")
    return next(iter(distinct), None)


def _read_table(path: Path, name: str) -> pd.DataFrame:
    """Read a required Parquet artifact without masking schema/read failures."""

    if not path.is_file():
        raise FileNotFoundError(f"required {name} artifact not found: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as error:
        raise ValueError(f"malformed {name} artifact: {path}") from error


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

    if "detections" not in values:
        count = values.get("detection_count", 0)
        if isinstance(count, bool) or not isinstance(count, int):
            # Parquet commonly materializes integer columns as numpy integers.
            try:
                count = int(cast(Any, count))
            except Exception as error:
                raise ValueError("object detection_count must be an integer") from error
        counts = cast(dict[str, int], values.get("counts", {}))
        labels = [label for label, total in counts.items() for _ in range(int(total))]
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
    table: pd.DataFrame,
    name: str,
    contract: type[_EvidenceT],
    version: str,
    canonical: dict[str, tuple[str, int]],
    lineage: str | None,
) -> dict[str, _EvidenceT]:
    """Validate specialist rows, uniqueness, canonical identity, and lineage."""

    if "frame_id" not in table.columns:
        raise ValueError(f"{name} artifact is missing frame_id")
    frame_ids = table["frame_id"].map(lambda value: str(value)).tolist()
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError(f"{name} artifact contains duplicate frame_id values")

    rows: dict[str, _EvidenceT] = {}
    for raw in cast(list[dict[str, Any]], table.to_dict(orient="records")):
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
        video_id, frame_idx = canonical[row.frame_id]
        if row.video_id != video_id or row.frame_idx != frame_idx:
            raise ValueError(
                f"{name} row does not match canonical identity: {row.frame_id}"
            )
        if row.artifact_version != version:
            raise ValueError(f"{name} row artifact version does not match manifest")
        if row.frame_store_id is not None and row.frame_store_id != lineage:
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


def _valid_existing_bundle(
    context_path: Path,
    manifest_path: Path,
    identity: dict[str, Any],
    canonical: dict[str, tuple[str, int]],
) -> bool:
    """Accept resume only for exact identity, coverage, order, and row lineage."""

    if not context_path.is_file() or not manifest_path.is_file():
        return False
    try:
        if read_json(manifest_path) != identity:
            return False
        table = pd.read_parquet(context_path)
        rows = [
            FrameContext.model_validate(
                {key: _optional(value) for key, value in raw.items()}
            )
            for raw in cast(list[dict[str, Any]], table.to_dict(orient="records"))
        ]
    except Exception:
        return False
    if [row.frame_id for row in rows] != list(canonical):
        return False
    return all(
        (row.video_id, row.frame_idx) == canonical[row.frame_id]
        and row.context_version == identity["context_version"]
        and row.caption_version == identity["caption_version"]
        and row.ocr_version == identity["ocr_version"]
        and row.object_version == identity["object_version"]
        and row.frame_store_id == identity["frame_store_id"]
        for row in rows
    )


def _publish_staged_bundle(
    staged: tuple[Path, Path], published: tuple[Path, Path]
) -> None:
    """Publish context data then manifest, restoring the prior bundle on error."""

    backups = tuple(
        target.with_name(f".{target.name}.backup") for target in published
    )
    for backup in backups:
        if backup.exists():
            raise RuntimeError(f"refusing to overwrite stale backup: {backup}")

    replaced: list[Path] = []
    restore_complete = False
    try:
        for target, backup in zip(published, backups):
            if target.exists():
                target.replace(backup)
        for source, target in zip(staged, published):
            replaced.append(target)
            source.replace(target)
    except Exception:
        for target in replaced:
            target.unlink(missing_ok=True)
        for target, backup in zip(published, backups):
            if backup.exists():
                backup.replace(target)
        restore_complete = True
        raise
    else:
        restore_complete = True
    finally:
        if restore_complete:
            for backup in backups:
                backup.unlink(missing_ok=True)


def _write_bundle(
    output: Path,
    rows: list[FrameContext],
    identity: dict[str, Any],
    canonical: dict[str, tuple[str, int]],
) -> Path:
    """Stage, validate, and atomically publish the complete context bundle."""

    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "frame_context_v1.parquet"
    manifest_path = output / "manifest.json"
    staged = (
        output / ".frame_context_v1.parquet.staged",
        output / ".manifest.json.staged",
    )
    try:
        table = pd.DataFrame(
            [row.model_dump(mode="json") for row in rows],
            columns=list(FrameContext.model_fields),
        )
        atomic_write(
            staged[0], lambda path: write_parquet(table, path, index=False)
        )
        atomic_write(staged[1], lambda path: write_json(identity, path))
        if not _valid_existing_bundle(staged[0], staged[1], identity, canonical):
            raise ValueError("staged FrameContext bundle failed validation")
        _publish_staged_bundle(staged, (context_path, manifest_path))
    finally:
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
    frames = list(FrameStore.load(frames_file).iter_frames())
    if not frames:
        raise ValueError("canonical frame store must contain at least one frame")
    canonical_order = [frame.frame_id for frame in frames]
    canonical = {
        frame.frame_id: (frame.video_id, frame.frame_idx) for frame in frames
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
        _read_table(caption_file, "caption"),
        "caption",
        CaptionEvidence,
        caption_version,
        canonical,
        lineage,
    )
    ocr_rows = _validated_rows(
        _read_table(ocr_file, "OCR"),
        "OCR",
        OCREvidence,
        ocr_version,
        canonical,
        lineage,
    )
    object_rows = _validated_rows(
        _read_table(object_file, "object"),
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
    output = Path(output_dir)
    context_path = output / "frame_context_v1.parquet"
    manifest_path = output / "manifest.json"
    if _valid_existing_bundle(context_path, manifest_path, identity, canonical):
        return context_path

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
                frame_store_id=lineage,
            )
        )
    return _write_bundle(output, contexts, identity, canonical)


__all__ = ["build_frame_context"]
