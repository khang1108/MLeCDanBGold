"""Prepare per-video enrichment inputs and validate specialist handoff lineage.

This module materializes temporary FrameArtifact tables from an already validated
native staging bundle and writes a compact handoff after Caption, OCR, Objects,
and ASR artifacts preserve canonical identity. It does not run any model,
flatten specialist evidence, alter native state JSON, or publish a global frame
store.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Literal

import pandas as pd

from offline.ingestion.models import FrameArtifact
from hcmai.common.utils.io import atomic_write, read_json, write_json, write_parquet
from offline.ingestion.custom_frames import (
    NativeValidationReport,
    iter_native_frame_records,
    validate_native_video_bundle,
)


_REQUIRED_ARTIFACT_KEYS = ("caption", "ocr", "objects", "asr")


@dataclass(frozen=True)
class _ArtifactEntry:
    """Private normalized artifact path/status retained in the compact handoff."""

    path: str
    status: str


def _infer_staging_context(bundle_root: str | Path) -> tuple[Path, Path]:
    """Infer the native run root from an exact ``staging/{video_id}`` bundle path.

    Args:
        bundle_root: Existing per-video staging directory.

    Returns:
        Tuple of resolved ``(run_root, bundle_root)`` paths.

    Raises:
        ValueError: If the path is not an existing direct child of ``staging``.
    """

    bundle = Path(bundle_root).expanduser().resolve()
    if not bundle.is_dir() or bundle.parent.name != "staging":
        raise ValueError("bundle_root must be an existing staging/{video_id} directory")
    run_root = bundle.parent.parent
    if not run_root.is_dir():
        raise ValueError("native run_root inferred from bundle_root is unavailable")
    return run_root, bundle


def _frame_table(records: list[FrameArtifact]) -> pd.DataFrame:
    """Convert already validated frame contracts into stable Parquet columns.

    Args:
        records: Canonical frame records in native sample-index order.

    Returns:
        DataFrame with every declared FrameArtifact column in contract order.
    """

    return pd.DataFrame(
        [record.model_dump(mode="python") for record in records],
        columns=list(FrameArtifact.model_fields),
    )


def materialize_video_enrichment_frames(
    bundle_root: str | Path,
    output_path: str | Path,
    *,
    image_variant: Literal["durable", "enrichment"],
) -> Path:
    """Write a per-video FrameArtifact table for durable or OCR image enrichment.

    Args:
        bundle_root: Existing ``staging/{video_id}`` native bundle.
        output_path: Temporary Parquet destination for a downstream specialist.
        image_variant: ``durable`` for Caption/Objects/visual or ``enrichment``
            for temporary high-resolution OCR.

    Returns:
        Final temporary Parquet path after atomic write and contract revalidation.

    Raises:
        ValueError: If the native bundle, requested image variant, or staged table
            violates canonical FrameArtifact identity.
    """

    run_root, bundle = _infer_staging_context(bundle_root)
    records = list(
        iter_native_frame_records(
            bundle,
            run_root=run_root,
            image_variant=image_variant,
        )
    )
    destination = Path(output_path)

    def write_table(temporary_path: Path) -> None:
        """Write the selected image variant through the atomic temporary path.

        Args:
            temporary_path: Sibling temporary file supplied by ``atomic_write``.

        Returns:
            None; writes a complete FrameArtifact Parquet table.
        """

        write_parquet(_frame_table(records), temporary_path, index=False)

    atomic_write(destination, write_table)
    table = pd.read_parquet(destination)
    loaded_rows = table.astype(object).where(table.notna(), None).to_dict(orient="records")
    loaded_records = [FrameArtifact.model_validate(row) for row in loaded_rows]
    if [record.frame_id for record in loaded_records] != [record.frame_id for record in records]:
        raise ValueError("staged enrichment frame table changed canonical identity order")
    if [record.image_path for record in loaded_records] != [record.image_path for record in records]:
        raise ValueError("staged enrichment frame table changed selected image paths")
    return destination


def _null_scalar(value: object) -> bool:
    """Return whether a pandas value represents a missing scalar field.

    Args:
        value: Scalar cell value from a Parquet table.

    Returns:
        True for ``None``, pandas ``NA``, or floating-point NaN values.
    """

    return value is None or value is pd.NA or (
        isinstance(value, float) and math.isnan(value)
    )


def _require_identity_integer(value: object, *, field_name: str, label: str) -> int:
    """Validate an exact non-negative canonical coordinate from artifact data.

    Args:
        value: Candidate pandas scalar.
        field_name: Canonical identity field name.
        label: Artifact/row diagnostic prefix.

    Returns:
        Exact non-negative Python integer.

    Raises:
        ValueError: If value is boolean, non-integral, or negative.
    """

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} {field_name} must be a non-negative integer")
    return int(value)


def _resolve_frame_artifact(path: Path, *, kind: str) -> Path:
    """Resolve a known frame-native specialist table from a file or artifact directory.

    Args:
        path: Direct Parquet file or artifact directory supplied by the caller.
        kind: Required specialist artifact key.

    Returns:
        Existing Parquet table containing one row per canonical frame.

    Raises:
        ValueError: If no expected artifact table is available.
    """

    if path.is_file():
        return path
    candidate_names = {
        "caption": ("captions.parquet", "frames.parquet"),
        "ocr": ("frames.parquet",),
        "objects": ("frames.parquet",),
    }[kind]
    if path.is_dir():
        for name in candidate_names:
            candidate = path / name
            if candidate.is_file():
                return candidate
    raise ValueError(f"{kind} artifact must resolve to an existing Parquet table: {path}")


def _validate_frame_native_artifact(
    path: Path,
    *,
    kind: str,
    records: list[FrameArtifact],
    frame_store_id: str,
) -> _ArtifactEntry:
    """Require exact ordered frame identity in one specialist artifact table.

    Args:
        path: Direct Parquet file or directory for Caption, OCR, or Objects.
        kind: Specialist key used in errors and filename resolution.
        records: Native durable frames in canonical sample-index order.
        frame_store_id: Required isolated custom-corpus lineage.

    Returns:
        Validated handoff entry with an absolute specialist table path.

    Raises:
        ValueError: If columns, order, coordinates, or lineage do not match.
    """

    table_path = _resolve_frame_artifact(path, kind=kind).resolve()
    table = pd.read_parquet(table_path)
    required_columns = {"frame_id", "video_id", "frame_idx", "timestamp_ms"}
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise ValueError(f"{kind} artifact is missing canonical columns: {', '.join(missing)}")
    artifact_rows = table.astype(object).where(table.notna(), None).to_dict(orient="records")
    expected_ids = [record.frame_id for record in records]
    actual_ids = [row.get("frame_id") for row in artifact_rows]
    if actual_ids != expected_ids:
        raise ValueError(f"{kind} frame identity does not match native order")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError(f"{kind} frame identity contains duplicate frame_id values")
    for row, record in zip(artifact_rows, records, strict=True):
        label = f"{kind} artifact {table_path}"
        if (
            row.get("video_id") != record.video_id
            or _require_identity_integer(
                row.get("frame_idx"),
                field_name="frame_idx",
                label=label,
            )
            != record.frame_idx
            or _require_identity_integer(
                row.get("timestamp_ms"),
                field_name="timestamp_ms",
                label=label,
            )
            != record.timestamp_ms
        ):
            raise ValueError(f"{kind} frame identity does not match native coordinates")
        if "frame_store_id" in row and row["frame_store_id"] != frame_store_id:
            raise ValueError(f"{kind} artifact frame_store_id does not match handoff")
    return _ArtifactEntry(path=str(table_path), status="validated")


def _resolve_asr_artifact(path: Path) -> Path:
    """Resolve one direct per-video ASR Parquet table without global aggregation.

    Args:
        path: Direct Parquet table or a directory containing exactly one table.

    Returns:
        Existing per-video ASR Parquet path.

    Raises:
        ValueError: If a direct unambiguous table cannot be selected.
    """

    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(candidate for candidate in path.rglob("*.parquet") if candidate.is_file())
        if len(candidates) == 1:
            return candidates[0]
    raise ValueError(f"asr artifact must resolve to one per-video Parquet table: {path}")


def _asr_manifest_video_id(table_path: Path) -> str | None:
    """Read an optional transcript sidecar video ID for empty-timeline evidence.

    Args:
        table_path: Direct ASR Parquet table path.

    Returns:
        Sidecar ``video_id`` when a matching JSON manifest exists; otherwise None.
    """

    candidates = (
        table_path.with_suffix(".manifest.json"),
        table_path.parent / "manifest.json",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        value = read_json(candidate)
        if isinstance(value, dict) and isinstance(value.get("video_id"), str):
            return value["video_id"]
    return None


def _validate_asr_artifact(
    path: Path,
    *,
    video_id: str,
    native_duration_ms: int,
) -> _ArtifactEntry:
    """Validate timeline-native ASR evidence without manufacturing frame rows.

    Args:
        path: Direct ASR table or directory containing exactly one such table.
        video_id: Native source video identity expected in every ASR segment.
        native_duration_ms: Decoded video duration used to bound segment intervals.

    Returns:
        Validated handoff entry with an absolute ASR table path.

    Raises:
        ValueError: If ASR identity, ordering, or timeline coverage is invalid.
    """

    table_path = _resolve_asr_artifact(path).resolve()
    table = pd.read_parquet(table_path)
    required_columns = {"video_id", "start_ms", "end_ms"}
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise ValueError(f"asr artifact is missing timeline columns: {', '.join(missing)}")
    rows = table.astype(object).where(table.notna(), None).to_dict(orient="records")
    if not rows:
        if _asr_manifest_video_id(table_path) != video_id:
            raise ValueError("asr empty timeline requires a matching video manifest")
        return _ArtifactEntry(path=str(table_path), status="validated")

    previous_start = -1
    previous_end = -1
    for row in rows:
        label = f"asr artifact {table_path}"
        if row.get("video_id") != video_id:
            raise ValueError("asr video_id does not match native bundle")
        start_ms = _require_identity_integer(
            row.get("start_ms"),
            field_name="start_ms",
            label=label,
        )
        end_ms = _require_identity_integer(
            row.get("end_ms"),
            field_name="end_ms",
            label=label,
        )
        if end_ms <= start_ms:
            raise ValueError("asr timeline segments must have positive duration")
        if native_duration_ms >= 0 and end_ms > native_duration_ms:
            raise ValueError("asr timeline segment exceeds native video duration")
        if start_ms < previous_start or end_ms < previous_end:
            raise ValueError("asr timeline segments must be monotonic")
        previous_start, previous_end = start_ms, end_ms
    return _ArtifactEntry(path=str(table_path), status="validated")


def _normalize_artifact_input(value: object, *, kind: str) -> tuple[str, Path | None]:
    """Normalize an explicit evaluated/not-evaluated artifact declaration.

    Args:
        value: Path-like input, ``None``, or mapping with ``status`` and ``path``.
        kind: Required specialist key used in diagnostics.

    Returns:
        Tuple of ``(status, path_or_none)``.

    Raises:
        ValueError: If the declaration is ambiguous, blank, or unsupported.
    """

    if value is None:
        return "not_evaluated", None
    if isinstance(value, Mapping):
        status = value.get("status")
        raw_path = value.get("path")
        if status == "not_evaluated":
            if raw_path not in (None, ""):
                raise ValueError(f"{kind} not_evaluated artifact must not carry a path")
            return "not_evaluated", None
        if status != "validated" or not isinstance(raw_path, (str, Path)):
            raise ValueError(f"{kind} artifact mapping must declare validated path or not_evaluated")
        value = raw_path
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{kind} artifact path must be explicit and non-blank")
    return "validated", Path(value).expanduser()


def _native_duration_ms(bundle_root: Path) -> int:
    """Read the decoded native duration used to bound timeline ASR evidence.

    Args:
        bundle_root: Existing native staging bundle.

    Returns:
        Non-negative decoded duration or ``-1`` when native decoding reported unknown.

    Raises:
        ValueError: If the native manifest duration is missing or non-integral.
    """

    manifest = read_json(bundle_root / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("native manifest must contain an object")
    duration_ms = manifest.get("duration_ms")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, Integral) or duration_ms < -1:
        raise ValueError("native manifest duration_ms must be an integer at least -1")
    return int(duration_ms)


def write_enrichment_handoff(
    bundle_root: str | Path,
    *,
    artifact_paths: Mapping[str, object],
    output_path: str | Path,
    frame_store_id: str,
) -> Path:
    """Validate specialist artifacts and atomically write the native handoff JSON.

    Args:
        bundle_root: Existing ``staging/{video_id}`` native bundle.
        artifact_paths: Explicit mapping for caption, ocr, objects, and asr. A
            value of ``None`` or ``{"status": "not_evaluated"}`` records an
            intentionally unevaluated modality without implying negative evidence.
        output_path: Handoff destination within the selected staging bundle.
        frame_store_id: Required custom FrameStore lineage shared by frame-native
            specialist artifacts.

    Returns:
        Final handoff JSON path after atomic write and round-trip validation.

    Raises:
        ValueError: If native, artifact, path, or lineage identity is inconsistent.
    """

    if not isinstance(frame_store_id, str) or not frame_store_id.strip():
        raise ValueError("frame_store_id must be a non-blank string")
    if set(artifact_paths) != set(_REQUIRED_ARTIFACT_KEYS):
        missing = sorted(set(_REQUIRED_ARTIFACT_KEYS).difference(artifact_paths))
        unexpected = sorted(set(artifact_paths).difference(_REQUIRED_ARTIFACT_KEYS))
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise ValueError("artifact_paths must contain exactly caption, ocr, objects, asr: " + "; ".join(details))

    run_root, bundle = _infer_staging_context(bundle_root)
    report: NativeValidationReport = validate_native_video_bundle(
        bundle,
        run_root=run_root,
        expected_status="enrichment_pending",
    )
    records = list(iter_native_frame_records(bundle, run_root=run_root))
    destination = Path(output_path).expanduser().resolve()
    try:
        destination.relative_to(bundle)
    except ValueError as error:
        raise ValueError("handoff output_path must remain within the staging bundle") from error

    entries: dict[str, _ArtifactEntry] = {}
    for kind in _REQUIRED_ARTIFACT_KEYS:
        status, path = _normalize_artifact_input(artifact_paths[kind], kind=kind)
        if status == "not_evaluated":
            entries[kind] = _ArtifactEntry(path="", status=status)
        elif kind == "asr":
            assert path is not None
            entries[kind] = _validate_asr_artifact(
                path,
                video_id=report.video_id,
                native_duration_ms=_native_duration_ms(bundle),
            )
        else:
            assert path is not None
            entries[kind] = _validate_frame_native_artifact(
                path,
                kind=kind,
                records=records,
                frame_store_id=frame_store_id,
            )

    native_manifest_path = (bundle / "manifest.json").resolve().relative_to(run_root).as_posix()
    handoff = {
        "video_id": report.video_id,
        "frame_count": report.frame_count,
        "frame_id_digest": report.frame_id_digest,
        "frame_store_id": frame_store_id,
        "config_hash": report.config_hash,
        "native_manifest_path": native_manifest_path,
        "artifacts": {
            kind: {"path": entries[kind].path, "status": entries[kind].status}
            for kind in _REQUIRED_ARTIFACT_KEYS
        },
    }
    atomic_write(destination, lambda path: write_json(handoff, path))
    if read_json(destination) != handoff:
        raise ValueError("staged enrichment handoff failed round-trip validation")
    return destination


__all__ = [
    "materialize_video_enrichment_frames",
    "write_enrichment_handoff",
]
