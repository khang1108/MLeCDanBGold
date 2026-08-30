"""Validate native custom-frame bundles before exposing canonical FrameRecord rows.

This module owns the Python safety boundary between C++ JSONL/image artifacts
and the shared FrameRecord/Parquet contracts. It validates native identity,
timestamps, image coverage, and the submission coordinate formula; it does not
decode media, regenerate images, or alter BTC frame stores.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from hcmai.common.schemas import FrameRecord
from hcmai.common.utils.io import atomic_write, read_json, write_json, write_parquet
from offline.enrichment.bundle import publish_staged_bundle
from hcmai.corpus.stores.frame import FrameStore


_NATIVE_EXTRACTOR_VERSION = "hcmai-keyframes-extractor/0.1.0"
_EXPECTED_STATUSES = frozenset({"enrichment_pending", "published"})
_IMAGE_VARIANTS = frozenset({"durable", "enrichment"})


@dataclass(frozen=True)
class NativeValidationReport:
    """Inspectable result of one complete native per-video bundle validation.

    Attributes:
        video_id: Canonical source video represented by the bundle.
        frame_count: Number of validated native JSONL rows.
        expected_frame_count: Native manifest target count for the decoded video.
        duplicate_submission_coordinate_groups: Count of non-unique
            ``(video_id, frame_idx)`` groups, which are observable but allowed.
        frame_id_digest: SHA-256 over newline-separated frame IDs in JSONL order.
        config_hash: Extraction configuration provenance retained by the bundle.
    """

    video_id: str
    frame_count: int
    expected_frame_count: int
    duplicate_submission_coordinate_groups: int
    frame_id_digest: str
    config_hash: str


@dataclass(frozen=True)
class _ValidatedBundle:
    """Private validated bundle state reused while converting rows to contracts."""

    report: NativeValidationReport
    rows: tuple[dict[str, object], ...]
    run_root: Path
    bundle_root: Path


@dataclass(frozen=True)
class CustomFrameStoreConfig:
    """Configuration for one atomic custom raw-video FrameStore publication.

    Attributes:
        run_root: Native lifecycle root containing validated published bundles.
        output_root: Directory that receives ``frames.parquet`` and ``manifest.json``.
        frame_store_id: Isolated lineage identifier for this custom corpus.
        selected_video_ids: Explicit source IDs required in the published corpus.
    """

    run_root: Path
    output_root: Path
    frame_store_id: str
    selected_video_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize paths and reject an ambiguous or empty corpus selection.

        Raises:
            ValueError: If lineage or selected IDs are blank, unsafe, duplicated,
                or absent.
        """

        normalized_run_root = Path(self.run_root).expanduser()
        normalized_output_root = Path(self.output_root).expanduser()
        if not str(normalized_run_root).strip() or not str(normalized_output_root).strip():
            raise ValueError("run_root and output_root must not be blank")
        if not isinstance(self.frame_store_id, str) or not self.frame_store_id.strip():
            raise ValueError("frame_store_id must be a non-blank string")
        selected = tuple(self.selected_video_ids)
        if not selected:
            raise ValueError("selected_video_ids must not be empty")
        for video_id in selected:
            _require_safe_video_id(video_id, label="selected_video_ids")
        if len(selected) != len(set(selected)):
            raise ValueError("selected_video_ids must not contain duplicates")
        object.__setattr__(self, "run_root", normalized_run_root)
        object.__setattr__(self, "output_root", normalized_output_root)
        object.__setattr__(self, "selected_video_ids", selected)


def _require_object(value: object, *, label: str) -> dict[str, object]:
    """Return a JSON object or raise a contextual validation error.

    Args:
        value: Parsed JSON value to validate.
        label: Human-readable source location or artifact role.

    Returns:
        Plain JSON object dictionary.

    Raises:
        ValueError: If ``value`` is not a JSON object.
    """

    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    """Read exactly one JSON object from an existing artifact file.

    Args:
        path: Existing JSON artifact path.
        label: Human-readable artifact role for diagnostics.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the artifact is missing, malformed, or has a non-object root.
    """

    if not path.is_file():
        raise ValueError(f"{label} must be an existing regular file: {path}")
    try:
        return _require_object(json.loads(path.read_text(encoding="utf-8")), label=label)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label} JSON: {path}") from error


def _require_string(
    value: object,
    *,
    field_name: str,
    label: str,
) -> str:
    """Validate a non-blank string field without silently coercing JSON values.

    Args:
        value: Candidate JSON value.
        field_name: Required field name.
        label: Artifact location used in diagnostics.

    Returns:
        Original non-blank string value.

    Raises:
        ValueError: If value is not a string or is blank after trimming.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} {field_name} must be a non-blank string")
    return value


def _require_integer(
    value: object,
    *,
    field_name: str,
    label: str,
    minimum: int | None = None,
) -> int:
    """Validate an exact JSON integer without accepting bools or floats.

    Args:
        value: Candidate JSON value.
        field_name: Required field name.
        label: Artifact location used in diagnostics.
        minimum: Optional inclusive lower bound.

    Returns:
        Validated integer value.

    Raises:
        ValueError: If value is not an integer or violates ``minimum``.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} {field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} {field_name} must be at least {minimum}")
    return value


def _require_finite_number(
    value: object,
    *,
    field_name: str,
    label: str,
    greater_than: float | None = None,
) -> float:
    """Validate a finite JSON numeric field with an optional strict lower bound.

    Args:
        value: Candidate JSON number.
        field_name: Required field name.
        label: Artifact location used in diagnostics.
        greater_than: Optional strict lower bound.

    Returns:
        Finite floating-point value.

    Raises:
        ValueError: If value is boolean, non-numeric, non-finite, or too small.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} {field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} {field_name} must be finite")
    if greater_than is not None and number <= greater_than:
        raise ValueError(f"{label} {field_name} must be greater than {greater_than}")
    return number


def _require_safe_video_id(value: object, *, label: str) -> str:
    """Validate a native video ID before using it to derive expected paths.

    Args:
        value: Candidate manifest or row video ID.
        label: Artifact location used in diagnostics.

    Returns:
        Safe canonical video ID.

    Raises:
        ValueError: If the identifier is blank or contains unsafe characters.
    """

    video_id = _require_string(value, field_name="video_id", label=label)
    if not all(character.isalnum() or character in "_.-" for character in video_id):
        raise ValueError(f"{label} video_id contains unsafe characters")
    return video_id


def _resolve_bundle_root(
    bundle_root: str | Path,
    *,
    run_root: str | Path,
    expected_status: str,
    video_id: str,
) -> tuple[Path, Path]:
    """Resolve and confine a bundle to its exact staging or published location.

    Args:
        bundle_root: Caller-supplied native bundle directory.
        run_root: Root that owns all native lifecycle artifacts.
        expected_status: Required bundle state token.
        video_id: Validated source-video identifier from its manifest.

    Returns:
        Tuple of resolved ``(run_root, bundle_root)`` paths.

    Raises:
        ValueError: If paths are missing, status is unsupported, or the bundle is
            not exactly below the appropriate lifecycle directory.
    """

    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"run_root must be an existing directory: {root}")
    if expected_status not in _EXPECTED_STATUSES:
        raise ValueError(f"unsupported expected_status: {expected_status}")
    directory = "staging" if expected_status == "enrichment_pending" else "published"
    expected = (root / directory / video_id).resolve()
    resolved_bundle = Path(bundle_root).expanduser().resolve()
    if resolved_bundle != expected:
        raise ValueError(
            "native bundle must be exactly under the expected lifecycle directory"
        )
    if not resolved_bundle.is_dir():
        raise ValueError(f"native bundle must be an existing directory: {resolved_bundle}")
    return root, resolved_bundle


def _relative_image_path(
    value: object,
    *,
    sample_index: int,
    variant: Literal["durable", "enrichment"],
    label: str,
) -> Path:
    """Validate the exact bundle-relative JPEG location for one sample variant.

    Args:
        value: Native JSON image path value.
        sample_index: Zero-based native sample ordinal used in the filename.
        variant: ``durable`` or temporary ``enrichment`` image selection.
        label: JSONL row location used in diagnostics.

    Returns:
        Validated relative image path.

    Raises:
        ValueError: If a path is blank, absolute, escapes, or differs from the
            deterministic native filename convention.
    """

    field_name = "image_path" if variant == "durable" else "enrichment_image_path"
    raw_path = _require_string(value, field_name=field_name, label=label)
    candidate = Path(raw_path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{label} {field_name} must be relative and confined")
    directory = "images" if variant == "durable" else "enrichment_images"
    expected = Path(directory) / f"{sample_index:09d}.jpg"
    if candidate != expected:
        raise ValueError(f"{label} {field_name} does not match the native sample path")
    return candidate


def _validate_image(
    row: dict[str, object],
    *,
    row_label: str,
    bundle_root: Path,
    run_root: Path,
    sample_index: int,
    variant: Literal["durable", "enrichment"],
) -> Path:
    """Verify a selected image exists, stays confined, and matches native bytes.

    Args:
        row: Native JSONL row containing image metadata.
        row_label: JSONL row location used in diagnostics.
        bundle_root: Resolved native bundle directory.
        run_root: Resolved lifecycle root used for confinement.
        sample_index: Native sample ordinal used in deterministic path validation.
        variant: Image variant to validate.

    Returns:
        Resolved image file path below ``run_root``.

    Raises:
        ValueError: If image path, type, existence, or byte-size metadata is invalid.
    """

    path_key = "image_path" if variant == "durable" else "enrichment_image_path"
    size_key = "image_size_bytes" if variant == "durable" else "enrichment_image_size_bytes"
    relative_path = _relative_image_path(
        row.get(path_key),
        sample_index=sample_index,
        variant=variant,
        label=row_label,
    )
    resolved = (bundle_root / relative_path).resolve()
    try:
        resolved.relative_to(run_root)
    except ValueError as error:
        raise ValueError(f"{row_label} {path_key} escapes run_root") from error
    if not resolved.is_file():
        raise ValueError(f"{row_label} {path_key} must be an existing regular file")
    expected_size = _require_integer(
        row.get(size_key),
        field_name=size_key,
        label=row_label,
        minimum=1,
    )
    actual_size = resolved.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{row_label} {size_key} does not match image bytes")
    return resolved


def _read_native_rows(frames_path: Path) -> tuple[dict[str, object], ...]:
    """Parse non-empty native JSONL rows while preserving source order.

    Args:
        frames_path: Existing native frame JSONL path.

    Returns:
        Immutable sequence of parsed row dictionaries in JSONL order.

    Raises:
        ValueError: If a line is blank, invalid JSON, or not an object.
    """

    if not frames_path.is_file():
        raise ValueError(f"native frame JSONL must be an existing regular file: {frames_path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        frames_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        label = f"{frames_path}:{line_number}"
        if not line.strip():
            raise ValueError(f"native frame JSONL contains a blank row: {label}")
        try:
            rows.append(_require_object(json.loads(line), label=label))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid native frame JSONL row: {label}") from error
    return tuple(rows)


def _frame_id_digest(frame_ids: list[str]) -> str:
    """Return the handoff digest over frame IDs in canonical native row order.

    Args:
        frame_ids: Ordered internal IDs from one validated native JSONL file.

    Returns:
        Lowercase SHA-256 digest of newline-separated UTF-8 identifiers.
    """

    return hashlib.sha256("\n".join(frame_ids).encode("utf-8")).hexdigest()


def _validated_native_bundle(
    bundle_root: str | Path,
    *,
    run_root: str | Path,
    expected_status: Literal["enrichment_pending", "published"],
    image_variant: Literal["durable", "enrichment"],
) -> _ValidatedBundle:
    """Validate a bundle once and retain rows for report and FrameRecord conversion.

    Args:
        bundle_root: Native staging or published per-video bundle directory.
        run_root: Lifecycle root that bounds every artifact path.
        expected_status: Required native manifest status and bundle location.
        image_variant: Image representation whose files are checked for callers.

    Returns:
        Private validated bundle value containing report, native rows, and paths.

    Raises:
        ValueError: If any manifest, row identity, temporal, formula, or image
            invariant fails.
    """

    if expected_status not in _EXPECTED_STATUSES:
        raise ValueError(f"unsupported expected_status: {expected_status}")
    if image_variant not in _IMAGE_VARIANTS:
        raise ValueError(f"unsupported image_variant: {image_variant}")

    supplied_bundle = Path(bundle_root).expanduser()
    preliminary_root = Path(run_root).expanduser().resolve()
    try:
        supplied_bundle.resolve().relative_to(preliminary_root)
    except ValueError as error:
        raise ValueError("native bundle must remain under run_root") from error
    manifest = _read_json_object(
        supplied_bundle / "manifest.json",
        label="native manifest",
    )
    manifest_label = str(supplied_bundle / "manifest.json")
    video_id = _require_safe_video_id(manifest.get("video_id"), label=manifest_label)
    resolved_run_root, resolved_bundle_root = _resolve_bundle_root(
        supplied_bundle,
        run_root=run_root,
        expected_status=expected_status,
        video_id=video_id,
    )
    if _require_string(manifest.get("status"), field_name="status", label=manifest_label) != expected_status:
        raise ValueError("native manifest status does not match expected_status")
    if _require_string(
        manifest.get("extractor_version"),
        field_name="extractor_version",
        label=manifest_label,
    ) != _NATIVE_EXTRACTOR_VERSION:
        raise ValueError("native manifest extractor_version is unsupported")
    config_hash = _require_string(
        manifest.get("config_hash"),
        field_name="config_hash",
        label=manifest_label,
    )
    expected_frame_count = _require_integer(
        manifest.get("expected_frame_count"),
        field_name="expected_frame_count",
        label=manifest_label,
        minimum=0,
    )
    emitted_frame_count = _require_integer(
        manifest.get("emitted_frame_count"),
        field_name="emitted_frame_count",
        label=manifest_label,
        minimum=0,
    )
    if expected_frame_count != emitted_frame_count:
        raise ValueError("native manifest expected/emitted frame counts differ")
    _require_finite_number(
        manifest.get("avg_fps"),
        field_name="avg_fps",
        label=manifest_label,
        greater_than=0,
    )
    _require_integer(
        manifest.get("avg_fps_num"),
        field_name="avg_fps_num",
        label=manifest_label,
        minimum=1,
    )
    _require_integer(
        manifest.get("avg_fps_den"),
        field_name="avg_fps_den",
        label=manifest_label,
        minimum=1,
    )
    frames_jsonl_value = _require_string(
        manifest.get("frames_jsonl"),
        field_name="frames_jsonl",
        label=manifest_label,
    )
    if Path(frames_jsonl_value).is_absolute() or Path(frames_jsonl_value) != Path("frames.jsonl"):
        raise ValueError("native manifest frames_jsonl must be the bundle-relative frames.jsonl")
    rows = _read_native_rows(resolved_bundle_root / "frames.jsonl")
    if len(rows) != expected_frame_count:
        raise ValueError("native frame JSONL count does not match manifest")

    frame_ids: list[str] = []
    frame_coordinates: list[tuple[str, int]] = []
    previous_timestamp_ms = -1
    for expected_sample_index, row in enumerate(rows):
        row_label = f"{resolved_bundle_root / 'frames.jsonl'}:{expected_sample_index + 1}"
        sample_index = _require_integer(
            row.get("sample_index"),
            field_name="sample_index",
            label=row_label,
            minimum=0,
        )
        if sample_index != expected_sample_index:
            raise ValueError("native sample_index must begin at zero and increase by one")
        row_video_id = _require_safe_video_id(row.get("video_id"), label=row_label)
        if row_video_id != video_id:
            raise ValueError("native frame row video_id does not match manifest")
        frame_id = _require_string(row.get("frame_id"), field_name="frame_id", label=row_label)
        expected_frame_id = f"{video_id}_raw1fps_{sample_index:09d}"
        if frame_id != expected_frame_id:
            raise ValueError("native frame_id does not match the custom identity contract")
        target_timestamp_ms = _require_integer(
            row.get("target_timestamp_ms"),
            field_name="target_timestamp_ms",
            label=row_label,
            minimum=0,
        )
        if target_timestamp_ms != sample_index * 1_000:
            raise ValueError("native target_timestamp_ms does not match one-FPS order")
        timestamp_ms = _require_integer(
            row.get("timestamp_ms"),
            field_name="timestamp_ms",
            label=row_label,
            minimum=0,
        )
        if timestamp_ms < previous_timestamp_ms:
            raise ValueError("native actual timestamps must be monotonic")
        previous_timestamp_ms = timestamp_ms
        avg_fps = _require_finite_number(
            row.get("avg_fps"),
            field_name="avg_fps",
            label=row_label,
            greater_than=0,
        )
        _require_integer(
            row.get("avg_fps_num"),
            field_name="avg_fps_num",
            label=row_label,
            minimum=1,
        )
        _require_integer(
            row.get("avg_fps_den"),
            field_name="avg_fps_den",
            label=row_label,
            minimum=1,
        )
        _require_integer(row.get("pts"), field_name="pts", label=row_label)
        _require_integer(
            row.get("time_base_num"),
            field_name="time_base_num",
            label=row_label,
            minimum=1,
        )
        _require_integer(
            row.get("time_base_den"),
            field_name="time_base_den",
            label=row_label,
            minimum=1,
        )
        _require_integer(row.get("width"), field_name="width", label=row_label, minimum=1)
        _require_integer(row.get("height"), field_name="height", label=row_label, minimum=1)
        frame_idx = _require_integer(
            row.get("frame_idx"),
            field_name="frame_idx",
            label=row_label,
            minimum=0,
        )
        expected_frame_idx = math.floor(math.ceil(avg_fps) * timestamp_ms / 1_000)
        if frame_idx != expected_frame_idx:
            raise ValueError("native frame_idx formula does not match actual timestamp")
        _validate_image(
            row,
            row_label=row_label,
            bundle_root=resolved_bundle_root,
            run_root=resolved_run_root,
            sample_index=sample_index,
            variant=image_variant,
        )
        frame_ids.append(frame_id)
        frame_coordinates.append((video_id, frame_idx))

    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("native frame JSONL contains duplicate frame_id values")
    coordinate_counts = Counter(frame_coordinates)
    duplicate_groups = sum(count > 1 for count in coordinate_counts.values())
    report = NativeValidationReport(
        video_id=video_id,
        frame_count=len(rows),
        expected_frame_count=expected_frame_count,
        duplicate_submission_coordinate_groups=duplicate_groups,
        frame_id_digest=_frame_id_digest(frame_ids),
        config_hash=config_hash,
    )
    return _ValidatedBundle(
        report=report,
        rows=rows,
        run_root=resolved_run_root,
        bundle_root=resolved_bundle_root,
    )


def validate_native_video_bundle(
    bundle_root: str | Path,
    *,
    run_root: str | Path,
    expected_status: Literal["enrichment_pending", "published"] = "published",
) -> NativeValidationReport:
    """Validate a native bundle before any canonical artifact publication.

    Args:
        bundle_root: Per-video ``staging`` or ``published`` bundle directory.
        run_root: Root that bounds the native state and all image artifacts.
        expected_status: Required manifest status and matching lifecycle directory.

    Returns:
        Summary of the validated bundle, including identity digest and duplicate
        submission-coordinate diagnostics.

    Raises:
        ValueError: If native manifest, JSONL, formula, or durable images violate
            the custom extraction contract.
    """

    return _validated_native_bundle(
        bundle_root,
        run_root=run_root,
        expected_status=expected_status,
        image_variant="durable",
    ).report


def _record_from_native_row(
    row: dict[str, object],
    *,
    run_root: Path,
    bundle_root: Path,
    image_variant: Literal["durable", "enrichment"],
) -> FrameRecord:
    """Map one previously validated native row to the shared frame contract.

    Args:
        row: Native JSONL row already checked by ``_validated_native_bundle``.
        run_root: Resolved root used to emit portable image paths.
        bundle_root: Resolved bundle root that owns row-relative JPEGs.
        image_variant: Durable retrieval or temporary OCR image representation.

    Returns:
        Validated ``FrameRecord`` retaining native identity and temporal metadata.
    """

    path_key = "image_path" if image_variant == "durable" else "enrichment_image_path"
    image_path = (bundle_root / str(row[path_key])).resolve()
    return FrameRecord(
        frame_id=str(row["frame_id"]),
        video_id=str(row["video_id"]),
        frame_idx=int(row["frame_idx"]),
        keyframe_order=None,
        timestamp_ms=int(row["timestamp_ms"]),
        fps=float(row["avg_fps"]),
        image_path=image_path.relative_to(run_root).as_posix(),
        thumbnail_path=None,
        width=int(row["width"]),
        height=int(row["height"]),
        shot_id=None,
        event_id=None,
        is_anchor=True,
        pts=int(row["pts"]),
        time_base=f"{int(row['time_base_num'])}/{int(row['time_base_den'])}",
        motion_score=0.0,
        shot_score=0.0,
        event_score=0.0,
        selection_reasons=("custom_raw_1fps",),
    )


def iter_native_frame_records(
    bundle_root: str | Path,
    *,
    run_root: str | Path,
    image_variant: Literal["durable", "enrichment"] = "durable",
) -> Iterator[FrameRecord]:
    """Yield fully validated custom frames in native sample-index order.

    Args:
        bundle_root: Per-video staging or published native bundle directory.
        run_root: Root that bounds every referenced source image.
        image_variant: Durable image for final corpus use or temporary enrichment
            image for OCR-only per-video work.

    Yields:
        ``FrameRecord`` values with native internal IDs, actual timestamps, and
        competition-facing ``frame_idx`` values preserved exactly.

    Raises:
        ValueError: If the inferred lifecycle location or requested image variant
            fails native bundle validation.
    """

    path = Path(bundle_root).expanduser().resolve()
    root = Path(run_root).expanduser().resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("native bundle must remain under run_root") from error
    if len(relative.parts) != 2 or relative.parts[0] not in {"staging", "published"}:
        raise ValueError("native bundle must be under staging/{video_id} or published/{video_id}")
    expected_status: Literal["enrichment_pending", "published"] = (
        "enrichment_pending" if relative.parts[0] == "staging" else "published"
    )
    validated = _validated_native_bundle(
        path,
        run_root=root,
        expected_status=expected_status,
        image_variant=image_variant,
    )
    for row in validated.rows:
        yield _record_from_native_row(
            row,
            run_root=validated.run_root,
            bundle_root=validated.bundle_root,
            image_variant=image_variant,
        )


def _submission_coordinate_diagnostics(
    records: list[FrameRecord],
) -> dict[str, int]:
    """Summarize allowed non-unique competition coordinates for observability.

    Args:
        records: Fully validated canonical rows in final corpus order.

    Returns:
        Counts for duplicate groups, duplicate rows, and maximum multiplicity.
    """

    counts = Counter((record.video_id, record.frame_idx) for record in records)
    duplicate_counts = [count for count in counts.values() if count > 1]
    return {
        "duplicate_submission_coordinate_groups": len(duplicate_counts),
        "duplicate_submission_coordinate_rows": sum(duplicate_counts),
        "maximum_submission_coordinate_multiplicity": max(counts.values(), default=0),
    }


def _validate_staged_frame_store(
    frames_path: Path,
    *,
    expected_frame_ids: list[str],
) -> None:
    """Re-read staged Parquet through shared FrameRecord and FrameStore contracts.

    Args:
        frames_path: Staged Parquet file before global publication.
        expected_frame_ids: Exact final deterministic ordering expected from rows.

    Returns:
        None; returns only after all staged rows and store indexes validate.

    Raises:
        ValueError: If re-read rows or order differ from the validated input.
    """

    table = pd.read_parquet(frames_path)
    rows = table.astype(object).where(table.notna(), None).to_dict(orient="records")
    records = [FrameRecord.model_validate(row) for row in rows]
    if [record.frame_id for record in records] != expected_frame_ids:
        raise ValueError("staged custom FrameStore changed canonical frame order")
    store = FrameStore(frames_path)
    if [record.frame_id for record in store.iter_frames()] != expected_frame_ids:
        raise ValueError("staged custom FrameStore failed runtime order validation")


def materialize_custom_frame_store(config: CustomFrameStoreConfig) -> Path:
    """Atomically publish validated custom native frames as a canonical Parquet store.

    Args:
        config: Explicit native run, output root, custom lineage, and selected IDs.

    Returns:
        Final ``frames.parquet`` path under ``config.output_root``.

    Raises:
        ValueError: If any selected published bundle is missing, invalid, mixed in
            config provenance, or cannot form one canonical FrameStore.
        OSError: If staged artifact publication cannot complete atomically.
    """

    run_root = config.run_root.resolve()
    if not run_root.is_dir():
        raise ValueError(f"run_root must be an existing directory: {run_root}")

    selected_video_ids = tuple(sorted(config.selected_video_ids))
    records: list[FrameRecord] = []
    reports: list[NativeValidationReport] = []
    per_video_frame_counts: dict[str, int] = {}
    for video_id in selected_video_ids:
        bundle_root = run_root / "published" / video_id
        if not bundle_root.is_dir():
            raise ValueError(f"missing validated published bundle: {video_id}")
        validated = _validated_native_bundle(
            bundle_root,
            run_root=run_root,
            expected_status="published",
            image_variant="durable",
        )
        video_records = [
            _record_from_native_row(
                row,
                run_root=validated.run_root,
                bundle_root=validated.bundle_root,
                image_variant="durable",
            )
            for row in validated.rows
        ]
        if len(video_records) != validated.report.frame_count:
            raise ValueError("validated native bundle changed during materialization")
        reports.append(validated.report)
        records.extend(video_records)
        per_video_frame_counts[video_id] = len(video_records)

    frame_ids = [record.frame_id for record in records]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("custom corpus contains duplicate frame_id values")
    config_hashes = {report.config_hash for report in reports}
    if len(config_hashes) != 1:
        raise ValueError("custom corpus cannot mix native config_hash values")

    diagnostics = _submission_coordinate_diagnostics(records)
    manifest: dict[str, object] = {
        "pipeline_version": "custom-raw-video-1fps-v1",
        "source": "custom_raw_video_1fps",
        "frame_store_id": config.frame_store_id,
        "run_root": str(run_root),
        "video_count": len(selected_video_ids),
        "frame_count": len(records),
        "sample_period_ms": 1_000,
        "submission_coordinate_formula": "floor(ceil(avg_fps) * timestamp_ms / 1000)",
        "resume_enabled": True,
        "config_hash": next(iter(config_hashes)),
        "frame_id_digest": _frame_id_digest(frame_ids),
        "per_video_frame_counts": per_video_frame_counts,
        **diagnostics,
    }
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "frames.parquet"
    manifest_path = output_root / "manifest.json"
    staged_frames = output_root / ".frames.parquet.staged"
    staged_manifest = output_root / ".manifest.json.staged"
    table_rows = [record.model_dump(mode="python") for record in records]
    try:
        atomic_write(
            staged_frames,
            lambda path: write_parquet(
                pd.DataFrame(
                    table_rows,
                    columns=list(FrameRecord.model_fields),
                ),
                path,
                index=False,
            ),
        )
        atomic_write(staged_manifest, lambda path: write_json(manifest, path))
        _validate_staged_frame_store(
            staged_frames,
            expected_frame_ids=frame_ids,
        )
        if read_json(staged_manifest) != manifest:
            raise ValueError("staged custom FrameStore manifest failed round-trip validation")
        publish_staged_bundle(
            (staged_frames, staged_manifest),
            (output_path, manifest_path),
        )
    finally:
        staged_frames.unlink(missing_ok=True)
        staged_manifest.unlink(missing_ok=True)
    return output_path


__all__ = [
    "CustomFrameStoreConfig",
    "NativeValidationReport",
    "iter_native_frame_records",
    "materialize_custom_frame_store",
    "validate_native_video_bundle",
]
