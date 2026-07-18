"""Validate canonical frame metadata and freeze data artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from os import PathLike
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pydantic import ValidationError

from hcmai.common.schemas.frame import FrameRecord
from hcmai.common.utils.image import load_image
from hcmai.common.utils.io import write_json
from hcmai.data.extract import (
    COLLISION_COLUMNS,
    COLLISION_POLICY,
    FRAME_COLUMNS,
    _images_by_video,
    _keyframe_images,
    _load_mappings,
    _numeric_images,
    _sha256,
    collision_report_rows,
)


PathValue = str | PathLike[str]
MAX_ERROR_EXAMPLES = 200


def _add_error(
    errors: list[dict[str, Any]],
    counts: Counter[str],
    code: str,
    message: str,
    **context: Any,
) -> None:
    """Count an error and retain a bounded example."""

    counts[code] += 1
    if len(errors) < MAX_ERROR_EXAMPLES:
        errors.append({"code": code, "message": message, **context})


def _native(value: Any) -> Any:
    """Convert pandas scalars and missing values to Python values."""

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _read_records(
    path: Path,
    errors: list[dict[str, Any]],
    counts: Counter[str],
) -> tuple[pd.DataFrame, list[FrameRecord]]:
    """Read metadata and validate every row against FrameRecord."""

    try:
        frames = pd.read_parquet(path).reset_index(drop=True)
    except Exception as error:
        _add_error(
            errors,
            counts,
            "metadata_unreadable",
            "Canonical metadata cannot be read.",
            path=str(path),
            detail=str(error),
        )
        return pd.DataFrame(), []

    columns = set(frames.columns)
    missing = sorted(set(FRAME_COLUMNS) - columns)
    unexpected = sorted(columns - set(FRAME_COLUMNS))
    if missing:
        _add_error(
            errors,
            counts,
            "missing_columns",
            "Canonical metadata is missing columns.",
            columns=missing,
        )
    if unexpected:
        _add_error(
            errors,
            counts,
            "unexpected_columns",
            "Canonical metadata has unknown columns.",
            columns=unexpected,
        )

    records = []
    for index, row in frames.iterrows():
        payload = {
            column: _native(row[column])
            for column in FRAME_COLUMNS
            if column in frames
        }
        try:
            records.append(FrameRecord.model_validate(payload, strict=True))
        except (ValidationError, TypeError, ValueError) as error:
            _add_error(
                errors,
                counts,
                "record_schema",
                "Metadata row does not satisfy FrameRecord.",
                row=int(index),
                detail=str(error),
            )
    return frames, records


def _validate_identifiers(
    frames: pd.DataFrame,
    records: list[FrameRecord],
    errors: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    """Validate stable IDs, uniqueness, and per-video order."""

    if "frame_id" in frames and frames["frame_id"].duplicated().any():
        _add_error(
            errors,
            counts,
            "duplicate_frame_id",
            "frame_id values must be globally unique.",
        )
    if {"video_id", "frame_idx"}.issubset(frames) and frames.duplicated(
        ["video_id", "frame_idx"]
    ).any():
        _add_error(
            errors,
            counts,
            "duplicate_video_frame",
            "video_id and frame_idx pairs must be unique.",
        )

    grouped: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in records:
        expected = f"{record.video_id}_{record.frame_idx:08d}"
        if record.frame_id != expected:
            _add_error(
                errors,
                counts,
                "frame_id_format",
                "frame_id does not match video_id and frame_idx.",
                frame_id=record.frame_id,
                expected=expected,
            )
        grouped[record.video_id].append(record)
    for video_id, items in grouped.items():
        order = [(item.timestamp_ms, item.frame_idx) for item in items]
        if order != sorted(order):
            _add_error(
                errors,
                counts,
                "metadata_order",
                "Frames must be monotonic within each video.",
                video_id=video_id,
            )


def _resolve_path(value: str, roots: Iterable[Path]) -> Path:
    """Resolve an absolute or root-relative artifact path."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [root / path for root in roots]
    return next(
        (candidate for candidate in candidates if candidate.exists()),
        candidates[0],
    )


def _validate_files(
    records: list[FrameRecord],
    dataset_root: Path,
    output_root: Path,
    errors: list[dict[str, Any]],
    counts: Counter[str],
) -> dict[str, Path]:
    """Validate source images and thumbnails referenced by metadata."""

    resolved = {}
    for record in records:
        source = _resolve_path(
            record.image_path,
            (dataset_root, output_root),
        )
        resolved[record.frame_id] = source
        if not source.is_file():
            _add_error(
                errors,
                counts,
                "image_missing",
                "Frame image does not exist.",
                frame_id=record.frame_id,
                path=str(source),
            )
        else:
            try:
                if load_image(source).size != (record.width, record.height):
                    raise ValueError("source dimensions do not match metadata")
            except (OSError, ValueError) as error:
                _add_error(
                    errors,
                    counts,
                    "image_invalid",
                    "Frame image is unreadable or has wrong dimensions.",
                    frame_id=record.frame_id,
                    detail=str(error),
                )

        if record.thumbnail_path is None:
            _add_error(
                errors,
                counts,
                "thumbnail_invalid",
                "Frame thumbnail path is missing.",
                frame_id=record.frame_id,
            )
            continue
        thumbnail = _resolve_path(
            record.thumbnail_path,
            (output_root, dataset_root),
        )
        try:
            width, height = load_image(thumbnail).size
            if width > record.width or height > record.height:
                raise ValueError("thumbnail exceeds source dimensions")
        except (OSError, ValueError) as error:
            _add_error(
                errors,
                counts,
                "thumbnail_invalid",
                "Thumbnail is unreadable or has wrong dimensions.",
                frame_id=record.frame_id,
                detail=str(error),
            )
    return resolved


def _validate_collision_report(
    path: Path,
    expected_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    """Validate collision decisions against the source mapping."""

    try:
        actual = pd.read_csv(path, keep_default_na=False)
        expected = pd.DataFrame(expected_rows, columns=COLLISION_COLUMNS)
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            atol=1e-9,
            rtol=1e-9,
        )
    except (OSError, pd.errors.ParserError, AssertionError) as error:
        _add_error(
            errors,
            counts,
            "collision_report_invalid",
            "Collision report does not match source mappings.",
            path=str(path),
            detail=str(error)[:500],
        )


def _validate_mappings(
    dataset_root: Path,
    output_root: Path,
    records: list[FrameRecord],
    resolved_images: dict[str, Path],
    errors: list[dict[str, Any]],
    counts: Counter[str],
) -> dict[str, Any]:
    """Validate source coverage and metadata-to-mapping joins."""

    paths, raw, canonical, collisions, mapping_errors = _load_mappings(
        dataset_root
    )
    if not paths:
        _add_error(
            errors,
            counts,
            "mapping_missing",
            "No official mapping CSV files were found.",
        )
    for error in mapping_errors:
        _add_error(
            errors,
            counts,
            "mapping_invalid",
            "Mapping CSV is invalid.",
            **error,
        )

    source_images = _keyframe_images(dataset_root)
    grouped_images = _images_by_video(source_images)
    indexes = {}
    image_count = 0
    for video_id in sorted(set(raw) | set(grouped_images)):
        index, image_errors = _numeric_images(
            grouped_images.get(video_id, [])
        )
        indexes[video_id] = index
        image_count += len(index)
        for error in image_errors:
            _add_error(
                errors,
                counts,
                "image_mapping_invalid",
                "Keyframe filename is invalid or duplicated.",
                video_id=video_id,
                **error,
            )
        mapped = {int(number) for number in raw.get(video_id, {}).get("n", [])}
        if mapped != set(index):
            _add_error(
                errors,
                counts,
                "mapping_coverage",
                "Mapping rows and images must have one-to-one coverage.",
                video_id=video_id,
                missing_images=sorted(mapped - set(index))[:10],
                missing_mappings=sorted(set(index) - mapped)[:10],
            )

    collision_path = output_root / "reports" / "mapping_collisions.csv"
    _validate_collision_report(
        collision_path,
        collision_report_rows(collisions, grouped_images),
        errors,
        counts,
    )
    lookups = {
        video_id: {
            int(row.frame_idx): (
                int(row.n),
                round(float(row.pts_time) * 1000),
            )
            for row in table.itertuples(index=False)
        }
        for video_id, table in canonical.items()
    }
    for record in records:
        expected = lookups.get(record.video_id, {}).get(record.frame_idx)
        if expected is None:
            _add_error(
                errors,
                counts,
                "record_mapping",
                "Frame metadata has no authoritative mapping row.",
                frame_id=record.frame_id,
            )
            continue
        expected_n, expected_time = expected
        image = resolved_images.get(record.frame_id)
        try:
            image_n = int(image.stem) if image else None
        except ValueError:
            image_n = None
        if image_n != expected_n:
            _add_error(
                errors,
                counts,
                "mapping_image_n",
                "Frame image name does not match mapping n.",
                frame_id=record.frame_id,
            )
        if record.timestamp_ms != expected_time:
            _add_error(
                errors,
                counts,
                "mapping_timestamp",
                "timestamp_ms does not match mapping pts_time.",
                frame_id=record.frame_id,
                expected=expected_time,
                actual=record.timestamp_ms,
            )

    return {
        "paths": paths,
        "source_images": source_images,
        "raw_rows": sum(map(len, raw.values())),
        "canonical_rows": sum(map(len, canonical.values())),
        "image_count": image_count,
        "collision_groups": len(
            {(row["video_id"], row["frame_idx"]) for row in collisions}
        ),
        "discarded_aliases": len(collisions),
        "collision_path": collision_path,
    }


def _validate_extraction(
    output_root: Path,
    frame_count: int,
    canonical_rows: int,
    errors: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    """Check that metadata contains the requested successful extraction."""

    path = output_root / "reports" / "extraction_report.json"
    if not path.is_file():
        return
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _add_error(
            errors,
            counts,
            "extraction_report_invalid",
            "Extraction report cannot be read.",
            detail=str(error),
        )
        return
    limit = report.get("requested_limit")
    expected = canonical_rows if limit is None else min(limit, canonical_rows)
    processed = report.get("processed_frames")
    if processed != frame_count or frame_count != expected:
        _add_error(
            errors,
            counts,
            "extraction_counts",
            "Metadata count does not match the extraction request.",
            expected=expected,
            actual=frame_count,
        )
    if report.get("failed_videos"):
        _add_error(
            errors,
            counts,
            "extraction_failures",
            "Extraction contains failed videos.",
        )


def _write_checksums(
    path: Path,
    metadata_path: Path,
    mapping_paths: list[Path],
    image_paths: list[Path],
    collision_path: Path,
    dataset_root: Path,
    output_root: Path,
    deep: bool,
) -> int:
    """Write canonical and optional source SHA-256 checksums."""

    inputs = {metadata_path.resolve()} if metadata_path.is_file() else set()
    if deep:
        inputs.update(item.resolve() for item in mapping_paths)
        inputs.update(item.resolve() for item in image_paths if item.is_file())
        if collision_path.is_file():
            inputs.add(collision_path.resolve())
    entries = []
    for item in sorted(inputs):
        try:
            label = f"output/{item.relative_to(output_root).as_posix()}"
        except ValueError:
            label = f"dataset/{item.relative_to(dataset_root).as_posix()}"
        entries.append((label, _sha256(item)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in entries),
        encoding="utf-8",
    )
    return len(entries)


def _write_audit(path: Path, records: list[FrameRecord], limit: int) -> int:
    """Write deterministic round-robin records for manual review."""

    grouped: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in records:
        grouped[record.video_id].append(record)
    selected = []
    offset = 0
    while len(selected) < min(limit, len(records)):
        for video_id in sorted(grouped):
            if offset < len(grouped[video_id]):
                selected.append(grouped[video_id][offset])
                if len(selected) == min(limit, len(records)):
                    break
        offset += 1

    fields = (
        "frame_id",
        "video_id",
        "frame_idx",
        "timestamp_ms",
        "image_path",
        "thumbnail_path",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: getattr(record, field) for field in fields}
            for record in selected
        )
    return len(selected)


def _write_corpus_report(path: Path, report: dict[str, Any]) -> None:
    """Write a concise validation and rebuild summary."""

    status = "PASSED" if report["valid"] else "FAILED"
    path.write_text(
        "\n".join(
            [
                "# Corpus report",
                "",
                f"- Status: **{status}**",
                f"- Dataset version: `{report['dataset_version']}`",
                f"- Canonical frames: {report['row_count']}",
                f"- Errors: {report['error_count']}",
                f"- Audit samples: {report['audit_rows']}",
                "",
                "## Rebuild",
                "",
                "```bash",
                "PYTHONPATH=src python scripts/prepare_data.py",
                (
                    "PYTHONPATH=src python scripts/prepare_data.py "
                    "--validate-only"
                ),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def validate_dataset(
    dataset_root: PathValue,
    output_root: PathValue,
    dataset_version: str | None = None,
    *,
    deep: bool = False,
    audit_limit: int = 50,
    metadata_path: PathValue | None = None,
) -> dict[str, Any]:
    """Validate canonical metadata and write audit artifacts."""

    if audit_limit < 0:
        raise ValueError("audit_limit must be non-negative")
    dataset_path = Path(dataset_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    frames_path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else output_path / "metadata" / "frames.parquet"
    )
    reports = output_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    validation_path = reports / "validation_report.json"
    corpus_path = reports / "corpus_report.md"
    audit_path = reports / "audit_samples.csv"
    checksums_path = output_path / "checksums.sha256"

    errors: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    if not dataset_path.is_dir():
        _add_error(
            errors,
            error_counts,
            "dataset_missing",
            "Dataset root does not exist.",
            path=str(dataset_path),
        )
    frames, records = _read_records(frames_path, errors, error_counts)
    _validate_identifiers(frames, records, errors, error_counts)
    resolved = _validate_files(
        records,
        dataset_path,
        output_path,
        errors,
        error_counts,
    )
    mapping = _validate_mappings(
        dataset_path,
        output_path,
        records,
        resolved,
        errors,
        error_counts,
    )
    _validate_extraction(
        output_path,
        len(frames),
        mapping["canonical_rows"],
        errors,
        error_counts,
    )
    checksum_count = _write_checksums(
        checksums_path,
        frames_path,
        mapping["paths"],
        mapping["source_images"],
        mapping["collision_path"],
        dataset_path,
        output_path,
        deep,
    )
    audit_rows = _write_audit(audit_path, records, audit_limit)

    report: dict[str, Any] = {
        "dataset_version": dataset_version,
        "dataset_root": str(dataset_path),
        "output_root": str(output_path),
        "metadata_path": str(frames_path),
        "valid": not error_counts,
        "status": "passed" if not error_counts else "failed",
        "row_count": len(frames),
        "mapping_files": len(mapping["paths"]),
        "mapping_rows": mapping["raw_rows"],
        "canonical_mapping_rows": mapping["canonical_rows"],
        "mapping_collisions": mapping["collision_groups"],
        "discarded_aliases": mapping["discarded_aliases"],
        "collision_policy": COLLISION_POLICY,
        "keyframe_images": mapping["image_count"],
        "checksum_files": checksum_count,
        "audit_rows": audit_rows,
        "error_count": sum(error_counts.values()),
        "error_counts": dict(sorted(error_counts.items())),
        "errors": errors,
        "outputs": {
            "validation_report": str(validation_path),
            "corpus_report": str(corpus_path),
            "audit_samples": str(audit_path),
            "mapping_collisions": str(mapping["collision_path"]),
            "checksums": str(checksums_path),
        },
    }
    _write_corpus_report(corpus_path, report)
    write_json(report, validation_path)
    return report


__all__ = ["validate_dataset"]
