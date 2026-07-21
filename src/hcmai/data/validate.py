"""Validate canonical frame metadata and freeze data artifacts.

This module provides ``validate_dataset``, a comprehensive quality gate
that runs after ``ingest_dataset`` to ensure every ``FrameRecord`` in
``frames.parquet`` satisfies the canonical schema, references existing
files, and is consistent with the official Kaggle mapping CSVs.

Validation layers
-----------------
1. **Schema** – every row is parsed by ``FrameRecord.model_validate``.
2. **Identifiers** – global uniqueness of ``frame_id`` and per-video
   ``frame_idx``, correct ID formula, and monotonic ordering.
3. **Files** – source images exist with the expected dimensions;
   thumbnails exist and are not larger than the originals.
4. **Mapping cross-check** – each record can be joined back to an
   authoritative CSV row; ``timestamp_ms`` and image ``n`` agree.
5. **Extraction report** – ``processed_frames`` count matches metadata.

Output artifacts
----------------
::

    {output_root}/
    ├── checksums.sha256
    └── reports/
        ├── validation_report.json
        ├── corpus_report.md
        └── audit_samples.csv
"""

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
    """Append a bounded error example and increment its error counter.

    At most ``MAX_ERROR_EXAMPLES`` (200) error dicts are kept in
    ``errors`` to prevent the report from growing without bound.  The
    ``counts`` counter is always updated regardless of the cap.

    Args:
        errors: Mutable list that accumulates error example dicts.
        counts: Mutable ``Counter`` tracking the number of occurrences
            of each error code.
        code: Short snake_case identifier for the error category
            (e.g. ``"duplicate_frame_id"``).
        message: Human-readable description of the error.
        **context: Additional key-value pairs merged into the error dict
            for debugging (e.g. ``frame_id=...``, ``path=...``).
    """

    counts[code] += 1
    if len(errors) < MAX_ERROR_EXAMPLES:
        errors.append({"code": code, "message": message, **context})


def _native(value: Any) -> Any:
    """Convert a pandas scalar or missing value to a plain Python object.

    Pandas operations often return numpy scalars (e.g. ``np.int64``) or
    pandas NA types.  This helper converts ``NA``/``NaN`` to ``None``
    and unwraps numpy scalars via their ``.item()`` method so that the
    result is safe to pass to Pydantic validators.

    Args:
        value: Any scalar value, potentially a pandas or numpy type.

    Returns:
        ``None`` if ``value`` is a missing value recognised by
        ``pd.isna``; the result of ``.item()`` if ``value`` exposes
        that method (numpy scalar); otherwise ``value`` unchanged.
    """

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
    """Read ``frames.parquet`` and validate every row against ``FrameRecord``.

    Attempts to read the Parquet file and then validates each row with
    ``FrameRecord.model_validate(strict=True)``.  Rows that fail
    validation are recorded via ``_add_error`` rather than raising so
    that all schema errors are reported together.

    Args:
        path: Path to ``frames.parquet`` (typically
            ``{output_root}/metadata/frames.parquet``).
        errors: Mutable error list populated by ``_add_error``.
        counts: Mutable error counter populated by ``_add_error``.

    Returns:
        A two-tuple ``(frames, records)`` where:

        * ``frames`` – raw ``DataFrame`` (empty if the file cannot be
          read).
        * ``records`` – list of successfully validated ``FrameRecord``
          objects (one per valid row).
    """

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
    """Validate identifier uniqueness, format correctness, and per-video order.

    Checks the following invariants and records a bounded error for each
    violation:

    * ``frame_id`` values are globally unique across the whole table.
    * ``(video_id, frame_idx)`` pairs are unique.
    * Each ``frame_id`` equals ``f"{video_id}_{frame_idx:08d}"``.
    * Within each video, frames are sorted by
      ``(timestamp_ms, frame_idx)`` (monotonically non-decreasing).

    Args:
        frames: Raw ``DataFrame`` as returned by ``_read_records``.
            May be empty.
        records: List of validated ``FrameRecord`` objects from
            ``_read_records``.
        errors: Mutable error list populated by ``_add_error``.
        counts: Mutable error counter populated by ``_add_error``.
    """

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
    """Resolve an artifact path that may be absolute or root-relative.

    If ``value`` is an absolute path it is returned directly.  Otherwise
    each root in ``roots`` is tried in order and the first candidate
    that exists on disk is returned.  If none exist, the first candidate
    is returned as a best guess (so callers can produce a useful error
    message).

    Args:
        value: Raw path string from a ``FrameRecord`` field.
        roots: Iterable of base directories to prepend when the path is
            not absolute (e.g. ``(dataset_root, output_root)``).

    Returns:
        Resolved ``Path`` object.  Not guaranteed to exist on disk.
    """

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
    """Validate source images and thumbnails referenced by the metadata.

    For each ``FrameRecord``, resolves ``image_path`` against
    ``dataset_root`` and ``output_root``, checks that the file exists,
    and verifies that its pixel dimensions match the metadata ``width``
    and ``height`` fields.  Then checks that ``thumbnail_path`` is not
    ``None``, that the thumbnail file exists, and that its dimensions do
    not exceed the source image's dimensions.

    Args:
        records: Validated ``FrameRecord`` objects to inspect.
        dataset_root: Dataset root directory used as a fallback base
            path when ``image_path`` is not absolute.
        output_root: Output root directory used as a fallback base path
            when ``thumbnail_path`` is not absolute.
        errors: Mutable error list populated by ``_add_error``.
        counts: Mutable error counter populated by ``_add_error``.

    Returns:
        Dict mapping each ``frame_id`` to its resolved source image
        ``Path`` (for use in ``_validate_mappings``).
    """

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
    """Verify the on-disk collision CSV matches the recomputed expected rows.

    Reads the CSV at ``path`` and performs a strict ``DataFrame``
    equality check against ``expected_rows`` (recomputed from the source
    mapping).  Any mismatch or parse error is recorded as a
    ``collision_report_invalid`` error.

    Args:
        path: Path to the existing ``mapping_collisions.csv`` file.
        expected_rows: List of enriched collision ``dict`` records as
            produced by ``collision_report_rows`` from the current source
            mapping.
        errors: Mutable error list populated by ``_add_error``.
        counts: Mutable error counter populated by ``_add_error``.
    """

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
    """Cross-validate metadata records against the official Kaggle mapping CSVs.

    Reloads all mapping CSVs, rebuilds the canonical mapping, and
    checks that:

    * Every video has one-to-one coverage between mapping rows and
      keyframe images.
    * The on-disk collision report matches the recomputed expected rows.
    * Each ``FrameRecord`` can be joined to a canonical mapping row on
      ``(video_id, frame_idx)``.
    * The image filename stem matches the mapping row's ``n`` value.
    * ``timestamp_ms`` matches ``round(pts_time * 1000)`` from the CSV.

    Args:
        dataset_root: Root directory of the mounted AIC dataset.
        output_root: Output directory containing ``reports/`` artifacts.
        records: Validated ``FrameRecord`` objects from ``_read_records``.
        resolved_images: Dict mapping ``frame_id`` to the resolved source
            image ``Path``, as returned by ``_validate_files``.
        errors: Mutable error list populated by ``_add_error``.
        counts: Mutable error counter populated by ``_add_error``.

    Returns:
        Summary ``dict`` with keys ``paths``, ``source_images``,
        ``raw_rows``, ``canonical_rows``, ``image_count``,
        ``collision_groups``, ``discarded_aliases``, and
        ``collision_path`` for use in ``validate_dataset``.
    """

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
    """Verify that the extraction report is consistent with the metadata.

    Reads ``{output_root}/reports/extraction_report.json`` and checks
    that ``processed_frames`` matches the number of rows in the current
    ``frames.parquet`` and that no ``failed_videos`` were recorded.
    Silently returns if the report file does not exist (the step is
    skipped for externally produced metadata).

    Args:
        output_root: Output root directory containing ``reports/``.
        frame_count: Number of rows in the current ``frames.parquet``.
        canonical_rows: Total canonical mapping rows across all videos,
            used to compute the expected frame count when a ``limit``
            was specified during ingestion.
        errors: Mutable error list populated by ``_add_error``.
        counts: Mutable error counter populated by ``_add_error``.
    """

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
    """Compute and write SHA-256 checksums for all validated artifacts.

    Always includes ``metadata_path`` (the canonical Parquet file).
    When ``deep=True``, also includes all mapping CSV files, the
    collision report, and every source keyframe image.  Each line in the
    output file follows the standard ``sha256sum`` format::

        <hex_digest>  <label>\n

    Labels are relative paths prefixed with ``output/`` for files under
    ``output_root`` and ``dataset/`` for files under ``dataset_root``.

    Args:
        path: Destination path for the ``checksums.sha256`` file.
        metadata_path: Path to ``frames.parquet`` (always hashed).
        mapping_paths: List of mapping CSV ``Path`` objects discovered
            by ``_load_mappings``.
        image_paths: List of keyframe image ``Path`` objects (hashed
            only when ``deep=True``).
        collision_path: Path to ``mapping_collisions.csv`` (hashed
            only when ``deep=True`` and the file exists).
        dataset_root: Dataset root used to compute relative labels for
            files that live outside ``output_root``.
        output_root: Output root used to compute relative labels.
        deep: When ``True``, hash mapping CSVs, source images, and the
            collision report in addition to the metadata file.

    Returns:
        Total number of files whose checksums were written.
    """

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
    """Write a deterministic round-robin audit sample CSV for manual review.

    Selects up to ``limit`` records spread evenly across all videos
    using a round-robin strategy, then writes them to a CSV file with
    the columns ``frame_id``, ``video_id``, ``frame_idx``,
    ``timestamp_ms``, ``image_path``, and ``thumbnail_path``.

    Args:
        path: Destination path for the audit CSV file.
        records: All validated ``FrameRecord`` objects from the metadata.
        limit: Maximum number of records to write.  The actual count
            may be less if ``len(records) < limit``.

    Returns:
        Number of records written to the CSV file.
    """

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
    """Write a concise Markdown validation and rebuild summary.

    Produces a short Markdown file (``corpus_report.md``) showing the
    validation status, canonical frame count, error count, and audit
    sample count, plus the CLI commands needed to rebuild or re-validate
    the dataset.

    Args:
        path: Destination path for the Markdown report file.
        report: Validation report ``dict`` as assembled in
            ``validate_dataset``.  Must contain ``valid``,
            ``dataset_version``, ``row_count``, ``error_count``, and
            ``audit_rows`` keys.
    """

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
    """Validate canonical frame metadata and write all audit artifacts.

    Runs five sequential validation layers (schema, identifiers, files,
    mapping cross-check, extraction report) and writes four output
    artifacts regardless of whether any errors were found:

    * ``{output_root}/reports/validation_report.json``
    * ``{output_root}/reports/corpus_report.md``
    * ``{output_root}/reports/audit_samples.csv``
    * ``{output_root}/checksums.sha256``

    The ``valid`` key in the returned report is ``True`` only when all
    five layers produce zero errors.

    Args:
        dataset_root: Root directory of the mounted AIC dataset.  Used
            to reload mapping CSVs for the cross-validation step.
        output_root: Directory containing the ingested metadata and
            where audit artifacts are written.
        dataset_version: Optional label stored verbatim in the report
            for traceability.  ``None`` is allowed (the field is
            included as ``null`` in the JSON output).
        deep: When ``True``, the checksum file includes SHA-256 digests
            for all mapping CSVs and source keyframe images in addition
            to ``frames.parquet``.  Defaults to ``False``.
        audit_limit: Maximum number of records to write to
            ``audit_samples.csv``.  Defaults to 50.
        metadata_path: Override the default metadata path
            ``{output_root}/metadata/frames.parquet``.  Useful when
            validating a specific shard or an externally produced file.

    Returns:
        Validation report ``dict`` with the following top-level keys:
        ``valid``, ``status``, ``row_count``, ``error_count``,
        ``error_counts``, ``errors``, ``outputs``, and various
        mapping/checksum statistics.

    Raises:
        ValueError: If ``audit_limit`` is negative.
    """

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
