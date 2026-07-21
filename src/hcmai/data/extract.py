"""Inventory and ingest a mounted AIC keyframe dataset.

This module implements the core offline data pipeline that transforms a
raw AIC 2025 S1 dataset (JPEG keyframes + Kaggle mapping CSVs) into a
canonical ``frames.parquet`` file that downstream components (embeddings,
reranker, API) can rely on.

Mapping algorithm
-----------------
1. ``n`` in each CSV row points to the numeric image filename
   (e.g. ``n=3`` → ``003.jpg``).
2. ``frame_idx`` is taken verbatim from the CSV; it is **never**
   re-computed from ``pts_time`` or FPS because variable-frame-rate
   videos make that mapping unsafe.
3. When multiple rows share the same ``frame_idx``, the row with the
   smallest ``n`` is kept (``keep_smallest_n`` policy); discarded rows
   are written to ``reports/mapping_collisions.csv``.
4. Stable identifiers follow the formula
   ``frame_id = f"{video_id}_{frame_idx:08d}"``.
5. ``timestamp_ms = round(pts_time * 1000)`` is stored for preview and
   temporal search only.

Output layout
-------------
::

    {output_root}/
    ├── metadata/
    │   ├── frames.parquet          # canonical FrameRecord table
    │   └── shards/{video_id}.parquet  # per-video resume checkpoints
    ├── thumbnails/{video_id}/{frame_id}.jpg
    └── reports/
        ├── corpus_inventory.json
        ├── mapping_collisions.csv
        └── extraction_report.json
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from hcmai.common.schemas.frame import FrameRecord
from hcmai.common.utils.io import write_json


PathValue = str | PathLike[str]
MAPPING_COLUMNS = ("n", "pts_time", "fps", "frame_idx")
FRAME_COLUMNS = tuple(FrameRecord.model_fields)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
COLLISION_POLICY = "keep_smallest_n"
COLLISION_COLUMNS = (
    "video_id",
    "frame_idx",
    "canonical_n",
    "canonical_pts_time",
    "canonical_image_path",
    "canonical_sha256",
    "discarded_n",
    "discarded_pts_time",
    "discarded_image_path",
    "discarded_sha256",
    "fps",
    "policy",
)


def _checked_root(value: PathValue) -> Path:
    """Resolve and validate a dataset root directory.

    Args:
        value: Filesystem path to the dataset root directory. Accepts
            any path-like object or string, including ``~`` expansions.

    Returns:
        Resolved absolute ``Path`` pointing to an existing directory.

    Raises:
        FileNotFoundError: If the resolved path does not exist or is
            not a directory.
    """

    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def _checked_version(value: str) -> str:
    """Strip and validate a dataset version string.

    Args:
        value: Raw dataset version label supplied by the caller.

    Returns:
        Stripped, non-empty version string.

    Raises:
        ValueError: If ``value`` is empty or consists only of whitespace.
    """

    version = value.strip()
    if not version:
        raise ValueError("dataset_version must not be empty")
    return version


def _checked_limit(value: int | None) -> int | None:
    """Validate an optional upper bound on the number of ingested frames.

    Args:
        value: Desired frame limit, or ``None`` to process all frames.

    Returns:
        The original ``value`` unchanged if it is ``None`` or positive.

    Raises:
        ValueError: If ``value`` is not ``None`` and is less than one.
    """

    if value is not None and value < 1:
        raise ValueError("limit must be greater than zero")
    return value


def _mapping_files(root: Path) -> list[Path]:
    """Discover official Kaggle mapping CSV files in deterministic order.

    Checks standard sub-directories under ``root`` for mapping CSV files.

    Args:
        root: Resolved dataset root directory.

    Returns:
        Sorted list of ``Path`` objects for every mapping CSV found.
        Returns an empty list if no mapping directory exists.
    """
    candidates = [
        root / "map-keyframes-aic25-b1" / "map-keyframes",
        root / "map-keyframes",
        root / "map-keyframes-aic25-b1",
        root / "map_keyframes",
    ]
    directory = next((d for d in candidates if d.is_dir()), root)
    return sorted(path for path in directory.glob("*.csv") if path.is_file())


def _keyframe_images(root: Path) -> list[Path]:
    """Discover all supported keyframe image files under the dataset root.

    Searches standard keyframe path patterns under ``root`` for image files
    whose suffix is one of ``.jpg``, ``.jpeg``, ``.png``, or ``.webp``.

    Args:
        root: Resolved dataset root directory.

    Returns:
        Sorted list of resolved absolute ``Path`` objects for every
        keyframe image found.
    """
    patterns = [
        "Keyframes_L*/keyframes/*/*",
        "keyframes/*/*",
        "keyframes/*",
        "Keyframes_L*/*/*",
    ]
    images: list[Path] = []
    for pattern in patterns:
        found = [
            path.resolve()
            for path in root.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if found:
            images = found
            break
    return sorted(images)


def _read_mapping(path: Path) -> pd.DataFrame:
    """Read and validate one official Kaggle mapping CSV file.

    Ensures the file contains the required columns ``n``, ``pts_time``,
    ``fps``, and ``frame_idx``, that all values are finite numerics,
    that ``n`` and ``frame_idx`` are integers, that ``n`` is strictly
    monotonically increasing without duplicates, that ``frame_idx`` and
    ``pts_time`` are non-decreasing, and that ``fps`` is constant.

    Args:
        path: Absolute path to the CSV mapping file for one video.

    Returns:
        Cleaned ``DataFrame`` with columns ``(n, pts_time, fps,
        frame_idx)`` in their canonical dtypes, reset index.

    Raises:
        ValueError: If any required column is missing, any value is
            non-finite or violates the monotonicity/uniqueness
            constraints, or the file cannot be parsed.
    """

    table = pd.read_csv(path)
    missing = [column for column in MAPPING_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")

    table = table.loc[:, MAPPING_COLUMNS].copy()
    for column in MAPPING_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    finite = table[list(MAPPING_COLUMNS)].apply(
        lambda column: column.map(
            lambda value: pd.notna(value) and math.isfinite(float(value))
        )
    )
    if table.empty or not finite.all(axis=None):
        raise ValueError(f"Invalid numeric mapping values: {path}")

    for column in ("n", "frame_idx"):
        if not table[column].eq(table[column].round()).all():
            raise ValueError(f"Column {column} must contain integers: {path}")
        table[column] = table[column].astype("int64")
    table["pts_time"] = table["pts_time"].astype("float64")
    table["fps"] = table["fps"].astype("float64")

    if (table["n"] < 1).any() or (table["frame_idx"] < 0).any():
        raise ValueError(f"Invalid frame identifiers in mapping: {path}")
    if (table["pts_time"] < 0).any() or (table["fps"] <= 0).any():
        raise ValueError(f"Invalid time or FPS values in mapping: {path}")
    if table["n"].duplicated().any() or table["n"].diff().dropna().le(0).any():
        raise ValueError(f"Duplicate or non-monotonic n values: {path}")
    if table["frame_idx"].diff().dropna().lt(0).any():
        raise ValueError(f"Decreasing frame_idx values in mapping: {path}")
    if table["pts_time"].diff().dropna().lt(0).any():
        raise ValueError(f"Decreasing pts_time values in mapping: {path}")
    if table["fps"].round(6).nunique() != 1:
        raise ValueError(f"Inconsistent FPS values in mapping: {path}")
    return table.reset_index(drop=True)


def resolve_mapping_collisions(
    video_id: str,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Resolve duplicate ``frame_idx`` entries by keeping the smallest ``n``.

    When the Kaggle mapping CSV contains multiple rows that share the
    same ``frame_idx``, only the row with the smallest ``n`` (i.e. the
    earliest numeric image filename) is retained as the canonical
    representative.  All other rows are recorded as collision events.

    Args:
        video_id: Identifier of the video being processed (used only to
            populate the collision records).
        mapping: Raw mapping ``DataFrame`` for the video, containing at
            least the columns ``n``, ``frame_idx``, ``pts_time``, and
            ``fps``.  Must already be sorted by ``n`` in ascending order
            as produced by ``_read_mapping``.

    Returns:
        A two-tuple ``(canonical, collisions)`` where:

        * ``canonical`` – ``DataFrame`` with one row per unique
          ``frame_idx``, retaining the row with the smallest ``n``.
        * ``collisions`` – list of ``dict`` records describing each
          discarded row, suitable for the collision report.
    """

    canonical = mapping.drop_duplicates("frame_idx", keep="first").copy()
    canonical_by_idx = canonical.set_index("frame_idx")
    collisions = []
    for row in mapping[mapping["frame_idx"].duplicated()].itertuples(
        index=False
    ):
        kept = canonical_by_idx.loc[int(row.frame_idx)]
        collisions.append(
            {
                "video_id": video_id,
                "frame_idx": int(row.frame_idx),
                "canonical_n": int(kept["n"]),
                "canonical_pts_time": float(kept["pts_time"]),
                "discarded_n": int(row.n),
                "discarded_pts_time": float(row.pts_time),
                "fps": float(row.fps),
                "policy": COLLISION_POLICY,
            }
        )
    return canonical.reset_index(drop=True), collisions


def _load_mappings(
    root: Path,
) -> tuple[
    list[Path],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    """Load all mapping CSVs and resolve frame-index collisions.

    For each CSV found by ``_mapping_files``, the file is read with
    ``_read_mapping`` and collision-resolved with
    ``resolve_mapping_collisions``.  Files that raise parsing or
    validation errors are collected in the ``errors`` list rather than
    raising immediately so that the caller can decide how to proceed.

    Args:
        root: Resolved dataset root directory.

    Returns:
        A five-tuple ``(paths, raw, canonical, collisions, errors)``
        where:

        * ``paths`` – list of all mapping ``Path`` objects discovered.
        * ``raw`` – ``dict`` mapping ``video_id`` to the unresolved
          ``DataFrame`` from each CSV.
        * ``canonical`` – ``dict`` mapping ``video_id`` to the
          collision-resolved ``DataFrame``.
        * ``collisions`` – list of collision event ``dict`` records
          across all videos.
        * ``errors`` – list of ``dict`` records for files that could
          not be read or validated.
    """

    paths = _mapping_files(root)
    raw: dict[str, pd.DataFrame] = {}
    canonical: dict[str, pd.DataFrame] = {}
    collisions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        video_id = path.stem
        try:
            if video_id in raw:
                raise ValueError("duplicate mapping file")
            raw[video_id] = _read_mapping(path)
            canonical[video_id], discarded = resolve_mapping_collisions(
                video_id,
                raw[video_id],
            )
            collisions.extend(discarded)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            errors.append(
                {
                    "video_id": video_id,
                    "path": str(path.resolve()),
                    "error": str(error),
                }
            )
    return paths, raw, canonical, collisions, errors


def _select_rows(
    mappings: dict[str, pd.DataFrame],
    limit: int | None,
) -> dict[str, pd.DataFrame]:
    """Select mapping rows via round-robin sampling across all videos.

    When ``limit`` is ``None``, the full mapping is returned unchanged.
    Otherwise frames are chosen one-at-a-time from each video in
    alphabetical order, cycling until ``limit`` rows have been selected.
    This ensures that a small fixture covers many different videos rather
    than exhausting a single video first.

    Args:
        mappings: Dict mapping each ``video_id`` to its canonical
            mapping ``DataFrame``.
        limit: Total number of rows to return across all videos, or
            ``None`` to return all rows.

    Returns:
        Dict with the same keys as ``mappings`` but each ``DataFrame``
        contains only the selected rows, with the index reset.
    """

    if limit is None:
        return {video: table.copy() for video, table in mappings.items()}
    selected: dict[str, list[int]] = defaultdict(list)
    positions = {video: 0 for video in sorted(mappings)}
    count = 0
    while count < limit:
        advanced = False
        for video_id in positions:
            position = positions[video_id]
            if position >= len(mappings[video_id]):
                continue
            selected[video_id].append(position)
            positions[video_id] += 1
            count += 1
            advanced = True
            if count == limit:
                break
        if not advanced:
            break
    return {
        video: mappings[video].iloc[indices].reset_index(drop=True)
        for video, indices in selected.items()
    }


def _images_by_video(paths: list[Path]) -> dict[str, list[Path]]:
    """Group keyframe image paths by their parent directory name (video ID).

    The parent directory name is assumed to be the ``video_id``
    (e.g. ``Keyframes_L21/keyframes/L21_V001/003.jpg`` → ``L21_V001``).

    Args:
        paths: Flat list of keyframe image paths as returned by
            ``_keyframe_images``.

    Returns:
        Dict mapping each ``video_id`` to the list of image paths
        belonging to that video.
    """

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.parent.name].append(path)
    return dict(grouped)


def _numeric_images(
    paths: list[Path],
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    """Build an integer-keyed index of image paths from numeric filenames.

    Each file is expected to have a purely numeric stem (e.g. ``003.jpg``
    → key ``3``).  Files with non-numeric stems or duplicate stems are
    excluded from the index and recorded as errors.

    Args:
        paths: List of image paths for a single video, as returned by
            ``_images_by_video``.

    Returns:
        A two-tuple ``(index, errors)`` where:

        * ``index`` – ``dict`` mapping each integer ``n`` to its unique
          ``Path``; entries with duplicate stems are omitted.
        * ``errors`` – list of ``dict`` records describing non-numeric
          stems or duplicate numeric stems.
    """

    grouped: dict[int, list[Path]] = defaultdict(list)
    errors: list[dict[str, Any]] = []
    for path in paths:
        try:
            grouped[int(path.stem)].append(path)
        except ValueError:
            errors.append({"path": str(path), "reason": "non-numeric stem"})
    for number, candidates in sorted(grouped.items()):
        if len(candidates) > 1:
            errors.append(
                {
                    "n": number,
                    "paths": [str(path) for path in candidates],
                    "reason": "duplicate numeric stem",
                }
            )
    return (
        {
            number: candidates[0]
            for number, candidates in grouped.items()
            if len(candidates) == 1
        },
        errors,
    )


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file's contents.

    Reads the file in 1 MiB chunks to avoid loading large files into
    memory all at once.

    Args:
        path: Absolute path to the file to hash.

    Returns:
        Lowercase hex string of the SHA-256 digest (64 characters).

    Raises:
        OSError: If the file cannot be opened or read.
    """

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collision_report_rows(
    collisions: list[dict[str, Any]],
    images_by_video: dict[str, list[Path]],
) -> list[dict[str, Any]]:
    """Enrich collision records with resolved image paths and SHA-256 hashes.

    For each collision event produced by ``resolve_mapping_collisions``,
    this function resolves the canonical and discarded image paths from
    ``images_by_video`` and computes their SHA-256 digests so that the
    written CSV can be audited and replayed deterministically.

    Args:
        collisions: List of raw collision ``dict`` records as returned
            by ``resolve_mapping_collisions``.  Each record must contain
            ``video_id``, ``frame_idx``, ``canonical_n``,
            ``discarded_n``, ``pts_time``, and ``fps``.
        images_by_video: Dict mapping each ``video_id`` to the list of
            image paths for that video.

    Returns:
        Sorted list of enriched collision ``dict`` records ordered by
        ``(video_id, frame_idx, discarded_n)``.  Each record contains
        all ``COLLISION_COLUMNS`` fields including
        ``canonical_image_path``, ``canonical_sha256``,
        ``discarded_image_path``, and ``discarded_sha256``.
    """

    indexes = {
        video: _numeric_images(paths)[0]
        for video, paths in images_by_video.items()
    }
    rows = []
    for collision in collisions:
        index = indexes.get(str(collision["video_id"]), {})
        canonical = index.get(int(collision["canonical_n"]))
        discarded = index.get(int(collision["discarded_n"]))
        rows.append(
            {
                **collision,
                "canonical_image_path": (
                    str(canonical.resolve()) if canonical else ""
                ),
                "canonical_sha256": _sha256(canonical) if canonical else "",
                "discarded_image_path": (
                    str(discarded.resolve()) if discarded else ""
                ),
                "discarded_sha256": _sha256(discarded) if discarded else "",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["video_id"]),
            int(row["frame_idx"]),
            int(row["discarded_n"]),
        ),
    )


def _write_collision_report(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write collision decisions to a CSV file atomically.

    Creates the parent directory if necessary, writes to a temporary
    file first, then replaces the target path in a single ``rename``
    call to avoid partial writes.

    Args:
        path: Destination CSV path (e.g.
            ``reports/mapping_collisions.csv``).
        rows: List of enriched collision ``dict`` records as produced by
            ``collision_report_rows``.  Each record must contain all
            keys listed in ``COLLISION_COLUMNS``.

    Raises:
        OSError: If the file cannot be created or the rename fails.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=COLLISION_COLUMNS)
            writer.writeheader()
            writer.writerows(
                {column: row.get(column, "") for column in COLLISION_COLUMNS}
                for row in rows
            )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _inspect_images(
    paths: list[Path],
) -> tuple[Counter[str], Counter[str], list[dict[str, str]]]:
    """Scan keyframe images and gather format, resolution, and health stats.

    Opens each image with Pillow to read its dimensions and calls
    ``Image.verify()`` to detect corruption.  Images that raise an
    ``OSError`` or ``ValueError`` are included in the corrupt list.

    Args:
        paths: Flat list of keyframe image paths to inspect.

    Returns:
        A three-tuple ``(formats, resolutions, corrupt)`` where:

        * ``formats`` – ``Counter`` keyed by lowercase file extension
          (without leading dot) mapping to image count.
        * ``resolutions`` – ``Counter`` keyed by ``"{width}x{height}"``
          strings mapping to image count.
        * ``corrupt`` – list of ``dict`` records with ``path`` and
          ``error`` keys for every image that could not be verified.
    """

    formats: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    corrupt = []
    for path in paths:
        formats[path.suffix.lower().lstrip(".")] += 1
        try:
            with Image.open(path) as image:
                resolutions[f"{image.width}x{image.height}"] += 1
                image.verify()
        except (OSError, ValueError) as error:
            corrupt.append({"path": str(path), "error": str(error)})
    return formats, resolutions, corrupt


def _audit_samples(
    mappings: dict[str, pd.DataFrame],
    grouped_images: dict[str, list[Path]],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Return up to 50 deterministic round-robin samples for manual audit.

    Selects rows via ``_select_rows`` capped at ``min(50, limit or 50)``
    and enriches each row with the resolved image path and derived
    ``frame_id``.  The result is sorted by ``(n, video_id)`` for
    reproducibility.

    Args:
        mappings: Collision-resolved mapping ``DataFrame`` per video.
        grouped_images: Dict mapping each ``video_id`` to its image
            paths, as returned by ``_images_by_video``.
        limit: Optional frame cap passed from the ingestion request.
            At most ``min(50, limit)`` rows are returned.

    Returns:
        List of up to 50 ``dict`` records, each containing
        ``video_id``, ``n``, ``frame_idx``, ``frame_id``,
        ``timestamp_ms``, and ``image_path``.
    """

    selected = _select_rows(mappings, min(50, limit or 50))
    samples = []
    for video_id, table in selected.items():
        images = _numeric_images(grouped_images.get(video_id, []))[0]
        for row in table.itertuples(index=False):
            frame_idx = int(row.frame_idx)
            image = images.get(int(row.n))
            samples.append(
                {
                    "video_id": video_id,
                    "n": int(row.n),
                    "frame_idx": frame_idx,
                    "frame_id": f"{video_id}_{frame_idx:08d}",
                    "timestamp_ms": round(float(row.pts_time) * 1000),
                    "image_path": str(image) if image else None,
                }
            )
    return sorted(samples, key=lambda row: (row["n"], row["video_id"]))[:50]


def _inventory_markdown(report: dict[str, Any]) -> str:
    """Render a concise Markdown summary of the corpus inventory report.

    Args:
        report: Inventory report ``dict`` as returned by
            ``inventory_corpus``.  Must contain at least ``counts``,
            ``mapping_coverage``, and ``dataset_version`` keys.

    Returns:
        Multi-line Markdown string suitable for writing to
        ``corpus_inventory.md``.
    """

    counts = report["counts"]
    coverage = report["mapping_coverage"]
    return "\n".join(
        [
            "# Corpus inventory",
            "",
            f"- Dataset version: `{report['dataset_version']}`",
            f"- Raw mapping rows: {counts['mapping_rows']}",
            f"- Canonical frames: {counts['canonical_mapping_rows']}",
            f"- Mapping collisions: {counts['mapping_collisions']}",
            f"- Keyframe images: {counts['keyframe_images']}",
            f"- Corrupt images: {counts['corrupt_images']}",
            f"- Mapping coverage: {coverage['ratio']:.4f}",
            "",
        ]
    )


def inventory_corpus(
    dataset_root: PathValue,
    output_root: PathValue,
    dataset_version: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Scan the dataset and produce a comprehensive corpus inventory.

    Loads all mapping CSVs, discovers keyframe images and optional video
    files, computes coverage statistics, detects corrupt or duplicate
    images, and writes the following outputs:

    * ``{output_root}/reports/corpus_inventory.json``
    * ``{output_root}/reports/corpus_inventory.md``
    * ``{output_root}/reports/mapping_collisions.csv``

    This function does **not** write ``frames.parquet`` or thumbnails.

    Args:
        dataset_root: Root directory of the mounted AIC dataset.
        output_root: Directory where report files are written.
        dataset_version: Non-empty label identifying the dataset
            release (stored in the report for traceability).
        limit: Optional frame cap passed through to audit sample
            generation.  Does not restrict the inventory scan itself.

    Returns:
        Inventory report ``dict`` containing counts, storage sizes,
        image formats and resolutions, coverage ratios, collision
        records, corrupt images, and audit samples.

    Raises:
        FileNotFoundError: If ``dataset_root`` does not exist.
        ValueError: If ``dataset_version`` is empty.
    """

    root = _checked_root(dataset_root)
    output = Path(output_root).expanduser().resolve()
    version = _checked_version(dataset_version)
    selected_limit = _checked_limit(limit)
    paths, raw, canonical, collisions, mapping_errors = _load_mappings(root)
    images = _keyframe_images(root)
    grouped_images = _images_by_video(images)
    collision_rows = collision_report_rows(collisions, grouped_images)
    formats, resolutions, corrupt = _inspect_images(images)

    matched = 0
    missing_images = []
    missing_mappings = []
    duplicate_images = []
    duplicate_mappings = []
    for video_id in sorted(set(raw) | set(grouped_images)):
        image_index, image_errors = _numeric_images(
            grouped_images.get(video_id, [])
        )
        duplicate_images.extend(
            {"video_id": video_id, **error} for error in image_errors
        )
        mapped = set(raw.get(video_id, pd.DataFrame()).get("n", []))
        mapped = {int(number) for number in mapped}
        for number in sorted(mapped - set(image_index)):
            missing_images.append({"video_id": video_id, "n": number})
        for number in sorted(set(image_index) - mapped):
            missing_mappings.append(
                {
                    "video_id": video_id,
                    "n": number,
                    "image_path": str(image_index[number]),
                }
            )
        matched += len(mapped & set(image_index))
        if video_id in raw:
            for column in ("n", "frame_idx"):
                duplicates = raw[video_id][column].duplicated(keep=False)
                values = set(raw[video_id].loc[duplicates, column])
                for value in sorted(values):
                    duplicate_mappings.append(
                        {
                            "video_id": video_id,
                            "column": column,
                            "value": int(value),
                        }
                    )

    media_info = sorted(
        path
        for directory in root.iterdir()
        if directory.is_dir()
        and "media" in directory.name.lower()
        and "info" in directory.name.lower().replace("_", "-")
        for path in directory.rglob("*")
        if path.is_file()
    )
    videos = sorted(
        path.resolve()
        for directory in root.iterdir()
        if directory.is_dir() and directory.name.lower().startswith("video")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    raw_rows = sum(map(len, raw.values()))
    canonical_rows = sum(map(len, canonical.values()))
    collision_groups = len(
        {(row["video_id"], row["frame_idx"]) for row in collisions}
    )
    all_files = paths + images + media_info + videos
    sizes = {path: path.stat().st_size for path in all_files}
    report: dict[str, Any] = {
        "dataset_version": version,
        "dataset_root": str(root),
        "requested_limit": selected_limit,
        "counts": {
            "mapping_files": len(paths),
            "mapping_rows": raw_rows,
            "canonical_mapping_rows": canonical_rows,
            "mapping_collisions": collision_groups,
            "discarded_aliases": len(collisions),
            "keyframe_images": len(images),
            "media_info_files": len(media_info),
            "video_files": len(videos),
            "corrupt_images": len(corrupt),
            "duplicate_images": len(duplicate_images),
            "duplicate_mappings": len(duplicate_mappings),
        },
        "storage_bytes": {
            "mappings": sum(sizes[path] for path in paths),
            "keyframes": sum(sizes[path] for path in images),
            "media_info": sum(sizes[path] for path in media_info),
            "videos": sum(sizes[path] for path in videos),
        },
        "formats": {
            "images": dict(sorted(formats.items())),
            "videos": dict(
                sorted(
                    Counter(
                        path.suffix[1:].lower() for path in videos
                    ).items()
                )
            ),
        },
        "resolutions": dict(sorted(resolutions.items())),
        "fps": dict(
            sorted(
                Counter(
                    f"{fps:g}"
                    for table in raw.values()
                    for fps in table["fps"]
                ).items()
            )
        ),
        "mapping_coverage": {
            "matched": matched,
            "missing_images": len(missing_images),
            "missing_mappings": len(missing_mappings),
            "ratio": matched / raw_rows if raw_rows else 0.0,
        },
        "corrupt_images": corrupt,
        "duplicate_images": duplicate_images,
        "duplicate_mappings": duplicate_mappings,
        "collision_policy": COLLISION_POLICY,
        "missing_images": missing_images,
        "missing_mappings": missing_mappings,
        "mapping_errors": mapping_errors,
        "unavailable": {"duration": True, "vfr": True, "audio": True},
        "videos": [{"path": str(path)} for path in videos],
        "audit_samples": _audit_samples(
            canonical,
            grouped_images,
            selected_limit,
        ),
    }
    reports = output / "reports"
    _write_collision_report(reports / "mapping_collisions.csv", collision_rows)
    write_json(report, reports / "corpus_inventory.json")
    (reports / "corpus_inventory.md").write_text(
        _inventory_markdown(report),
        encoding="utf-8",
    )
    return report


def _thumbnail(source: Path, target: Path, max_edge: int) -> tuple[int, int]:
    """Validate a source image and write a downscaled JPEG thumbnail.

    Loads the full image to retrieve its dimensions, converts it to
    ``RGB``, applies ``Image.thumbnail`` with LANCZOS resampling so the
    longer edge is at most ``max_edge`` pixels, and saves the result as
    JPEG at quality 85.  Writes to a temporary file first and then
    replaces the target atomically.

    Args:
        source: Absolute path to the original keyframe image.
        target: Desired output path for the thumbnail JPEG.
        max_edge: Maximum pixel length of the longer edge in the
            resized thumbnail.

    Returns:
        A two-tuple ``(width, height)`` of the **original** image
        dimensions in pixels.

    Raises:
        OSError: If the source image cannot be opened or read.
        Exception: Any Pillow or filesystem error; the incomplete
            temporary file is removed before re-raising.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.jpg")
    try:
        with Image.open(source) as image:
            image.load()
            size = image.size
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            thumbnail.save(temporary, format="JPEG", quality=85)
        temporary.replace(target)
        return size
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _frame_records(
    video_id: str,
    mapping: pd.DataFrame,
    image_paths: list[Path],
    thumbnail_root: Path,
    max_edge: int,
) -> list[dict[str, Any]]:
    """Build validated ``FrameRecord`` dicts for every frame of one video.

    Iterates over each mapping row, resolves the corresponding image
    path via ``_numeric_images``, creates a thumbnail with
    ``_thumbnail``, then constructs and validates a ``FrameRecord``
    instance.  All records are returned as plain ``dict`` objects ready
    for assembly into a Parquet shard.

    Args:
        video_id: Identifier of the video being processed.
        mapping: Collision-resolved mapping ``DataFrame`` for the video
            with columns ``n``, ``pts_time``, ``fps``, ``frame_idx``.
        image_paths: List of keyframe image paths belonging to the
            video, as returned by ``_images_by_video``.
        thumbnail_root: Root directory under which per-video thumbnail
            sub-directories are created.
        max_edge: Maximum pixel length of the longer thumbnail edge,
            forwarded to ``_thumbnail``.

    Returns:
        Ordered list of ``dict`` records, one per mapping row, each
        satisfying the ``FrameRecord`` schema.

    Raises:
        ValueError: If any image filename is ambiguous (duplicate numeric
            stem within the video).
        FileNotFoundError: If the image for a mapping row's ``n`` value
            cannot be found in ``image_paths``.
        OSError: If thumbnail creation fails for any frame.
    """

    images, image_errors = _numeric_images(image_paths)
    if image_errors:
        raise ValueError(f"Ambiguous keyframe images for video {video_id}")
    records = []
    for row in mapping.itertuples(index=False):
        image = images.get(int(row.n))
        if image is None:
            raise FileNotFoundError(
                f"Missing keyframe n={int(row.n)} for video {video_id}"
            )
        frame_idx = int(row.frame_idx)
        frame_id = f"{video_id}_{frame_idx:08d}"
        thumbnail = thumbnail_root / video_id / f"{frame_id}.jpg"
        width, height = _thumbnail(image, thumbnail, max_edge)
        records.append(
            FrameRecord(
                frame_id=frame_id,
                video_id=video_id,
                frame_idx=frame_idx,
                timestamp_ms=round(float(row.pts_time) * 1000),
                image_path=str(image.resolve()),
                thumbnail_path=str(thumbnail.resolve()),
                width=width,
                height=height,
            ).model_dump(mode="python")
        )
    return records


def _valid_shard(
    path: Path,
    video_id: str,
    mapping: pd.DataFrame,
    max_edge: int,
) -> bool:
    """Check whether an existing shard Parquet is consistent with the mapping.

    A shard is considered valid when:

    * It can be read and contains exactly the expected ``FRAME_COLUMNS``.
    * ``frame_id`` values match ``{video_id}_{frame_idx:08d}`` for every
      mapping row.
    * ``timestamp_ms`` values match ``round(pts_time * 1000)``.
    * Every referenced image and thumbnail file exists on disk.
    * No thumbnail's longer edge exceeds ``max_edge``.

    Args:
        path: Path to the existing shard Parquet file.
        video_id: Expected video identifier for all rows in the shard.
        mapping: Canonical mapping ``DataFrame`` the shard must match.
        max_edge: Thumbnail size constraint used during ingestion.

    Returns:
        ``True`` if the shard is fully valid and can be reused;
        ``False`` if anything is missing, inconsistent, or unreadable.
    """

    try:
        table = pd.read_parquet(path)
        expected_ids = [
            f"{video_id}_{int(frame_idx):08d}"
            for frame_idx in mapping["frame_idx"]
        ]
        expected_times = [
            round(float(value) * 1000) for value in mapping["pts_time"]
        ]
        if (
            list(table.columns) != list(FRAME_COLUMNS)
            or table["frame_id"].tolist() != expected_ids
            or table["timestamp_ms"].tolist() != expected_times
        ):
            return False
        for record, source in zip(
            table.itertuples(index=False),
            mapping.itertuples(index=False),
        ):
            image = Path(record.image_path)
            thumbnail = Path(record.thumbnail_path)
            if (
                int(image.stem) != int(source.n)
                or not image.is_file()
                or not thumbnail.is_file()
            ):
                return False
            with Image.open(thumbnail) as preview:
                if max(preview.size) > max_edge:
                    return False
        return True
    except (OSError, TypeError, ValueError, KeyError):
        return False


def _write_parquet(table: pd.DataFrame, path: Path) -> None:
    """Write a ``DataFrame`` to a Parquet file atomically.

    Writes to a temporary ``.tmp.parquet`` file first and then renames
    it to ``path`` to prevent partial files from surviving a crash.
    Creates the parent directory if it does not already exist.

    Args:
        table: ``DataFrame`` to serialize.  Written without an index.
        path: Destination Parquet path.

    Raises:
        OSError: If the directory cannot be created, the write fails,
            or the rename fails.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    try:
        table.to_parquet(temporary, index=False)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def ingest_dataset(
    dataset_root: PathValue,
    output_root: PathValue,
    dataset_version: str,
    *,
    limit: int | None = None,
    resume: bool = True,
    thumbnail_max_edge: int = 320,
) -> Path:
    """Build canonical frame metadata with per-video Parquet checkpoints.

    For each video in the canonical mapping, either reuses an existing
    valid shard (when ``resume=True``) or generates frame records from
    scratch: resolves image paths, creates JPEG thumbnails, validates
    each record against ``FrameRecord``, and persists a per-video shard
    to ``{output_root}/metadata/shards/{video_id}.parquet``.

    After all videos are processed, all shards are merged into a single
    ``{output_root}/metadata/frames.parquet`` sorted by
    ``(video_id, timestamp_ms, frame_idx)``.

    Side effects:
        * Writes or updates per-video shards.
        * Writes ``{output_root}/metadata/frames.parquet``.
        * Writes ``{output_root}/reports/mapping_collisions.csv``.
        * Writes ``{output_root}/reports/extraction_report.json``.

    Args:
        dataset_root: Root directory of the mounted AIC dataset.
        output_root: Directory for generated metadata, thumbnails,
            and reports.
        dataset_version: Non-empty label stored in the extraction
            report for traceability.
        limit: Maximum number of frames to process across all videos.
            Frames are selected round-robin so the fixture covers many
            videos. ``None`` processes the full dataset.
        resume: When ``True`` (default), skip videos that already have
            a valid shard on disk.  Set to ``False`` to rebuild all
            shards from scratch.
        thumbnail_max_edge: Maximum pixel length of the longer edge in
            generated thumbnail JPEGs.  Defaults to 320.

    Returns:
        Path to the merged ``{output_root}/metadata/frames.parquet``
        file.

    Raises:
        FileNotFoundError: If ``dataset_root`` does not exist or no
            mapping CSV files are found.
        ValueError: If ``dataset_version`` is empty,
            ``thumbnail_max_edge`` is less than 1, or duplicate
            canonical frame identifiers are detected after merging.
    """

    root = _checked_root(dataset_root)
    output = Path(output_root).expanduser().resolve()
    version = _checked_version(dataset_version)
    selected_limit = _checked_limit(limit)
    if thumbnail_max_edge < 1:
        raise ValueError("thumbnail_max_edge must be greater than zero")

    paths, raw, canonical, collisions, failures = _load_mappings(root)
    if not paths:
        raise FileNotFoundError(f"No mapping CSV files found under {root}")
    selected = _select_rows(canonical, selected_limit)
    grouped_images = _images_by_video(_keyframe_images(root))
    collision_path = output / "reports" / "mapping_collisions.csv"
    _write_collision_report(
        collision_path,
        collision_report_rows(collisions, grouped_images),
    )

    shards = output / "metadata" / "shards"
    thumbnails = output / "thumbnails"
    shard_paths = []
    successful = []
    skipped = []
    created = 0
    resumed = 0
    for video_id, mapping in sorted(selected.items()):
        shard = shards / f"{video_id}.parquet"
        if resume and _valid_shard(
            shard,
            video_id,
            mapping,
            thumbnail_max_edge,
        ):
            shard_paths.append(shard)
            skipped.append(video_id)
            resumed += len(mapping)
            continue
        try:
            records = _frame_records(
                video_id,
                mapping,
                grouped_images.get(video_id, []),
                thumbnails,
                thumbnail_max_edge,
            )
            _write_parquet(
                pd.DataFrame(records, columns=FRAME_COLUMNS),
                shard,
            )
        except Exception as error:
            failures.append({"video_id": video_id, "error": str(error)})
            continue
        shard_paths.append(shard)
        successful.append(video_id)
        created += len(records)

    tables = [pd.read_parquet(path) for path in shard_paths]
    frames = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(columns=FRAME_COLUMNS)
    )
    if frames["frame_id"].duplicated().any() or frames.duplicated(
        ["video_id", "frame_idx"]
    ).any():
        raise ValueError("Duplicate canonical frame identifiers")
    frames = frames.sort_values(
        ["video_id", "timestamp_ms", "frame_idx"],
        kind="stable",
    ).reset_index(drop=True)
    frames_path = output / "metadata" / "frames.parquet"
    _write_parquet(frames.loc[:, FRAME_COLUMNS], frames_path)

    collision_groups = len(
        {(row["video_id"], row["frame_idx"]) for row in collisions}
    )
    write_json(
        {
            "dataset_version": version,
            "dataset_root": str(root),
            "output_root": str(output),
            "requested_limit": selected_limit,
            "raw_mapping_rows": sum(map(len, raw.values())),
            "canonical_mapping_rows": sum(map(len, canonical.values())),
            "mapping_collisions": collision_groups,
            "discarded_aliases": len(collisions),
            "collision_policy": COLLISION_POLICY,
            "collision_report": str(collision_path),
            "processed_frames": len(frames),
            "created_frames": created,
            "resumed_frames": resumed,
            "successful_videos": successful,
            "skipped_videos": skipped,
            "failed_videos": failures,
            "frames_path": str(frames_path),
        },
        output / "reports" / "extraction_report.json",
    )
    return frames_path


def prepare_dataset(
    dataset_root: PathValue,
    output_root: PathValue,
    dataset_version: str,
    *,
    limit: int | None = None,
    resume: bool = True,
    thumbnail_max_edge: int = 320,
    deep_validation: bool = False,
) -> Path:
    """Run the full offline data pipeline: inventory → ingest → validate.

    This is the recommended entry point for preparing the canonical frame
    metadata.  It sequentially calls:

    1. ``inventory_corpus`` – scan the dataset and write summary reports.
    2. ``ingest_dataset`` – build ``frames.parquet`` with per-video
       checkpoints and JPEG thumbnails.
    3. ``validate_dataset`` – verify the produced metadata against the
       source mappings and write audit artifacts.

    Args:
        dataset_root: Root directory of the mounted AIC dataset.
        output_root: Directory for generated metadata, thumbnails,
            and all report files.
        dataset_version: Non-empty label identifying the dataset
            release, stored in all output reports.
        limit: Maximum number of frames to ingest across all videos.
            ``None`` processes the full dataset.  Useful for generating
            a small fixture (e.g. ``limit=100``) before a full run.
        resume: Skip videos with valid existing shards when ``True``
            (default).  Set to ``False`` to rebuild all shards.
        thumbnail_max_edge: Maximum pixel length of the longer edge in
            generated thumbnail JPEGs.  Defaults to 320.
        deep_validation: When ``True``, the final validation step also
            verifies SHA-256 checksums of mapping CSVs and source images.
            Defaults to ``False`` for faster iteration.

    Returns:
        Path to the final ``{output_root}/metadata/frames.parquet`` file
        produced by ``ingest_dataset``.

    Raises:
        FileNotFoundError: If ``dataset_root`` does not exist or no
            mapping CSV files are found.
        ValueError: If ``dataset_version`` is empty,
            ``thumbnail_max_edge`` is less than 1, duplicate canonical
            frame identifiers are detected, or the final validation
            step reports any error.
    """

    inventory_corpus(
        dataset_root,
        output_root,
        dataset_version,
        limit=limit,
    )
    frames_path = ingest_dataset(
        dataset_root,
        output_root,
        dataset_version,
        limit=limit,
        resume=resume,
        thumbnail_max_edge=thumbnail_max_edge,
    )
    from hcmai.data.validate import validate_dataset

    report = validate_dataset(
        dataset_root,
        output_root,
        dataset_version,
        deep=deep_validation,
        metadata_path=frames_path,
    )
    if not report["valid"]:
        raise ValueError(
            "Data validation failed; see reports/validation_report.json"
        )
    return frames_path
