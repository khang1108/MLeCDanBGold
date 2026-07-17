"""Inventory and ingest a mounted AIC keyframe dataset."""

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
    """Return an existing dataset directory."""

    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def _checked_version(value: str) -> str:
    """Return a non-empty dataset version."""

    version = value.strip()
    if not version:
        raise ValueError("dataset_version must not be empty")
    return version


def _checked_limit(value: int | None) -> int | None:
    """Return a valid optional frame limit."""

    if value is not None and value < 1:
        raise ValueError("limit must be greater than zero")
    return value


def _mapping_files(root: Path) -> list[Path]:
    """Return official mapping files in deterministic order."""

    directory = root / "map-keyframes-aic25-b1" / "map-keyframes"
    return sorted(path for path in directory.glob("*.csv") if path.is_file())


def _keyframe_images(root: Path) -> list[Path]:
    """Return all supported keyframe images."""

    return sorted(
        path.resolve()
        for path in root.glob("Keyframes_L*/keyframes/*/*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _read_mapping(path: Path) -> pd.DataFrame:
    """Read and validate one official mapping CSV."""

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
    """Keep the smallest n for each authoritative frame index."""

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
    """Load mappings and resolve known frame-index collisions."""

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
    """Select rows round-robin across videos."""

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
    """Group keyframe paths by video ID."""

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.parent.name].append(path)
    return dict(grouped)


def _numeric_images(
    paths: list[Path],
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    """Index numeric image names and report ambiguous files."""

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
    """Return one file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collision_report_rows(
    collisions: list[dict[str, Any]],
    images_by_video: dict[str, list[Path]],
) -> list[dict[str, Any]]:
    """Add source paths and hashes to collision decisions."""

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
    """Write deterministic collision decisions."""

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
    """Count image formats, resolutions, and corrupt files."""

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
    """Return at most 50 deterministic mapping samples."""

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
    """Render a concise corpus inventory."""

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
    """Inventory mappings, keyframes, and optional source files."""

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
    """Validate an image and create its thumbnail."""

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
    """Build canonical records for one video."""

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
    """Return whether a checkpoint matches its requested mapping."""

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
    """Atomically write one Parquet table."""

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
    """Build canonical metadata with per-video checkpoints."""

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
    """Run inventory, ingestion, and final validation."""

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
