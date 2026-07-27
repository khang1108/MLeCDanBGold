from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any, cast

import pandas as pd


PathValue = str | PathLike[str]
MAPPING_COLUMNS = ("n", "pts_time", "fps", "frame_idx")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
COLLISION_POLICY = "keep_smallest_n"


def checked_root(value: PathValue) -> Path:
    """Return an existing resolved dataset directory."""
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def checked_version(value: str) -> str:
    """Return a non-empty normalized dataset version."""
    version = value.strip()
    if not version:
        raise ValueError("dataset_version must not be empty")
    return version


def checked_limit(value: int | None) -> int | None:
    """Validate an optional positive frame limit."""
    if value is not None and value < 1:
        raise ValueError("limit must be greater than zero")
    return value


def mapping_files(root: Path) -> list[Path]:
    """Find official mapping CSV files in deterministic order."""
    candidates = [
        root / "map-keyframes-aic25-b1" / "map-keyframes",
        root / "map-keyframes",
        root / "map-keyframes-aic25-b1",
        root / "map_keyframes",
    ]
    directory = next((path for path in candidates if path.is_dir()), root)
    return sorted(path for path in directory.glob("*.csv") if path.is_file())


def keyframe_images(root: Path) -> list[Path]:
    """Find supported keyframe images under a dataset root."""
    patterns = ["Keyframes_L*/keyframes/*/*", "keyframes/*/*"]
    patterns += ["keyframes/*", "Keyframes_L*/*/*"]
    for pattern in patterns:
        images = [
            path.resolve()
            for path in root.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if images:
            return sorted(images)
    return []


def read_mapping(path: Path) -> pd.DataFrame:
    """Read and validate one authoritative keyframe mapping."""
    table = pd.read_csv(path)
    missing = [column for column in MAPPING_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
    table = table.loc[:, list(MAPPING_COLUMNS)].copy()
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
    video_id: str, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Keep the smallest image number for each repeated frame index."""
    canonical = mapping.drop_duplicates("frame_idx", keep="first").copy()
    values = {
        int(frame_idx): (int(n), float(pts_time))
        for n, pts_time, _fps, frame_idx
        in canonical.itertuples(index=False, name=None)
    }
    collisions = []
    duplicates = mapping[mapping["frame_idx"].duplicated()]
    for n, pts_time, fps, frame_idx in duplicates.itertuples(
        index=False, name=None
    ):
        canonical_n, canonical_time = values[int(frame_idx)]
        collisions.append(
            {
                "video_id": video_id,
                "frame_idx": int(frame_idx),
                "canonical_n": canonical_n,
                "canonical_pts_time": canonical_time,
                "discarded_n": int(n),
                "discarded_pts_time": float(pts_time),
                "fps": float(fps),
                "policy": COLLISION_POLICY,
            }
        )
    clean = cast(pd.DataFrame, canonical.reset_index(drop=True))
    return clean, collisions


def load_mappings(root: Path) -> tuple[
    list[Path], dict[str, pd.DataFrame], dict[str, pd.DataFrame],
    list[dict[str, Any]], list[dict[str, str]],
]:
    """Load mappings and collect per-file validation failures."""
    paths = mapping_files(root)
    raw: dict[str, pd.DataFrame] = {}
    canonical: dict[str, pd.DataFrame] = {}
    collisions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        video_id = path.stem
        try:
            raw[video_id] = read_mapping(path)
            canonical[video_id], discarded = resolve_mapping_collisions(
                video_id,
                raw[video_id],
            )
            collisions.extend(discarded)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            record = {"video_id": video_id, "path": str(path.resolve())}
            record["error"] = str(error)
            errors.append(record)
    return paths, raw, canonical, collisions, errors


def images_by_video(paths: list[Path]) -> dict[str, list[Path]]:
    """Group keyframe paths by parent video directory."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.parent.name].append(path)
    return dict(grouped)


def numeric_images(
    paths: list[Path],
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    """Index images by numeric stem and report invalid or duplicate stems."""
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
                {"n": number, "paths": [str(path) for path in candidates],
                 "reason": "duplicate numeric stem"}
            )
    index = {number: candidates[0] for number, candidates in grouped.items()
             if len(candidates) == 1}
    return index, errors


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
