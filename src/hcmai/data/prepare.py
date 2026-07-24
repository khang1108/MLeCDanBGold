"""Build canonical frame metadata from official mappings and keyframes."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from PIL import Image

from hcmai.common.schemas import FrameRecord

MAPPING_COLUMNS = ("n", "pts_time", "frame_idx")
FRAME_COLUMNS = (
    "frame_id",
    "video_id",
    "frame_idx",
    "keyframe_order",
    "timestamp_ms",
    "image_path",
    "width",
    "height",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _mapping_files(root: Path) -> list[Path]:
    candidates = (
        root / "features" / "map-keyframes",
        root / "map-keyframes-aic25-b1" / "map-keyframes",
        root / "map-keyframes",
        root / "map_keyframes",
    )
    directory = next((path for path in candidates if path.is_dir()), None)
    if directory is None:
        raise FileNotFoundError("Official mapping directory was not found")
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No mapping CSV files found in {directory}")
    return paths


def _images_by_video(root: Path) -> dict[str, list[Path]]:
    patterns = ("Keyframes_L*/keyframes/*/*", "keyframes/*/*")
    for pattern in patterns:
        images = sorted(
            path.resolve()
            for path in root.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            grouped: dict[str, list[Path]] = {}
            for image in images:
                grouped.setdefault(image.parent.name, []).append(image)
            return grouped
    raise FileNotFoundError("Keyframe image directory was not found")


def _read_mapping(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = [name for name in MAPPING_COLUMNS if name not in table]
    if missing:
        raise ValueError(f"{path.name}: missing columns: {', '.join(missing)}")
    table = table.loc[:, list(MAPPING_COLUMNS)].copy()
    for name in MAPPING_COLUMNS:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    if table.empty or table.isna().any(axis=None):
        raise ValueError(f"{path.name}: required mapping values are null")
    finite = table.apply(
        lambda column: column.map(lambda value: math.isfinite(float(value)))
    )
    if not finite.all(axis=None):
        raise ValueError(f"{path.name}: mapping values must be finite")
    for name in ("n", "frame_idx"):
        if not table[name].eq(table[name].round()).all():
            raise ValueError(f"{path.name}: {name} must contain integers")
        table[name] = table[name].astype("int64")
    if (table["n"] < 1).any():
        raise ValueError(f"{path.name}: n must be a positive integer")
    if (table["frame_idx"] < 0).any():
        raise ValueError(f"{path.name}: frame_idx must be non-negative")
    if (table["pts_time"] < 0).any():
        raise ValueError(f"{path.name}: pts_time must be non-negative")
    return table


def _image_index(paths: list[Path], video_id: str) -> dict[int, Path]:
    index: dict[int, Path] = {}
    for path in paths:
        try:
            order = int(path.stem)
        except ValueError as error:
            raise ValueError(
                f"{video_id}: non-numeric keyframe filename {path.name}"
            ) from error
        if order in index:
            raise ValueError(f"{video_id}: duplicate keyframe order {order}")
        index[order] = path
    return index


def _records_for_video(
    root: Path,
    mapping_path: Path,
    images: list[Path],
) -> list[dict[str, object]]:
    video_id = mapping_path.stem.strip()
    if not video_id:
        raise ValueError(f"{mapping_path}: video_id must not be empty")
    mapping = _read_mapping(mapping_path)
    index = _image_index(images, video_id)
    missing = [int(n) for n in mapping["n"] if int(n) not in index]
    if missing:
        examples = ", ".join(map(str, missing[:5]))
        raise ValueError(
            f"{video_id}: {len(missing)} mapping rows have no image; "
            f"keyframe_order examples: {examples}"
        )
    records = []
    for n, pts_time, frame_idx in mapping.itertuples(index=False, name=None):
        image = index[int(n)]
        relative = image.relative_to(root).as_posix()
        with Image.open(image) as opened:
            width, height = opened.size
        record = FrameRecord(
            frame_id=f"{video_id}_keyframe_{int(n):06d}",
            video_id=video_id,
            frame_idx=int(frame_idx),
            keyframe_order=int(n),
            timestamp_ms=round(float(pts_time) * 1000),
            image_path=relative,
            width=width,
            height=height,
        )
        records.append(record.model_dump(mode="python"))
    return records


def _validate_table(table: pd.DataFrame, expected_rows: int) -> None:
    if list(table.columns) != list(FRAME_COLUMNS):
        raise ValueError("Written Parquet schema does not match FrameRecord")
    if len(table) != expected_rows:
        raise ValueError("Written Parquet row count changed during serialization")
    if table["frame_id"].duplicated().any():
        raise ValueError("Duplicate frame_id values are not allowed")
    for row in table.to_dict(orient="records"):
        FrameRecord.model_validate(row)


def prepare_frames(dataset_root: Path, output_path: Path) -> Path:
    """Write one validated, deterministic canonical ``frames.parquet``."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    images = _images_by_video(root)
    records: list[dict[str, object]] = []

    for mapping_path in _mapping_files(root):
        records.extend(
            _records_for_video(
                root,
                mapping_path,
                images.get(mapping_path.stem, []),
            )
        )
    table = pd.DataFrame(records, columns=FRAME_COLUMNS)
    table = table.sort_values(
        ["video_id", "keyframe_order", "frame_id"], kind="stable"
    ).reset_index(drop=True)
    _validate_table(table, len(records))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        table.to_parquet(temporary, index=False)
        written = pd.read_parquet(temporary)
        _validate_table(written, len(table))
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
