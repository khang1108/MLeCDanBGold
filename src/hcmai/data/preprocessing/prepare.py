"""Build one public FrameStore from raw videos."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from hcmai.common.schemas.frame import FrameRecord
from hcmai.data.preprocessing.config import PreprocessingConfig
from hcmai.data.preprocessing.models import EfficientGEBDDetector, TransNetDetector
from hcmai.data.preprocessing.selection import (
    CandidateFrame, DinoEncoder, deduplicate, select_candidates,
)
from hcmai.data.preprocessing.video import (
    analyze_video,
    discover_videos,
    iter_source_frames,
)

_CHECKPOINT_COLUMNS = ["_config_hash", "_source_size", "_source_mtime_ns"]


def _config_hash(config: PreprocessingConfig) -> str:
    """Hash settings that affect selected frames."""
    values = config.model_dump(
        mode="json",
        exclude={"videos_root", "output_root"},
    )
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_parquet(table: pd.DataFrame, path: Path) -> None:
    """Publish Parquet only after writing it completely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.partial")
    table.to_parquet(partial, index=False)
    partial.replace(path)


def _checkpoint_path(config: PreprocessingConfig, video_id: str) -> Path:
    """Return the private checkpoint for one video."""
    return config.work_root / video_id.split("_", 1)[0] / f"{video_id}.parquet"


def _load_checkpoint(
    config: PreprocessingConfig, video_path: Path, resume: bool,
) -> pd.DataFrame | None:
    """Reuse a checkpoint when its source, config, and images still match."""
    path = _checkpoint_path(config, video_path.stem)
    if not resume or not path.is_file():
        return None
    table = pd.read_parquet(path)
    stat = video_path.stat()
    valid = (
        not table.empty
        and table["_config_hash"].iloc[0] == _config_hash(config)
        and int(table["_source_size"].iloc[0]) == stat.st_size
        and int(table["_source_mtime_ns"].iloc[0]) == stat.st_mtime_ns
        and all(
            (config.output_root / value).is_file()
            for value in table["image_path"]
        )
    )
    return table.drop(columns=_CHECKPOINT_COLUMNS) if valid else None


def _encode_images(paths: list[Path], encoder: Any, batch_size: int) -> np.ndarray:
    """Encode candidate images in small batches."""
    chunks = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        chunks.append(np.asarray(encoder.encode(images), dtype=np.float32))
    return np.concatenate(chunks)


def _record(candidate: CandidateFrame, image_path: Path) -> dict[str, object]:
    """Convert one candidate to canonical frame metadata."""
    frame = candidate.frame
    return FrameRecord(
        frame_id=f"{frame.video_id}_frame_{frame.frame_idx:09d}",
        video_id=frame.video_id,
        frame_idx=frame.frame_idx,
        timestamp_ms=frame.timestamp_ms,
        image_path=image_path.as_posix(),
        width=frame.width,
        height=frame.height,
        shot_id=f"{frame.video_id}_shot_{candidate.shot_id:06d}",
        is_anchor=candidate.protected,
        pts=frame.pts,
        time_base=frame.time_base,
        motion_score=frame.motion_score,
        shot_score=candidate.shot_score,
        event_score=candidate.event_score,
        selection_reasons=candidate.reasons,
    ).model_dump(mode="python")


def _materialize(
    video_path: Path,
    candidates: list[CandidateFrame],
    config: PreprocessingConfig,
    encoder: Any,
) -> pd.DataFrame:
    """Write, deduplicate, and atomically publish one video's candidates."""
    group = video_path.stem.split("_", 1)[0]
    final_dir = config.output_root / "images" / group / video_path.stem
    partial_dir = final_dir.parent / f".{video_path.stem}.partial"
    shutil.rmtree(partial_dir, ignore_errors=True)
    partial_dir.mkdir(parents=True)

    by_index = {item.frame.frame_idx: item for item in candidates}
    images: dict[int, Path] = {}
    for frame, source in iter_source_frames(video_path):
        candidate = by_index.get(frame.frame_idx)
        if candidate is None:
            continue
        image_path = partial_dir / f"{frame.frame_idx:09d}.jpg"
        source.to_image().convert("RGB").save(
            image_path, quality=config.image_quality
        )
        images[frame.frame_idx] = image_path
    if len(images) != len(candidates):
        raise ValueError(f"Could not extract every candidate from {video_path.name}")

    paths = [images[item.frame.frame_idx] for item in candidates]
    vectors = _encode_images(paths, encoder, config.dino_batch_size)
    retained = deduplicate(candidates, vectors, config)
    retained_ids = {item.frame.frame_idx for item in retained}
    for frame_idx, image_path in images.items():
        if frame_idx not in retained_ids:
            image_path.unlink()

    shutil.rmtree(final_dir, ignore_errors=True)
    partial_dir.replace(final_dir)
    image_root = Path("images") / group / video_path.stem
    rows = [
        _record(item, image_root / images[item.frame.frame_idx].name)
        for item in retained
    ]
    return pd.DataFrame(rows, columns=FrameRecord.model_fields)


def _prepare_video(
    path: Path, config: PreprocessingConfig, shot_detector: Any,
    event_detector: Any, encoder: Any, resume: bool,
) -> pd.DataFrame:
    """Prepare one video or reuse its checkpoint."""
    cached = _load_checkpoint(config, path, resume)
    if cached is not None:
        return cached
    analysis = analyze_video(path, config, event_detector)
    candidates = select_candidates(
        analysis.frames,
        shot_detector.score(path, analysis.shot_frames),
        analysis.event_scores,
        config,
    )
    table = _materialize(path, candidates, config, encoder)
    stat = path.stat()
    checkpoint = table.assign(
        _config_hash=_config_hash(config),
        _source_size=stat.st_size,
        _source_mtime_ns=stat.st_mtime_ns,
    )
    _write_parquet(checkpoint, _checkpoint_path(config, path.stem))
    return table


def prepare_frame_store(
    config: PreprocessingConfig,
    *,
    shot_detector: Any | None = None,
    event_detector: Any | None = None,
    encoder: Any | None = None,
    resume: bool = True,
    limit: int | None = None,
) -> Path:
    """Build and return the canonical ``frames.parquet`` path."""
    config.output_root.mkdir(parents=True, exist_ok=True)
    shot_detector = shot_detector or TransNetDetector(config)
    event_detector = event_detector or EfficientGEBDDetector(config)
    encoder = encoder or DinoEncoder(config)
    tables = [
        _prepare_video(
            path, config, shot_detector, event_detector, encoder, resume
        )
        for path in discover_videos(config, limit)
    ]
    frames = pd.concat(tables, ignore_index=True).sort_values(
        ["video_id", "timestamp_ms", "frame_idx"], kind="stable"
    )
    output = config.output_root / "frames.parquet"
    _write_parquet(frames.reset_index(drop=True), output)
    return output
