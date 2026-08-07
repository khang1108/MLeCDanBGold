"""Build the public FrameStore with per-video private checkpoints."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
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
from hcmai.data.preprocessing.video import analyze_video, discover_videos, iter_source_frames
from hcmai.data.stores.frame import FrameStore


def _config_hash(config: PreprocessingConfig) -> str:
    """Hash choices that can change selected frames."""
    values = config.model_dump(mode="json", exclude={
        "videos_root", "output_root", "resume", "limit"})
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    """Replace one Parquet only after serialization succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        table.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _group(video_id: str) -> str:
    """Return the organizer batch prefix used by the output tree."""
    return video_id.split("_", 1)[0]


def _checkpoint_path(config: PreprocessingConfig, video_id: str) -> Path:
    """Return one private per-video metadata checkpoint."""
    return config.work_root / _group(video_id) / f"{video_id}.parquet"


def _load_checkpoint(config: PreprocessingConfig,
                     video_path: Path) -> pd.DataFrame | None:
    """Load a completed checkpoint only when config, source, and images match."""
    path = _checkpoint_path(config, video_path.stem)
    if not config.resume or not path.is_file():
        return None
    table = pd.read_parquet(path)
    stat = video_path.stat()
    valid = (
        not table.empty
        and table["_config_hash"].iloc[0] == _config_hash(config)
        and int(table["_source_size"].iloc[0]) == stat.st_size
        and int(table["_source_mtime_ns"].iloc[0]) == stat.st_mtime_ns
        and all((config.output_root / value).is_file() for value in table["image_path"])
    )
    if not valid:
        return None
    return table.drop(columns=["_config_hash", "_source_size", "_source_mtime_ns"])


def _encode_images(paths: list[Path], encoder: Any,
                   batch_size: int) -> np.ndarray:
    """Encode candidate JPEGs in bounded batches."""
    chunks: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start: start + batch_size]:
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
        chunks.append(np.asarray(encoder.encode(images), dtype=np.float32))
    return np.concatenate(chunks) if chunks else np.empty((0, 0), dtype=np.float32)


def _record(candidate: CandidateFrame, relative_path: Path) -> dict[str, object]:
    """Create one canonical runtime frame record."""
    frame = candidate.frame
    return FrameRecord(
        frame_id=f"{frame.video_id}_frame_{frame.frame_idx:09d}",
        video_id=frame.video_id,
        frame_idx=frame.frame_idx,
        timestamp_ms=frame.timestamp_ms,
        image_path=relative_path.as_posix(),
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


def _materialize(video_path: Path, candidates: list[CandidateFrame],
                 config: PreprocessingConfig, encoder: Any) -> pd.DataFrame:
    """Write candidates, remove safe duplicates, and publish one video atomically."""
    group = _group(video_path.stem)
    final_dir = config.output_root / "images" / group / video_path.stem
    partial_dir = final_dir.parent / f".{video_path.stem}.partial"
    shutil.rmtree(partial_dir, ignore_errors=True)
    partial_dir.mkdir(parents=True)
    by_index = {candidate.frame.frame_idx: candidate for candidate in candidates}
    paths: dict[int, Path] = {}
    for frame_meta, source_frame in iter_source_frames(video_path):
        candidate = by_index.get(frame_meta.frame_idx)
        if candidate is None:
            continue
        path = partial_dir / f"{candidate.frame.frame_idx:09d}.jpg"
        source_frame.to_image().convert("RGB").save(
            path, quality=config.image_quality
        )
        paths[frame_meta.frame_idx] = path
    if len(paths) != len(candidates):
        raise ValueError(f"Could not materialize every candidate in {video_path.name}")
    ordered_paths = [paths[item.frame.frame_idx] for item in candidates]
    if config.dino_enabled:
        embeddings = _encode_images(ordered_paths, encoder, config.dino_batch_size)
        retained = deduplicate(candidates, embeddings, config)
    else:
        retained = candidates
    retained_ids = {item.frame.frame_idx for item in retained}
    for frame_idx, path in paths.items():
        if frame_idx not in retained_ids:
            path.unlink()
    shutil.rmtree(final_dir, ignore_errors=True)
    partial_dir.replace(final_dir)
    records = [
        _record(item, Path("images") / group / video_path.stem / paths[
            item.frame.frame_idx
        ].name)
        for item in retained
    ]
    return pd.DataFrame(records, columns=FrameRecord.model_fields)


def _prepare_video(path: Path, config: PreprocessingConfig,
                   shot_detector: Any, event_detector: Any,
                   encoder: Any) -> pd.DataFrame:
    """Prepare one video or reuse its complete private checkpoint."""
    cached = _load_checkpoint(config, path)
    if cached is not None:
        return cached
    analysis = analyze_video(path, config)
    shot_scores = shot_detector.score(path, analysis.shot_frames)
    event_scores = event_detector.score(path, analysis.shot_frames)
    candidates = select_candidates(analysis.frames, shot_scores, event_scores, config)
    table = _materialize(path, candidates, config, encoder)
    stat = path.stat()
    checkpoint = table.assign(
        _config_hash=_config_hash(config),
        _source_size=stat.st_size,
        _source_mtime_ns=stat.st_mtime_ns,
    )
    _atomic_parquet(checkpoint, _checkpoint_path(config, path.stem))
    return table


def prepare_frame_store(
    config: PreprocessingConfig, *, shot_detector: Any | None = None,
    event_detector: Any | None = None, encoder: Any | None = None,
) -> Path:
    """Build and return the single public ``frames.parquet`` path."""
    config.output_root.mkdir(parents=True, exist_ok=True)
    shot_detector = shot_detector or TransNetDetector(config)
    event_detector = event_detector or EfficientGEBDDetector(config)
    encoder = encoder or DinoEncoder(config)
    tables = [
        _prepare_video(path, config, shot_detector, event_detector, encoder)
        for path in discover_videos(config)
    ]
    frames = pd.concat(tables, ignore_index=True).sort_values(
        ["video_id", "timestamp_ms", "frame_idx"], kind="stable"
    )
    if frames["frame_id"].duplicated().any():
        raise ValueError("Duplicate canonical frame IDs are not allowed")
    output = config.output_root / "frames.parquet"
    _atomic_parquet(frames.reset_index(drop=True), output)
    FrameStore.load(output)
    return output
