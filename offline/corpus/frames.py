"""Extract deterministic 1-FPS VBS frames from local MP4 files.

The contract is intentionally small:

    <external-videos-root>/*.mp4
        -> <work-root>/frames/<video_id>/<sample>.jpg
        -> data/artifacts/frames.parquet
        -> data/artifacts/manifest.json

There is no downloader, archive lifecycle, S3 publication, media-info input, or
staging/published directory.  ffmpeg/ffprobe are the only external tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

import pandas as pd

from offline.corpus.models import FrameArtifact
from .paths import VBSDataPaths


@dataclass(frozen=True, slots=True)
class FrameBuildConfig:
    """Local frame extraction policy."""

    sample_period_ms: int = 1_000
    long_edge: int = 1_024
    jpeg_quality: int = 92
    frame_store_id: str = "vbs-local-v1"
    resume: bool = True

    def __post_init__(self) -> None:
        if self.sample_period_ms <= 0:
            raise ValueError("sample_period_ms must be positive")
        if self.long_edge <= 0:
            raise ValueError("long_edge must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if not self.frame_store_id.strip():
            raise ValueError("frame_store_id must not be blank")


@dataclass(frozen=True, slots=True)
class FrameBuildReport:
    video_count: int
    frame_count: int
    reused_video_count: int
    frames_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class _VideoProbe:
    video_id: str
    path: Path
    width: int
    height: int
    avg_fps: Fraction
    duration_ms: int | None

    @property
    def fps(self) -> float:
        return float(self.avg_fps)


_IMAGE_PATTERN = "%06d.jpg"


def discover_videos(videos_root: Path) -> list[Path]:
    """Return local MP4 files in deterministic order with unique stems."""

    if not videos_root.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {videos_root}")
    videos = sorted(path for path in videos_root.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in {videos_root}")
    ids = [path.stem for path in videos]
    if len(ids) != len(set(ids)):
        raise ValueError("MP4 filenames must have unique stems")
    return videos


def _run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("ffprobe returned a non-object JSON value")
    return value


def _fraction(value: object) -> Fraction:
    text = str(value or "").strip()
    if not text or text in {"0/0", "N/A"}:
        raise ValueError("video average FPS is unavailable")
    result = Fraction(text)
    if result <= 0:
        raise ValueError("video average FPS must be positive")
    return result


def probe_video(path: Path) -> _VideoProbe:
    """Read only the metadata needed for canonical VBS frame identity."""

    payload = _run_json([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
        "-of", "json",
        str(path),
    ])
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise ValueError(f"No decodable video stream: {path}")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions: {path}")
    avg_fps = _fraction(stream.get("avg_frame_rate"))

    duration_ms: int | None = None
    fmt = payload.get("format")
    if isinstance(fmt, dict):
        raw = fmt.get("duration")
        if raw not in (None, "N/A", ""):
            try:
                duration_ms = max(0, round(float(raw) * 1_000))
            except (TypeError, ValueError):
                duration_ms = None

    return _VideoProbe(
        video_id=path.stem,
        path=path,
        width=width,
        height=height,
        avg_fps=avg_fps,
        duration_ms=duration_ms,
    )


def _target_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    if max(width, height) <= long_edge:
        return width, height
    ratio = long_edge / max(width, height)
    target_width = max(2, int(round(width * ratio)))
    target_height = max(2, int(round(height * ratio)))
    # yuv-backed JPEG encoders are happiest with even dimensions.
    target_width -= target_width % 2
    target_height -= target_height % 2
    return max(2, target_width), max(2, target_height)


def _source_identity(path: Path, config: FrameBuildConfig) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source_path": str(path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "sample_period_ms": config.sample_period_ms,
        "long_edge": config.long_edge,
        "jpeg_quality": config.jpeg_quality,
    }


def _state_path(paths: VBSDataPaths, video_id: str) -> Path:
    return paths.frame_state / f"{video_id}.json"


def _load_state(paths: VBSDataPaths, probe: _VideoProbe, config: FrameBuildConfig) -> dict[str, Any] | None:
    path = _state_path(paths, probe.video_id)
    if not config.resume or not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict) or state.get("source") != _source_identity(probe.path, config):
        return None
    count = state.get("frame_count")
    if not isinstance(count, int) or count <= 0:
        return None
    frame_dir = paths.frames / probe.video_id
    expected = [frame_dir / f"{index:06d}.jpg" for index in range(count)]
    if not all(path.is_file() and path.stat().st_size > 0 for path in expected):
        return None
    return state


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_video(paths: VBSDataPaths, probe: _VideoProbe, config: FrameBuildConfig) -> int:
    """Atomically replace one video's sampled JPEG directory."""

    final_dir = paths.frames / probe.video_id
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    target_width, target_height = _target_size(probe.width, probe.height, config.long_edge)
    sample_fps = 1_000 / config.sample_period_ms
    ffmpeg_quality = max(2, min(31, round(2 + (100 - config.jpeg_quality) * 0.29)))

    with tempfile.TemporaryDirectory(prefix=f".{probe.video_id}-", dir=final_dir.parent) as tmp_name:
        tmp_dir = Path(tmp_name)
        filter_chain = f"fps={sample_fps:.12g},scale={target_width}:{target_height}"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(probe.path),
            "-map", "0:v:0",
            "-vf", filter_chain,
            "-q:v", str(ffmpeg_quality),
            "-start_number", "0",
            str(tmp_dir / _IMAGE_PATTERN),
        ]
        subprocess.run(command, check=True)
        images = sorted(tmp_dir.glob("*.jpg"))
        if not images:
            raise RuntimeError(f"No frames were extracted from {probe.path}")
        expected_names = [f"{index:06d}.jpg" for index in range(len(images))]
        if [path.name for path in images] != expected_names:
            raise RuntimeError(f"Unexpected frame filenames for {probe.video_id}")

        backup = final_dir.with_name(f".{final_dir.name}.old-{os.getpid()}")
        if backup.exists():
            shutil.rmtree(backup)
        if final_dir.exists():
            os.replace(final_dir, backup)
        try:
            os.replace(tmp_dir, final_dir)
        except Exception:
            if backup.exists() and not final_dir.exists():
                os.replace(backup, final_dir)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)
        return len(images)


def _records_for_video(
    paths: VBSDataPaths,
    probe: _VideoProbe,
    config: FrameBuildConfig,
    frame_count: int,
) -> list[FrameArtifact]:
    target_width, target_height = _target_size(probe.width, probe.height, config.long_edge)
    submission_fps = math.ceil(probe.fps)
    records: list[FrameArtifact] = []
    for sample_index in range(frame_count):
        timestamp_ms = sample_index * config.sample_period_ms
        frame_idx = math.floor(submission_fps * timestamp_ms / 1_000)
        image_path = (Path("frames") / probe.video_id / f"{sample_index:06d}.jpg").as_posix()
        records.append(FrameArtifact(
            frame_id=f"{probe.video_id}_{sample_index:06d}",
            video_id=probe.video_id,
            frame_idx=frame_idx,
            keyframe_order=sample_index + 1,
            timestamp_ms=timestamp_ms,
            fps=probe.fps,
            image_path=image_path,
            thumbnail_path=None,
            width=target_width,
            height=target_height,
            is_anchor=True,
            selection_reasons=("periodic",),
        ))
    return records


def _write_parquet_atomic(path: Path, rows: Iterable[FrameArtifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    table = pd.DataFrame(
        [row.model_dump(mode="python") for row in rows],
        columns=list(FrameArtifact.model_fields),
    )
    try:
        try:
            table.to_parquet(temporary, index=False)
        except ImportError as error:
            raise RuntimeError("Parquet support is required; install pyarrow") from error
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def build_frames(
    data_root: str | Path = "data",
    *,
    videos_root: str | Path | None = None,
    config: FrameBuildConfig | None = None,
    video_ids: set[str] | None = None,
    limit: int | None = None,
) -> FrameBuildReport:
    """Build or resume the canonical local VBS frame store."""

    cfg = config or FrameBuildConfig()
    paths = VBSDataPaths.from_roots(data_root, videos_root)
    paths.ensure()
    videos = discover_videos(paths.videos)
    if video_ids is not None:
        videos = [path for path in videos if path.stem in video_ids]
        missing = sorted(video_ids - {path.stem for path in videos})
        if missing:
            raise FileNotFoundError("Unknown video IDs: " + ", ".join(missing))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        videos = videos[:limit]
    if not videos:
        raise ValueError("No videos selected")

    records: list[FrameArtifact] = []
    reused = 0
    per_video: dict[str, int] = {}
    for video in videos:
        probe = probe_video(video)
        state = _load_state(paths, probe, cfg)
        if state is not None:
            frame_count = int(state["frame_count"])
            reused += 1
        else:
            frame_count = _extract_video(paths, probe, cfg)
            _write_json_atomic(
                _state_path(paths, probe.video_id),
                {
                    "source": _source_identity(probe.path, cfg),
                    "video_id": probe.video_id,
                    "frame_count": frame_count,
                    "avg_fps": probe.fps,
                    "duration_ms": probe.duration_ms,
                },
            )
        video_records = _records_for_video(paths, probe, cfg, frame_count)
        records.extend(video_records)
        per_video[probe.video_id] = frame_count

    frame_ids = [record.frame_id for record in records]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Generated duplicate frame_id values")

    _write_parquet_atomic(paths.frame_table, records)
    manifest = {
        "pipeline_version": "vbs-local-frames-v1",
        "source": "v3c_local_video",
        "videos_root": str(paths.videos),
        "frame_store_id": cfg.frame_store_id,
        "video_count": len(videos),
        "frame_count": len(records),
        "sample_period_ms": cfg.sample_period_ms,
        "image_long_edge": cfg.long_edge,
        "frame_id_format": "{video_id}_{sample_index:06d}",
        "image_path_format": "frames/{video_id}/{sample_index:06d}.jpg",
        "submission_coordinate_formula": "floor(ceil(avg_fps) * timestamp_ms / 1000)",
        "frame_id_digest": _digest(frame_ids),
        "per_video_frame_counts": per_video,
    }
    _write_json_atomic(paths.frame_manifest, manifest)
    return FrameBuildReport(
        video_count=len(videos),
        frame_count=len(records),
        reused_video_count=reused,
        frames_path=paths.frame_table,
        manifest_path=paths.frame_manifest,
    )
