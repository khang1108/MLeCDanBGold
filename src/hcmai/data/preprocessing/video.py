"""Video discovery, canonical timing, and sequential analysis decode."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from hcmai.data.preprocessing.config import PreprocessingConfig

ANALYSIS_SIZE = (320, 180)
TRANSNET_SIZE = (48, 27)
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


@dataclass(slots=True)
class FrameMeta:
    """Small per-frame record kept after the analysis image is released."""

    video_id: str
    decode_index: int
    frame_idx: int
    pts: int
    time_base: str
    timestamp_ms: int
    width: int
    height: int
    motion_score: float = 0.0


@dataclass(slots=True)
class VideoAnalysis:
    """Metadata and boundary scores produced by one analysis decode."""

    frames: list[FrameMeta]
    shot_frames: np.ndarray
    event_scores: np.ndarray


def discover_videos(
    config: PreprocessingConfig, limit: int | None = None,
) -> list[Path]:
    """Find supported videos recursively in deterministic order."""

    if config.videos_root is None:
        raise ValueError("local video discovery requires videos_root")
    root = config.videos_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Video root does not exist: {root}")
    paths = sorted(
        path for path in root.rglob("*")
        if path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if limit is not None:
        paths = paths[:limit]
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Video IDs must be unique across the corpus")
    if not paths:
        raise FileNotFoundError(f"No supported videos found in {root}")
    return paths


def peak_indices(scores: np.ndarray, threshold: float) -> set[int]:
    """Return one deterministic peak for each above-threshold score run."""

    return {
        index for index, score in enumerate(scores)
        if score >= threshold
        and (index == 0 or score > scores[index - 1])
        and (index == len(scores) - 1 or score >= scores[index + 1])
    }


def add_dynamic_coverage(
    frames: list[FrameMeta],
    reasons: list[set[str]],
    protected: set[int],
    config: PreprocessingConfig,
) -> None:
    """Add protected anchors before a motion-dependent gap is exceeded."""

    last = 0
    for index, frame in enumerate(frames[1:], start=1):
        ratio = min(frame.motion_score / max(config.motion_threshold, 1e-12), 1.0)
        gap = round(config.maximum_gap_ms - ratio * (
            config.maximum_gap_ms - config.minimum_gap_ms
        ))
        if frame.timestamp_ms - frames[last].timestamp_ms >= gap:
            anchor = index - 1 if index - last > 1 else index
            reasons[anchor].add("coverage_anchor")
            protected.add(anchor)
            last = anchor
        if reasons[index]:
            last = index


def iter_source_frames(path: Path) -> Iterator[tuple[FrameMeta, Any]]:
    """Yield canonical metadata and source frames in presentation order."""

    import av  # type: ignore[import-not-found]

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        if rate is None:
            raise ValueError(f"Average FPS is unavailable: {path}")
        fps = Fraction(rate.numerator, rate.denominator)
        start_position: Fraction | None = None
        for decode_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Frame PTS is unavailable: {path}")
            base = Fraction(frame.time_base.numerator, frame.time_base.denominator)
            position = frame.pts * base
            start_position = position if start_position is None else start_position
            timestamp_ms = round((position - start_position) * 1_000)
            yield FrameMeta(
                video_id=path.stem,
                decode_index=decode_index,
                frame_idx=round(Fraction(timestamp_ms, 1_000) * fps),
                pts=int(frame.pts),
                time_base=f"{base.numerator}/{base.denominator}",
                timestamp_ms=timestamp_ms,
                width=int(frame.width),
                height=int(frame.height),
            ), frame


def _camera_compensated_motion(
    previous: np.ndarray,
    current: np.ndarray,
    cv2: Any,
    flow_model: Any,
) -> float:
    """Measure dense residual flow after removing global camera motion."""

    prev_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    flow = flow_model.calc(prev_gray, curr_gray, None)
    points = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 5)
    if points is not None:
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points, None
        )
        source = points[status.ravel() == 1]
        target = tracked[status.ravel() == 1]
        matrix, _ = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC
        ) if len(source) >= 3 else (None, None)
        if matrix is not None:
            y, x = np.indices(prev_gray.shape, dtype=np.float32)
            flow[..., 0] -= matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2] - x
            flow[..., 1] -= matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2] - y
    diagonal = np.hypot(*prev_gray.shape)
    return float(np.linalg.norm(flow, axis=2).mean() / diagonal)


def _motion_score(
    previous: np.ndarray | None,
    current: np.ndarray,
    tools: tuple[Any, Any] | None,
) -> float:
    """Return optical-flow motion, with frame difference as CPU fallback."""

    if previous is None:
        return 0.0
    if tools is None:
        difference = np.abs(current.astype(np.float32) - previous)
        return float(difference.mean() / 255.0)
    return _camera_compensated_motion(previous, current, *tools)


def analyze_video(
    path: Path, config: PreprocessingConfig, event_detector: Any,
) -> VideoAnalysis:
    """Decode once for motion, TransNet, and streamed event detection."""

    try:
        import cv2  # type: ignore[import-not-found]

        tools: tuple[Any, Any] | None = (
            cv2,
            cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST),
        )
    except ModuleNotFoundError:
        tools = None
    records: list[FrameMeta] = []
    shot_frames: list[np.ndarray] = []
    previous: np.ndarray | None = None
    event_detector.start()
    for record, frame in iter_source_frames(path):
        rgb = frame.reformat(
            width=ANALYSIS_SIZE[0], height=ANALYSIS_SIZE[1], format="rgb24"
        ).to_ndarray()
        record.motion_score = _motion_score(previous, rgb, tools)
        records.append(record)
        shot_frames.append(
            np.asarray(Image.fromarray(rgb).resize(TRANSNET_SIZE), dtype=np.uint8)
        )
        event_detector.update(record, frame)
        previous = rgb
    if not records:
        raise ValueError(f"Video has no decodable frames: {path}")
    return VideoAnalysis(
        records,
        np.asarray(shot_frames, dtype=np.uint8),
        event_detector.scores(len(records)),
    )
