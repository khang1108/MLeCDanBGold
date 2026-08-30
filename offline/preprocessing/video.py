"""Decode source videos and preserve canonical timing metadata.

This module owns lightweight video discovery, frame decoding, and motion
analysis helpers. It intentionally does not own frame selection, model
loading, artifact publication, or preprocessing configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
from PIL import Image

from offline.ingestion.s3 import VIDEO_EXTENSIONS

ANALYSIS_SIZE = (320, 180)
TRANSNET_SIZE = (48, 27)


class VideoProcessingConfig(Protocol):
    """Configuration fields required by the retained video helpers."""

    videos_root: Path | None
    motion_threshold: float
    minimum_gap_ms: int
    maximum_gap_ms: int


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
    config: VideoProcessingConfig, limit: int | None = None,
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
    config: VideoProcessingConfig,
) -> None:
    """Add protected anchors before a motion-dependent gap is exceeded."""

    last = 0
    for index, frame in enumerate(frames[1:], start=1):
        # Tính toán tỉ lệ của motion score so với threshold 
        # ratio là tỷ lệ motion giữa frame hiện tại và frame trước
        # Nếu motion lớn thì ratio sẽ gần 1, gap sẽ nhỏ
        # Nếu motion nhỏ thì ratio sẽ gần 0, gap sẽ lớn
        ratio = min(frame.motion_score / max(config.motion_threshold, 1e-12), 1.0)

        # Tính toán gap dựa trên ratio
        # gap = maximum_gap_ms - ratio * (maximum_gap_ms - minimum_gap_ms)
        # Nếu motion lớn (ratio = 1) thì gap = minimum_gap_ms
        # Nếu motion nhỏ (ratio = 0) thì gap = maximum_gap_ms
        # gap có ý nghĩa là khoảng thời gian tối đa giữa 2 frame liên tiếp để không làm giảm sự đa dạng của video
        gap = round(config.maximum_gap_ms - ratio * (
            config.maximum_gap_ms - config.minimum_gap_ms
        ))
        # Nếu gap lớn hơn threshold thì thêm protected anchor
        # protected anchor là các frame được giữ lại để đảm bảo coverage
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

    # Mở video và đọc các thông tin cần thiết
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate

        if rate is None:
            raise ValueError(f"Average FPS is unavailable: {path}")

        fps = Fraction(rate.numerator, rate.denominator)

        # Duyệt qua từng frame
        for decode_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Frame PTS is unavailable: {path}")

            # Tính toán timestamp_ms của frame
            base = Fraction(frame.time_base.numerator, frame.time_base.denominator)
            position = frame.pts * base
            timestamp_ms = round(position * 1_000)

            # Tạo FrameMeta và yield
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

    # Chuyển sang grayscale để tính toán motion
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)

    # Tính toán optical flow
    flow = flow_model.calc(prev_gray, curr_gray, None)
    
    # Tìm các điểm đặc biệt trong frame
    points = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 5)

    # Nếu tìm thấy điểm đặc biệt
    if points is not None:
        # Tính toán optical flow của các điểm đặc biệt
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points, None
        )

        # Lọc ra các điểm đặc biệt được theo dõi thành công
        source = points[status.ravel() == 1]
        target = tracked[status.ravel() == 1]

        # Tính toán ma trận affine transformation
        matrix, _ = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC
        ) if len(source) >= 3 else (None, None)

        # Nếu tìm thấy ma trận affine transformation
        if matrix is not None:
            y, x = np.indices(prev_gray.shape, dtype=np.float32)
            flow[..., 0] -= matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2] - x
            flow[..., 1] -= matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2] - y

    # Tính toán motion score
    diagonal = np.hypot(*prev_gray.shape)
    return float(np.linalg.norm(flow, axis=2).mean() / diagonal)


def _motion_score(
    previous: np.ndarray | None,
    current: np.ndarray,
    tools: tuple[Any, Any] | None,
) -> float:
    """Return optical-flow motion, with frame difference as CPU fallback."""

    # Nếu không có frame trước thì motion score là 0
    if previous is None:
        return 0.0
    # Nếu không có tools thì tính toán motion score bằng frame difference
    if tools is None:
        difference = np.abs(current.astype(np.float32) - previous)
        return float(difference.mean() / 255.0)
    # Tính toán motion score bằng optical flow
    return _camera_compensated_motion(previous, current, *tools)


def analyze_video(
    path: Path, config: VideoProcessingConfig, event_detector: Any,
) -> VideoAnalysis:
    """Decode once for motion, TransNet, and streamed event detection."""

    try:
        import cv2  # type: ignore[import-not-found]

        tools: tuple[Any, Any] | None = (
            cv2,
            cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST),
        )
    except ModuleNotFoundError:
        tools = None

    # Khởi tạo danh sách metadata và shot frames
    records: list[FrameMeta] = []
    shot_frames: list[np.ndarray] = []
    previous: np.ndarray | None = None
    event_detector.start()

    # Duyệt qua từng frame
    for record, frame in iter_source_frames(path):
        rgb = frame.reformat(
            width=ANALYSIS_SIZE[0], height=ANALYSIS_SIZE[1], format="rgb24"
        ).to_ndarray()

        # Tính toán motion score
        record.motion_score = _motion_score(previous, rgb, tools)
        records.append(record)
        shot_frames.append(
            np.asarray(Image.fromarray(rgb).resize(TRANSNET_SIZE), dtype=np.uint8)
        )

        # Cập nhật event detector
        event_detector.update(record, frame)
        previous = rgb
    
    if not records:
        raise ValueError(f"Video has no decodable frames: {path}")
    return VideoAnalysis(
        records,
        np.asarray(shot_frames, dtype=np.uint8),
        event_detector.scores(len(records)),
    )
