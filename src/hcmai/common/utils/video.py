"""Video identifier formatting and timing utilities."""

from __future__ import annotations

from numbers import Integral
from pathlib import PurePosixPath
from typing import Any

_SKIP_PARTS = frozenset({
    "data",
    "raw",
    "videos",
    "keyframes",
    "video",
    "keyframe",
    "frames",
    "features",
    "images",
    "map-keyframes",
    "map_keyframes",
})


def _clean_folder1(name: str) -> str:
    """Strip Videos_ or Keyframes_ prefix if present."""
    if name.startswith("Videos_"):
        return name[len("Videos_"):]
    if name.startswith("Keyframes_"):
        return name[len("Keyframes_"):]
    return name


def _is_video_hierarchy_path(path_str: str) -> bool:
    """Check if path looks like Videos_... or Keyframes_... or contains /videos/ or /keyframes/."""
    p = path_str.replace("\\", "/")
    return (
        "Videos_" in p
        or "Keyframes_" in p
        or "/videos/" in p
        or "/keyframes/" in p
        or "/video/" in p
        or "/keyframe/" in p
    )


def format_video_id(video_id_or_path: str, fallback_path: str | None = None) -> str:
    """Format a video identifier or path to <folder1>.<folder2>.<id>.

    Examples:
        >>> format_video_id("Videos_L26_b/videos/L26_V196/001.mp4")
        'L26_b.L26_V196.001'
        >>> format_video_id("Keyframes_L26_b/keyframes/L26_V196/001/0001.jpg")
        'L26_b.L26_V196.001'
        >>> format_video_id("Videos_L26_b.L26_V196.001")
        'L26_b.L26_V196.001'
        >>> format_video_id("L26_b.L26_V196.001")
        'L26_b.L26_V196.001'
    """
    candidate = (video_id_or_path or "").strip()
    if not candidate and fallback_path:
        candidate = fallback_path.strip()

    # 1. If candidate is already 3 parts dot-separated (e.g. L26_b.L26_V196.001 or Videos_L26_b.L26_V196.001)
    if candidate.count(".") == 2 and "/" not in candidate and "\\" not in candidate:
        p1, p2, p3 = candidate.split(".")
        return f"{_clean_folder1(p1)}.{p2}.{p3}"

    # 2. If candidate contains path separators (/ or \)
    if "/" in candidate or "\\" in candidate:
        normalized = candidate.replace("\\", "/")
        parts = [p for p in normalized.split("/") if p and p not in _SKIP_PARTS]
        start_idx = next(
            (
                i
                for i, part in enumerate(parts)
                if part.startswith(("Videos_", "Keyframes_"))
            ),
            0 if len(parts) >= 3 else None,
        )
        if start_idx is not None and len(parts) - start_idx >= 3:
            folder1 = _clean_folder1(parts[start_idx])
            folder2 = parts[start_idx + 1]
            raw_id = parts[start_idx + 2]
            id_stem = PurePosixPath(raw_id).stem
            return f"{folder1}.{folder2}.{id_stem}"

    # 3. If candidate starts with Videos_ or Keyframes_ and is dot-separated with extension
    dot_parts = candidate.split(".")
    if (
        len(dot_parts) == 4
        and dot_parts[-1].lower() in {
            "mp4", "mkv", "avi", "mov", "webm", "jpg", "jpeg", "png", "webp", "csv"
        }
    ):
        return f"{_clean_folder1(dot_parts[0])}.{dot_parts[1]}.{dot_parts[2]}"

    # 4. If fallback_path is provided and matches video hierarchy (e.g. Videos_L26_b/videos/L26_V196/001.mp4)
    if fallback_path and _is_video_hierarchy_path(fallback_path):
        normalized_fb = fallback_path.replace("\\", "/")
        parts = [p for p in normalized_fb.split("/") if p and p not in _SKIP_PARTS]
        start_idx = next(
            (
                i
                for i, part in enumerate(parts)
                if part.startswith(("Videos_", "Keyframes_"))
            ),
            None,
        )
        if start_idx is not None and len(parts) - start_idx >= 3:
            folder1 = _clean_folder1(parts[start_idx])
            folder2 = parts[start_idx + 1]
            raw_id = parts[start_idx + 2]
            id_stem = PurePosixPath(raw_id).stem
            return f"{folder1}.{folder2}.{id_stem}"

    return _clean_folder1(candidate)


def derive_fps(frame: Any, default_fps: float = 25.0) -> float:
    """Extract or estimate video FPS from frame metadata."""
    if frame is None:
        return default_fps

    fps = getattr(frame, "fps", None)
    if fps is not None and isinstance(fps, (int, float)) and fps > 0:
        return float(fps)

    return default_fps


def official_frame_idx(frame: Any) -> int:
    """Return the exact BTC submission coordinate for a canonical frame.

    ``frame_idx`` is loaded from the organizer's BTC keyframe mapping. It is
    deliberately not derived from ``timestamp_ms`` and FPS: multiple internal
    frames may share one official coordinate, while ``frame_id`` and the
    timestamp still identify the exact frame used for retrieval and display.
    """

    value = getattr(frame, "frame_idx", None)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("canonical frame is missing a valid BTC frame_idx")
    return int(value)
