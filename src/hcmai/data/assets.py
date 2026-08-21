"""Quản lý đường dẫn Assets (Tài nguyên).

Đảm nhiệm việc phân giải và cung cấp đường dẫn chính xác (canonical paths) đến file ảnh và video.

Các tính năng chính:
1. Phân giải đường dẫn: Tạo đường dẫn tuyệt đối cho một Frame ID hoặc Video ID cụ thể.
2. Xác thực tồn tại (Validation): Kiểm tra xem ảnh/video đã được lưu trữ trên file system chưa.
3. Cấu trúc thư mục: Đảm bảo tuân thủ cấu trúc thư mục quy chuẩn (VD: video/frame_id.jpg) của project."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hcmai.common.schemas import FrameRecord


class FrameAssetError(OSError):
    """A canonical frame asset cannot be safely materialized."""


class FrameAssetMissingError(FrameAssetError, FileNotFoundError):
    """The configured canonical asset does not exist as a regular file."""


class FrameAssetOutsideRootError(FrameAssetError, PermissionError):
    """A canonical asset path escapes the configured dataset root."""


@dataclass(frozen=True, slots=True)
class FrameAssetStatus:
    """Deterministic sample-based availability used by health and diagnostics."""

    ready: bool
    checked: int
    available: int
    missing: int

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "ready": self.ready,
            "checked": self.checked,
            "available": self.available,
            "missing": self.missing,
        }


class FrameAssetResolver:
    """Resolve relative canonical paths without allowing dataset-root escape."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()

    def resolve_value(self, value: str | Path, *, require_file: bool = True) -> Path:
        path = Path(value).expanduser()
        resolved = path.resolve() if path.is_absolute() else (self.dataset_root / path).resolve()
        if not resolved.is_relative_to(self.dataset_root):
            raise FrameAssetOutsideRootError("frame asset escapes dataset root")
        if require_file and not resolved.is_file():
            raise FrameAssetMissingError("frame asset is not available")
        return resolved

    def resolve_frame(
        self,
        frame: FrameRecord,
        *,
        thumbnail: bool = False,
        require_file: bool = True,
    ) -> Path:
        value = frame.thumbnail_path if thumbnail else frame.image_path
        if value is None:
            value = frame.image_path
        return self.resolve_value(value, require_file=require_file)

    def sample_status(
        self,
        frames: Sequence[FrameRecord],
        *,
        sample_size: int = 100,
    ) -> FrameAssetStatus:
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        sample = frames[:sample_size]
        available = 0
        for frame in sample:
            try:
                self.resolve_frame(frame)
            except FrameAssetError:
                continue
            available += 1
        checked = len(sample)
        return FrameAssetStatus(
            ready=checked > 0 and available == checked,
            checked=checked,
            available=available,
            missing=checked - available,
        )
