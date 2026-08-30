"""Xuất bản và lưu trữ Transcript.

Đảm nhiệm việc định dạng lại kết quả lời thoại đã nhận diện và lưu trữ chúng cố định.

Các tính năng chính:
1. Tạo file Artifact: Ghi kết quả ASR và Diarization ra file JSON/Parquet chuẩn.
2. Export S3/Disk: Đẩy dữ liệu transcript lên hệ thống lưu trữ lâu dài.
3. Đồng bộ Catalog: Cập nhật hệ thống dữ liệu rằng Transcript cho video này đã sẵn sàng sử dụng."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

Replace = Callable[[str | bytes | os.PathLike[str] | os.PathLike[bytes], str | bytes | os.PathLike[str] | os.PathLike[bytes]], None]


def staging_path(target: Path) -> Path:
    """Return a deterministic sibling staging path for one target."""

    return target.with_name(f".{target.name}.staging")


def publish_staged(
    files: Mapping[Path, Path],
    *,
    replace: Replace = os.replace,
) -> None:
    """Promote validated staged files and roll back every previous target on failure."""

    if not files or any(not staged.is_file() for staged in files.values()):
        raise ValueError("every publication target requires a staged file")
    backups = {
        target: target.with_name(f".{target.name}.backup")
        for target in files
        if target.exists()
    }
    promoted: list[Path] = []
    try:
        for target, backup in backups.items():
            backup.unlink(missing_ok=True)
            replace(target, backup)
        for target, staged in files.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            replace(staged, target)
            promoted.append(target)
    except Exception:
        for target in promoted:
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                replace(backup, target)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        for staged in files.values():
            staged.unlink(missing_ok=True)
