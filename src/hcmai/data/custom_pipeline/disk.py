"""Real-byte disk measurement and write-admission checks.

The local pipeline must remain within a measured free-byte reserve and an
active-working-set cap (see ``DiskBudgetConfig``). This module measures
unique file inodes on disk so same-filesystem hard links are never
double-counted, and enforces both guardrails before any material batch write.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from hcmai.common.utils.logging import get_logger
from hcmai.data.custom_pipeline.config import DiskBudgetConfig

logger = get_logger(__name__)


class DiskAdmissionError(RuntimeError):
    """Raised when a planned write would breach a disk budget guardrail."""


def measure_tree_bytes(path: str | Path) -> int:
    """Return the total bytes of unique regular-file inodes under ``path``.

    Symlinks are ignored and same-filesystem hard links are counted once, so
    staged native links never inflate the measured active working set.
    """

    root = Path(path)
    if not root.exists():
        return 0

    seen: set[tuple[int, int]] = set()
    total = 0
    for entry in root.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        stat = entry.stat()
        key = (stat.st_dev, stat.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += stat.st_size
    return total


@dataclass(frozen=True)
class DiskSnapshot:
    """Real free and active bytes measured at one point in time."""

    free_bytes: int
    active_bytes: int


def snapshot_disk(run_root: str | Path, active_root: str | Path) -> DiskSnapshot:
    """Measure real free space at ``run_root`` and the active working set size."""

    usage = shutil.disk_usage(Path(run_root))
    active_bytes = measure_tree_bytes(active_root)
    logger.info(
        "disk snapshot: free=%d bytes active=%d bytes (run_root=%s, active_root=%s)",
        usage.free,
        active_bytes,
        run_root,
        active_root,
    )
    return DiskSnapshot(free_bytes=usage.free, active_bytes=active_bytes)


def require_write_capacity(
    budget: DiskBudgetConfig,
    run_root: str | Path,
    active_root: str | Path,
    estimated_bytes: int,
    *,
    operation: str,
) -> DiskSnapshot:
    """Enforce the disk admission invariant before one material write.

    The invariant is: free bytes after the write must stay at or above
    ``budget.min_free_bytes``, and the active working set after the write
    must stay at or below ``budget.max_active_bytes``.

    Raises:
        ValueError: If ``estimated_bytes`` is negative.
        DiskAdmissionError: If either guardrail would be breached, with
            per-field diagnostics for the caller to log and surface.
    """

    if estimated_bytes < 0:
        raise ValueError("estimated_bytes must not be negative")

    snapshot = snapshot_disk(run_root, active_root)
    free_after = snapshot.free_bytes - estimated_bytes
    active_after = snapshot.active_bytes + estimated_bytes

    if free_after < budget.min_free_bytes:
        raise DiskAdmissionError(
            _diagnostic(
                operation,
                "free_bytes_after_write below min_free_bytes reserve",
                snapshot,
                estimated_bytes,
                budget,
                free_after=free_after,
                active_after=active_after,
            )
        )
    if active_after > budget.max_active_bytes:
        raise DiskAdmissionError(
            _diagnostic(
                operation,
                "active_bytes_after_write exceeds max_active_bytes cap",
                snapshot,
                estimated_bytes,
                budget,
                free_after=free_after,
                active_after=active_after,
            )
        )

    logger.info(
        "disk admission accepted for %s: estimated_bytes=%d free_after=%d active_after=%d",
        operation,
        estimated_bytes,
        free_after,
        active_after,
    )
    return snapshot


def _diagnostic(
    operation: str,
    reason: str,
    snapshot: DiskSnapshot,
    estimated_bytes: int,
    budget: DiskBudgetConfig,
    *,
    free_after: int,
    active_after: int,
) -> str:
    """Render one actionable, fully-parameterized disk admission diagnostic."""

    return (
        f"{reason} (operation={operation}, estimated_bytes={estimated_bytes}, "
        f"free_bytes={snapshot.free_bytes}, active_bytes={snapshot.active_bytes}, "
        f"free_after={free_after}, active_after={active_after}, "
        f"min_free_bytes={budget.min_free_bytes}, "
        f"max_active_bytes={budget.max_active_bytes})"
    )


__all__ = [
    "DiskAdmissionError",
    "DiskSnapshot",
    "measure_tree_bytes",
    "require_write_capacity",
    "snapshot_disk",
]
