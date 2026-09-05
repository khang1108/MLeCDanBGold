"""Tests for real-byte disk measurement and write-admission guardrails.

Covers one-byte disk-budget boundaries and same-filesystem hard-link byte
accounting for the active working set.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from offline.ingestion.custom_pipeline.config import DiskBudgetConfig
from offline.ingestion.custom_pipeline.disk import (
    DiskAdmissionError,
    measure_tree_bytes,
    require_write_capacity,
)


def _sparse_file(path: Path, size: int) -> None:
    """Report ``size`` through st_size without allocating the bytes."""

    with open(path, "wb") as handle:
        handle.truncate(size)


# ---------------------------------------------------------------------------
# measure_tree_bytes: hard-link accounting
# ---------------------------------------------------------------------------


def test_measure_tree_bytes_counts_a_hard_link_once(tmp_path: Path) -> None:
    original = tmp_path / "frame.jpg"
    original.write_bytes(b"x" * 1000)
    linked = tmp_path / "linked.jpg"
    os.link(original, linked)

    assert measure_tree_bytes(tmp_path) == 1000


def test_measure_tree_bytes_ignores_symlinks(tmp_path: Path) -> None:
    original = tmp_path / "frame.jpg"
    original.write_bytes(b"x" * 500)
    symlink = tmp_path / "symlink.jpg"
    symlink.symlink_to(original)

    assert measure_tree_bytes(tmp_path) == 500


def test_measure_tree_bytes_sums_distinct_files(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"a" * 100)
    (tmp_path / "b.jpg").write_bytes(b"b" * 250)

    assert measure_tree_bytes(tmp_path) == 350


def test_measure_tree_bytes_returns_zero_for_missing_path(tmp_path: Path) -> None:
    assert measure_tree_bytes(tmp_path / "does-not-exist") == 0


# ---------------------------------------------------------------------------
# require_write_capacity: one-byte boundaries
# ---------------------------------------------------------------------------


def _patch_disk_usage(monkeypatch: pytest.MonkeyPatch, free_bytes: int) -> None:
    import offline.ingestion.custom_pipeline.disk as disk_module

    monkeypatch.setattr(
        disk_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=0, used=0, free=free_bytes),
    )


def test_write_capacity_accepts_exact_free_reserve_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = DiskBudgetConfig(min_free_gib=1, max_active_gib=100)
    estimated_bytes = 10
    # free_after == min_free_bytes exactly must be accepted.
    _patch_disk_usage(monkeypatch, budget.min_free_bytes + estimated_bytes)

    require_write_capacity(budget, tmp_path, tmp_path, estimated_bytes, operation="test")


def test_write_capacity_rejects_one_byte_below_free_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = DiskBudgetConfig(min_free_gib=1, max_active_gib=100)
    estimated_bytes = 10
    # free_after == min_free_bytes - 1 must be rejected.
    _patch_disk_usage(monkeypatch, budget.min_free_bytes + estimated_bytes - 1)

    with pytest.raises(DiskAdmissionError, match="below min_free_bytes reserve"):
        require_write_capacity(budget, tmp_path, tmp_path, estimated_bytes, operation="test")


def test_write_capacity_accepts_exact_active_cap_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = DiskBudgetConfig(min_free_gib=1, max_active_gib=1)
    active_root = tmp_path / "active"
    active_root.mkdir()
    estimated_bytes = 10
    remaining_capacity = budget.max_active_bytes - estimated_bytes
    _sparse_file(active_root / "existing.bin", remaining_capacity)
    _patch_disk_usage(monkeypatch, budget.min_free_bytes + 1_000_000)

    require_write_capacity(budget, tmp_path, active_root, estimated_bytes, operation="test")


def test_write_capacity_rejects_one_byte_over_active_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = DiskBudgetConfig(min_free_gib=1, max_active_gib=1)
    active_root = tmp_path / "active"
    active_root.mkdir()
    estimated_bytes = 10
    remaining_capacity = budget.max_active_bytes - estimated_bytes + 1
    _sparse_file(active_root / "existing.bin", remaining_capacity)
    _patch_disk_usage(monkeypatch, budget.min_free_bytes + 1_000_000)

    with pytest.raises(DiskAdmissionError, match="exceeds max_active_bytes cap"):
        require_write_capacity(budget, tmp_path, active_root, estimated_bytes, operation="test")


def test_write_capacity_rejects_negative_estimate(tmp_path: Path) -> None:
    budget = DiskBudgetConfig()
    with pytest.raises(ValueError, match="must not be negative"):
        require_write_capacity(budget, tmp_path, tmp_path, -1, operation="test")
