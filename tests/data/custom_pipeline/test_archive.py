"""Tests for one-archive download, safety inspection, extraction, and grouping.

Covers unsafe ZIP rejection (traversal, absolute paths, symlinks, wrong
naming, duplicates, declared-size abuse), resumed curl argv, actual-size
mismatch, ZIP-deletion timing, canonical batch grouping, and hard-link-only
staging.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from offline.ingestion.custom_pipeline.archive import (
    ArchiveInventory,
    ArchiveMember,
    ArchiveSafetyError,
    download_archive,
    extract_archive_atomically,
    inspect_archive,
    plan_archive_batches,
    stage_archive_source_links,
)
from offline.ingestion.custom_pipeline.config import DiskBudgetConfig
from offline.ingestion.custom_pipeline.disk import DiskAdmissionError


def _write_zip(path: Path, members: dict[str, bytes], *, symlink: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        if symlink is not None:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (0o120777 << 16)
            archive.writestr(info, "target.mp4")


def _generous_budget() -> DiskBudgetConfig:
    return DiskBudgetConfig(
        min_free_gib=0.000001,
        max_active_gib=1,
        max_archive_download_gib=1,
        max_archive_uncompressed_gib=1,
    )


# ---------------------------------------------------------------------------
# download_archive
# ---------------------------------------------------------------------------


def test_download_archive_uses_resumable_curl_argv_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "L01.zip"
    calls: list[dict[str, object]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        # Simulate curl writing the .part file so the byte-ceiling check passes.
        part_path = Path(argv[argv.index("-o") + 1])
        part_path.write_bytes(b"0" * 10)
        calls.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = download_archive(
        "https://example.org/L01.zip",
        destination,
        budget=_generous_budget(),
        run_root=tmp_path,
        active_root=tmp_path,
    )

    assert result == destination
    assert destination.read_bytes() == b"0" * 10
    assert calls[0]["shell"] is False
    assert calls[0]["check"] is True
    argv = calls[0]["argv"]
    assert argv[0] == "curl"
    assert "-C" in argv and argv[argv.index("-C") + 1] == "-"
    assert "--retry" in argv


def test_download_archive_rejects_oversized_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "L01.zip"
    budget = DiskBudgetConfig(
        min_free_gib=0.000001,
        max_active_gib=1,
        max_archive_download_gib=0.000001,  # ~1073 bytes ceiling
        max_archive_uncompressed_gib=1,
    )

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        part_path = Path(argv[argv.index("-o") + 1])
        part_path.write_bytes(b"0" * 10_000)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exceeds max_archive_download_bytes"):
        download_archive(
            "https://example.org/L01.zip",
            destination,
            budget=budget,
            run_root=tmp_path,
            active_root=tmp_path,
        )
    assert not destination.exists()


def test_download_archive_refuses_when_disk_reserve_is_insufficient(tmp_path: Path) -> None:
    budget = DiskBudgetConfig(
        min_free_gib=999_999,  # impossible to satisfy
        max_active_gib=1,
        max_archive_download_gib=1,
        max_archive_uncompressed_gib=1,
    )
    with pytest.raises(DiskAdmissionError):
        download_archive(
            "https://example.org/L01.zip",
            tmp_path / "L01.zip",
            budget=budget,
            run_root=tmp_path,
            active_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# inspect_archive: safety rejections
# ---------------------------------------------------------------------------


def test_inspect_archive_accepts_valid_nested_members(tmp_path: Path) -> None:
    zip_path = tmp_path / "Videos_L01.zip"
    _write_zip(
        zip_path,
        {"L01/L01_V001.mp4": b"a" * 100, "L01/L01_V002.mp4": b"b" * 200},
    )
    inventory = inspect_archive(zip_path, budget=_generous_budget())
    assert inventory.video_ids == ("L01_V001", "L01_V002")


def test_inspect_archive_rejects_absolute_path(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    _write_zip(zip_path, {"/etc/L01/L01_V001.mp4": b"a"})
    with pytest.raises(ArchiveSafetyError, match="absolute path"):
        inspect_archive(zip_path, budget=_generous_budget())


def test_inspect_archive_rejects_path_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    _write_zip(zip_path, {"../L01/L01_V001.mp4": b"a"})
    with pytest.raises(ArchiveSafetyError, match="traversal"):
        inspect_archive(zip_path, budget=_generous_budget())


def test_inspect_archive_rejects_symlink_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    _write_zip(
        zip_path,
        {"L01/L01_V001.mp4": b"a"},
        symlink="L01/L01_V002.mp4",
    )
    with pytest.raises(ArchiveSafetyError, match="symlink"):
        inspect_archive(zip_path, budget=_generous_budget())


def test_inspect_archive_rejects_non_matching_member_name(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    _write_zip(zip_path, {"L01/readme.txt": b"hello"})
    with pytest.raises(ArchiveSafetyError, match="must match"):
        inspect_archive(zip_path, budget=_generous_budget())


def test_inspect_archive_rejects_flat_non_nested_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    _write_zip(zip_path, {"L01_V001.mp4": b"a"})
    with pytest.raises(ArchiveSafetyError, match="must match"):
        inspect_archive(zip_path, budget=_generous_budget())


def test_inspect_archive_rejects_duplicate_video_id(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("L01/L01_V001.mp4", b"a")
        archive.writestr("L02/L01_V001.mp4", b"b")
    with pytest.raises(ArchiveSafetyError, match="duplicate video_id"):
        inspect_archive(zip_path, budget=_generous_budget())


def test_inspect_archive_rejects_declared_size_abuse(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    _write_zip(zip_path, {"L01/L01_V001.mp4": b"a" * 1000})
    tiny_budget = DiskBudgetConfig(
        min_free_gib=0.000001,
        max_active_gib=1,
        max_archive_download_gib=1,
        max_archive_uncompressed_gib=0.0000001,  # ~107 bytes
    )
    with pytest.raises(ArchiveSafetyError, match="max_archive_uncompressed_bytes"):
        inspect_archive(zip_path, budget=tiny_budget)


def test_inspect_archive_rejects_empty_archive(tmp_path: Path) -> None:
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w"):
        pass
    with pytest.raises(ArchiveSafetyError, match="no valid"):
        inspect_archive(zip_path, budget=_generous_budget())


# ---------------------------------------------------------------------------
# extract_archive_atomically
# ---------------------------------------------------------------------------


def test_extract_archive_atomically_commits_and_removes_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "Videos_L01.zip"
    _write_zip(zip_path, {"L01/L01_V001.mp4": b"a" * 50})
    inventory = inspect_archive(zip_path, budget=_generous_budget())
    output_dir = tmp_path / "extracted" / "L01"

    result = extract_archive_atomically(
        zip_path,
        inventory,
        output_dir,
        budget=_generous_budget(),
        run_root=tmp_path,
        active_root=tmp_path,
    )

    assert result == output_dir
    assert (output_dir / "L01" / "L01_V001.mp4").read_bytes() == b"a" * 50
    assert (output_dir / "archive_manifest.json").is_file()
    assert not zip_path.exists()  # zip removed only after manifest committed


def test_extract_archive_atomically_rejects_actual_size_mismatch(tmp_path: Path) -> None:
    zip_path = tmp_path / "Videos_L01.zip"
    _write_zip(zip_path, {"L01/L01_V001.mp4": b"a" * 50})
    real_inventory = inspect_archive(zip_path, budget=_generous_budget())
    tampered_inventory = ArchiveInventory(
        archive_id=real_inventory.archive_id,
        members=(
            ArchiveMember(
                video_id="L01_V001",
                member_name="L01/L01_V001.mp4",
                declared_size=999,  # disagrees with the actual 50-byte payload
            ),
        ),
    )
    output_dir = tmp_path / "extracted" / "L01"

    with pytest.raises(ArchiveSafetyError, match="extracted size mismatch"):
        extract_archive_atomically(
            zip_path,
            tampered_inventory,
            output_dir,
            budget=_generous_budget(),
            run_root=tmp_path,
            active_root=tmp_path,
        )
    # A failed extraction must preserve the ZIP for resume/inspection.
    assert zip_path.exists()


# ---------------------------------------------------------------------------
# plan_archive_batches
# ---------------------------------------------------------------------------


def _inventory_with(video_ids: list[str]) -> ArchiveInventory:
    return ArchiveInventory(
        archive_id="L01",
        members=tuple(
            ArchiveMember(video_id=vid, member_name=f"L01/{vid}.mp4", declared_size=1)
            for vid in video_ids
        ),
    )


def test_plan_archive_batches_groups_of_eight_plus_remainder() -> None:
    video_ids = [f"L01_V{i:03d}" for i in range(1, 20)]  # 19 videos
    inventory = _inventory_with(video_ids)
    groups = plan_archive_batches(inventory, batch_size=8)
    assert [len(group) for group in groups] == [8, 8, 3]
    assert [vid for group in groups for vid in group] == video_ids


def test_plan_archive_batches_rejects_non_positive_batch_size() -> None:
    inventory = _inventory_with(["L01_V001"])
    with pytest.raises(ValueError, match="positive"):
        plan_archive_batches(inventory, batch_size=0)


# ---------------------------------------------------------------------------
# stage_archive_source_links
# ---------------------------------------------------------------------------


def test_stage_archive_source_links_uses_hard_links_only(tmp_path: Path) -> None:
    extracted_dir = tmp_path / "extracted"
    (extracted_dir / "L01").mkdir(parents=True)
    source_file = extracted_dir / "L01" / "L01_V001.mp4"
    source_file.write_bytes(b"x" * 100)
    source_root = tmp_path / "native_source"

    linked = stage_archive_source_links(extracted_dir, ["L01_V001"], source_root)

    assert linked == (source_root / "L01_V001.mp4",)
    linked_stat = linked[0].stat()
    source_stat = source_file.stat()
    assert linked_stat.st_ino == source_stat.st_ino  # same inode: a hard link


def test_stage_archive_source_links_rejects_missing_video(tmp_path: Path) -> None:
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    with pytest.raises(ArchiveSafetyError, match="missing requested video"):
        stage_archive_source_links(extracted_dir, ["L01_V999"], tmp_path / "source")


def test_inspect_archive_accepts_a_non_line_number_directory(tmp_path: Path) -> None:
    zip_path = tmp_path / "Videos_L21_a.zip"
    _write_zip(zip_path, {"video/L21_V001.mp4": b"a" * 10})
    inventory = inspect_archive(zip_path, budget=_generous_budget())
    assert inventory.video_ids == ("L21_V001",)
