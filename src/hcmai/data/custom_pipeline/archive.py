"""One-archive acquisition, safe extraction, inventory, and canonical grouping.

Downloads exactly one organizer ZIP via resumable curl, validates it as a safe
archive of nested ``Lxx/Lxx_Vnnn.mp4`` members, extracts it atomically within
the local disk budget, and partitions its videos into canonical groups of at
most eight. This module never invokes yt-dlp and never retains more than one
archive's ZIP/extraction at a time.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hcmai.common.utils.io import atomic_write, write_json
from hcmai.common.utils.logging import get_logger
from hcmai.data.custom_pipeline.config import DiskBudgetConfig
from hcmai.data.custom_pipeline.disk import require_write_capacity

logger = get_logger(__name__)

# Organizer archives nest each video under a line-number directory, e.g.
# "L21/L21_V001.mp4". The directory and filename line numbers need not match
# so a video misfiled under a neighboring directory is still caught later as
# a duplicate video_id rather than silently rejected as an unknown shape.
_MEMBER_PATTERN = re.compile(r"^L\d{2}/L\d{2}_V\d{3}\.mp4$")
_ZIP_SYMLINK_UNIX_MODE = 0o120000
_DEFAULT_MAX_RETRIES = 5


class ArchiveSafetyError(ValueError):
    """Raised when an archive member fails a path, link, or naming safety check."""


@dataclass(frozen=True)
class ArchiveMember:
    """One validated ``Lxx/Lxx_Vnnn.mp4`` member inside an organizer ZIP."""

    video_id: str
    member_name: str
    declared_size: int


@dataclass(frozen=True)
class ArchiveInventory:
    """The complete, ordered, safety-validated member list of one archive."""

    archive_id: str
    members: tuple[ArchiveMember, ...]

    @property
    def video_ids(self) -> tuple[str, ...]:
        """Return every member's video_id in the archive's canonical order."""

        return tuple(member.video_id for member in self.members)


def download_archive(
    url: str,
    destination: str | Path,
    *,
    budget: DiskBudgetConfig,
    run_root: str | Path,
    active_root: str | Path,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> Path:
    """Resumably download one archive ZIP into ``destination`` via curl.

    The download lands in a sibling ``.part`` file so an interrupted attempt
    can resume with ``curl -C -`` instead of restarting from zero.

    Raises:
        DiskAdmissionError: If the download would breach the disk budget.
        RuntimeError: If curl exits non-zero, or the downloaded size exceeds
            ``budget.max_archive_download_bytes``.
    """

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination_path.with_name(destination_path.name + ".part")

    require_write_capacity(
        budget,
        run_root,
        active_root,
        budget.max_archive_download_bytes,
        operation=f"download_archive:{destination_path.name}",
    )

    argv = [
        "curl",
        "-fL",
        "-C",
        "-",
        "--retry",
        str(max_retries),
        "--retry-delay",
        "5",
        "-o",
        str(part_path),
        url,
    ]
    logger.info("downloading archive %s -> %s", url, part_path)
    try:
        subprocess.run(argv, check=True, shell=False, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        diagnostic = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"archive download failed for {url}: {diagnostic}") from error

    downloaded_bytes = part_path.stat().st_size
    if downloaded_bytes > budget.max_archive_download_bytes:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded archive exceeds max_archive_download_bytes: "
            f"{downloaded_bytes} > {budget.max_archive_download_bytes}"
        )

    part_path.replace(destination_path)
    logger.info("downloaded archive %s (%d bytes)", destination_path.name, downloaded_bytes)
    return destination_path


def _require_safe_member_path(name: str) -> None:
    """Reject absolute paths and any ``..`` path-traversal component."""

    posix = PurePosixPath(name)
    if posix.is_absolute():
        raise ArchiveSafetyError(f"archive member must not use an absolute path: {name}")
    if ".." in posix.parts:
        raise ArchiveSafetyError(f"archive member must not use path traversal: {name}")


def _require_not_a_link(info: zipfile.ZipInfo) -> None:
    """Reject ZIP members whose stored Unix mode marks them as a symlink."""

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if (unix_mode & 0xF000) == _ZIP_SYMLINK_UNIX_MODE:
        raise ArchiveSafetyError(f"archive member must not be a symlink: {info.filename}")


def inspect_archive(zip_path: str | Path, *, budget: DiskBudgetConfig) -> ArchiveInventory:
    """Validate one downloaded ZIP and return its safe, ordered member inventory.

    Raises:
        ArchiveSafetyError: If any member uses an unsafe path, is a link, does
            not match ``Lxx/Lxx_Vnnn.mp4``, duplicates a member or video_id, or
            the archive's declared uncompressed size exceeds
            ``budget.max_archive_uncompressed_bytes``.
    """

    path = Path(zip_path)
    archive_id = path.stem
    seen_member_names: set[str] = set()
    seen_video_ids: set[str] = set()
    total_declared_size = 0
    members: list[ArchiveMember] = []

    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            _require_safe_member_path(info.filename)
            _require_not_a_link(info)
            if _MEMBER_PATTERN.match(info.filename) is None:
                raise ArchiveSafetyError(
                    f"unexpected archive member (must match Lxx/Lxx_Vnnn.mp4): {info.filename}"
                )
            if info.filename in seen_member_names:
                raise ArchiveSafetyError(f"duplicate archive member: {info.filename}")

            video_id = Path(info.filename).stem
            if video_id in seen_video_ids:
                raise ArchiveSafetyError(f"duplicate video_id across archive members: {video_id}")

            seen_member_names.add(info.filename)
            seen_video_ids.add(video_id)
            total_declared_size += info.file_size
            members.append(
                ArchiveMember(
                    video_id=video_id,
                    member_name=info.filename,
                    declared_size=info.file_size,
                )
            )

    if not members:
        raise ArchiveSafetyError(f"archive contains no valid Lxx/Lxx_Vnnn.mp4 members: {path}")
    if total_declared_size > budget.max_archive_uncompressed_bytes:
        raise ArchiveSafetyError(
            f"declared uncompressed archive size {total_declared_size} exceeds "
            f"max_archive_uncompressed_bytes {budget.max_archive_uncompressed_bytes}"
        )

    ordered = tuple(sorted(members, key=lambda member: member.video_id))
    logger.info("inspected archive %s: %d valid member(s)", archive_id, len(ordered))
    return ArchiveInventory(archive_id=archive_id, members=ordered)


def extract_archive_atomically(
    zip_path: str | Path,
    inventory: ArchiveInventory,
    output_dir: str | Path,
    *,
    budget: DiskBudgetConfig,
    run_root: str | Path,
    active_root: str | Path,
) -> Path:
    """Extract every inventoried member into ``output_dir`` and commit atomically.

    The ZIP is deleted only after ``archive_manifest.json`` is written inside
    the committed ``output_dir``; a caller must never delete the ZIP earlier.

    Raises:
        DiskAdmissionError: If disk admission is refused before extraction.
        ArchiveSafetyError: If an extracted member's actual size disagrees
            with its declared inventory size.
    """

    zip_file_path = Path(zip_path)
    final_dir = Path(output_dir)
    staging_dir = final_dir.with_name(final_dir.name + ".staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    total_declared_size = sum(member.declared_size for member in inventory.members)
    require_write_capacity(
        budget,
        run_root,
        active_root,
        total_declared_size,
        operation=f"extract_archive:{inventory.archive_id}",
    )

    with zipfile.ZipFile(zip_file_path) as archive:
        for member in inventory.members:
            archive.extract(member.member_name, path=staging_dir)
            extracted_path = staging_dir / member.member_name
            actual_size = extracted_path.stat().st_size
            if actual_size != member.declared_size:
                raise ArchiveSafetyError(
                    f"extracted size mismatch for {member.member_name}: "
                    f"declared={member.declared_size} actual={actual_size}"
                )

    manifest = {
        "archive_id": inventory.archive_id,
        "video_ids": list(inventory.video_ids),
        "members": [
            {
                "video_id": member.video_id,
                "member_name": member.member_name,
                "declared_size": member.declared_size,
            }
            for member in inventory.members
        ],
    }
    atomic_write(staging_dir / "archive_manifest.json", lambda p: write_json(manifest, p))

    if final_dir.exists():
        shutil.rmtree(final_dir)
    staging_dir.replace(final_dir)
    # The ZIP is disposable only now that the committed manifest proves every
    # inventoried member survived extraction with its declared size.
    zip_file_path.unlink(missing_ok=True)
    logger.info(
        "extracted archive %s into %s (%d members); zip removed",
        inventory.archive_id,
        final_dir,
        len(inventory.members),
    )
    return final_dir


def plan_archive_batches(
    inventory: ArchiveInventory, batch_size: int = 8
) -> tuple[tuple[str, ...], ...]:
    """Partition every inventoried video_id into canonical ordered groups.

    Every group has at most ``batch_size`` members in inventory order; only
    the final group may be a smaller remainder. No video is dropped, reordered,
    or truncated.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    video_ids = inventory.video_ids
    return tuple(
        video_ids[start : start + batch_size] for start in range(0, len(video_ids), batch_size)
    )


def stage_archive_source_links(
    extracted_dir: str | Path,
    video_ids: Sequence[str],
    source_root: str | Path,
) -> tuple[Path, ...]:
    """Hard-link one batch's extracted MP4s into the native source root.

    Hard links are same-filesystem only and share disk bytes with the
    extracted archive, so staging a batch does not inflate measured active
    working-set bytes beyond the original extraction.

    Raises:
        ArchiveSafetyError: If a requested video is missing from the extracted
            archive directory.
        OSError: If hard-linking fails, e.g. across filesystems; this function
            never falls back to a cross-device copy.
    """

    extracted_path = Path(extracted_dir)
    destination_root = Path(source_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    linked: list[Path] = []
    for video_id in video_ids:
        matches = sorted(extracted_path.rglob(f"{video_id}.mp4"))
        if not matches:
            raise ArchiveSafetyError(
                f"extracted archive is missing requested video: {video_id}"
            )
        destination_file = destination_root / f"{video_id}.mp4"
        if destination_file.exists():
            destination_file.unlink()
        os.link(matches[0], destination_file)
        linked.append(destination_file)

    logger.info(
        "staged %d hard-linked source video(s) into %s", len(linked), destination_root
    )
    return tuple(linked)


__all__ = [
    "ArchiveInventory",
    "ArchiveMember",
    "ArchiveSafetyError",
    "download_archive",
    "extract_archive_atomically",
    "inspect_archive",
    "plan_archive_batches",
    "stage_archive_source_links",
]
