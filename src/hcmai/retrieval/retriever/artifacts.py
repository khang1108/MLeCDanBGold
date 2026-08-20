"""Integrity and publication primitives for retrieval artifact directories.

This module owns file-level integrity checks and safe same-filesystem directory
publication. It does not define index layouts or validate index contents.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterable
from pathlib import Path


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file without loading it all into memory.

    Args:
        path: Complete file to hash.
        chunk_bytes: Read size used while streaming the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[Path]) -> str:
    """Return an order-independent fingerprint for a named collection of files.

    Both each basename and its content digest participate so changing an
    artifact file or substituting it under a different artifact name changes
    the resulting provenance value.
    """
    entries = sorted(
        (Path(path).name, sha256_file(Path(path)))
        for path in paths
    )
    digest = hashlib.sha256()
    for name, file_digest in entries:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def publish_directory(staged: Path, destination: Path) -> Path:
    """Atomically replace a sibling directory and recover the old publication.

    ``staged`` must be a sibling of ``destination`` so both renames are on the
    same filesystem. A pre-existing backup signals an interrupted prior
    publication and is refused to avoid overwriting the only recoverable copy.
    """
    staged = staged.resolve()
    destination = destination.resolve()
    if not staged.is_dir():
        raise FileNotFoundError(staged)
    if staged == destination:
        raise ValueError("Staged and destination directories must differ")
    if staged.parent != destination.parent:
        raise ValueError(
            "Staged and destination directories must be siblings for atomic publication"
        )

    backup = destination.with_name(destination.name + ".backup")
    if backup.exists():
        raise RuntimeError(f"Stale publication backup exists: {backup}")

    try:
        if destination.exists():
            destination.replace(backup)
        staged.replace(destination)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        # If the second rename completed but cleanup failed, discard the newly
        # published directory so the previous complete bundle remains usable.
        if destination.exists() and backup.exists():
            shutil.rmtree(destination)
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    return destination
