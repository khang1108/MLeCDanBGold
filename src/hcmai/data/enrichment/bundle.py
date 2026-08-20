"""Publish complete enrichment bundles with a manifest commit marker.

Callers stage and validate domain-specific files before invoking this helper.
The helper owns only ordered publication and rollback; the manifest must be
the final target so readers never treat a partially published bundle as valid.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def publish_staged_bundle(
    staged: Sequence[Path],
    published: Sequence[Path],
) -> None:
    """Publish staged data files then manifest, restoring the prior bundle.

    Every staged file must already exist and both sequences must be ordered
    data-first, ``manifest.json`` last. A failed replacement removes any new
    files and restores old data before restoring the old manifest marker.
    """

    staged_paths = tuple(staged)
    published_paths = tuple(published)
    if not staged_paths or len(staged_paths) != len(published_paths):
        raise ValueError("staged and published bundle paths must align")
    if published_paths[-1].name != "manifest.json":
        raise ValueError("bundle manifest must be published last")
    if len(set(published_paths)) != len(published_paths):
        raise ValueError("published bundle paths must be unique")

    missing = [str(path) for path in staged_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "staged bundle is incomplete: " + ", ".join(missing)
        )

    backups = tuple(
        target.with_name(f".{target.name}.backup")
        for target in published_paths
    )
    for backup in backups:
        if backup.exists():
            raise RuntimeError(f"refusing to overwrite stale backup: {backup}")

    attempted: list[Path] = []
    restore_complete = False
    try:
        for target, backup in zip(published_paths, backups, strict=True):
            if target.exists():
                target.replace(backup)

        for source, target in zip(staged_paths, published_paths, strict=True):
            attempted.append(target)
            source.replace(target)
    except Exception:
        for target in attempted:
            target.unlink(missing_ok=True)
        # The tuple is data-first, so the old manifest is restored last.
        for target, backup in zip(published_paths, backups, strict=True):
            if backup.exists():
                backup.replace(target)
        restore_complete = True
        raise
    else:
        restore_complete = True
    finally:
        if restore_complete:
            for backup in backups:
                backup.unlink(missing_ok=True)


__all__ = ["publish_staged_bundle"]
