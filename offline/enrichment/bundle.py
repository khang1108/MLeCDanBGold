"""Publish complete enrichment bundles with a manifest commit marker.

Callers stage and validate domain-specific files before invoking this helper.
The helper owns only ordered publication and rollback; the manifest must be
the final target so readers never treat a partially published bundle as valid.

The staging checks every specialist repeats -- canonical identity, Parquet null
normalization, staged schema and order, and manifest lineage -- live here too.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


def canonical_identity(data: Mapping[str, Any]) -> bool:
    """Report whether one row carries exact, non-coercible canonical identity."""

    frame_id = data.get("frame_id")
    video_id = data.get("video_id")
    frame_idx = data.get("frame_idx")
    timestamp_ms = data.get("timestamp_ms")
    return (
        isinstance(frame_id, str)
        and bool(frame_id)
        and frame_id.strip() == frame_id
        and isinstance(video_id, str)
        and bool(video_id)
        and video_id.strip() == video_id
        and not isinstance(frame_idx, bool)
        and isinstance(frame_idx, Integral)
        and not isinstance(timestamp_ms, bool)
        and isinstance(timestamp_ms, Integral)
    )


def null_safe(data: Mapping[str, Any]) -> dict[str, Any]:
    """Translate Parquet nulls and array columns into contract-ready values."""

    values: dict[str, Any] = {}
    for key, value in data.items():
        to_list = getattr(value, "tolist", None)
        if callable(to_list) and getattr(value, "ndim", 0):
            values[key] = to_list()
        elif (
            value is None
            or value is pd.NA
            or (isinstance(value, float) and math.isnan(value))
        ):
            values[key] = None
        else:
            values[key] = value
    return values


def staged_records(
    path: Path,
    fields: Iterable[str],
    name: str,
    *,
    expected_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Reread one staged Parquet and require its schema and canonical order."""

    table = pd.read_parquet(path)
    if table.columns.tolist() != list(fields):
        raise ValueError(f"staged {name} has an invalid schema")
    if expected_order is not None and table["frame_id"].tolist() != expected_order:
        raise ValueError(f"staged {name} changed canonical order")
    return [
        null_safe(record)
        for record in table.astype(object)
        .where(table.notna(), None)
        .to_dict(orient="records")
    ]


class _Lineage(Protocol):
    """The two provenance fields every specialist evidence row carries."""

    artifact_version: str
    frame_store_id: str | None


def validate_bundle_lineage(
    rows: Iterable[_Lineage], manifest: Mapping[str, object], name: str
) -> None:
    """Require one artifact version and lineage shared by rows and manifest."""

    versions = {row.artifact_version for row in rows}
    lineages = {row.frame_store_id for row in rows}
    if len(versions) > 1 or len(lineages) > 1:
        raise ValueError(f"{name} bundle has mixed version or lineage")
    if versions and manifest.get("artifact_version") not in versions:
        raise ValueError(f"{name} manifest artifact_version mismatch")
    if lineages and manifest.get("frame_store_id") not in lineages:
        raise ValueError(f"{name} manifest frame_store_id mismatch")


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


__all__ = [
    "canonical_identity",
    "null_safe",
    "publish_staged_bundle",
    "staged_records",
    "validate_bundle_lineage",
]
