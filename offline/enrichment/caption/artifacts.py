"""Typed caption artifacts and their temporary legacy projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from offline.enrichment.caption.models import CaptionEvidence
from offline.enrichment.models import FrameEnrichment, ProcessingStatus
from hcmai.common.utils.io import (
    atomic_write,
    read_json,
    write_json,
    write_parquet,
)
from offline.enrichment.bundle import (
    canonical_identity,
    null_safe,
    publish_staged_bundle,
    staged_records,
    validate_bundle_lineage,
)


def valid_caption(
    data: dict[str, Any],
    *,
    artifact_version: str,
    model_name: str,
    frame_store_id: str | None,
) -> CaptionEvidence | None:
    """Return a reusable completed ``CaptionEvidence`` row, if valid."""

    try:
        if not canonical_identity(data):
            return None
        row = CaptionEvidence.model_validate(null_safe(data))
    except Exception:
        return None

    reusable = (
        row.status == ProcessingStatus.COMPLETED
        and row.artifact_version == artifact_version
        and row.model_name == model_name
        and (frame_store_id is None or row.frame_store_id == frame_store_id)
        and bool((row.text or "").strip())
    )
    return row if reusable else None


def _legacy_projection(row: CaptionEvidence) -> FrameEnrichment:
    """Derive the old frame-aligned view without making it authoritative."""

    return FrameEnrichment(
        frame_id=row.frame_id,
        frame_store_id=row.frame_store_id,
        caption=row.text if row.status == ProcessingStatus.COMPLETED else None,
        enrichment_version=row.artifact_version,
        objects=[],
        model_name=row.model_name,
        status=row.status,
        error_message=row.error_message,
    )


def write_caption_artifacts(
    output: Path,
    order: list[str],
    rows: dict[str, CaptionEvidence],
    failures: dict[str, dict[str, str]],
    manifest: dict[str, Any],
) -> None:
    """Stage, validate, and publish the complete Caption bundle."""

    missing_rows = [frame_id for frame_id in order if frame_id not in rows]
    if missing_rows:
        raise ValueError(
            "caption rows do not cover canonical order: "
            + ", ".join(missing_rows[:5])
        )
    frame_table = pd.DataFrame(
        [rows[frame_id].model_dump(mode="json") for frame_id in order],
        columns=list(CaptionEvidence.model_fields),
    )
    projection_table = pd.DataFrame(
        [
            _legacy_projection(rows[frame_id]).model_dump(mode="json")
            for frame_id in order
        ],
        columns=list(FrameEnrichment.model_fields),
    )
    failure_rows = [
        failures[frame_id] for frame_id in order if frame_id in failures
    ]

    output.mkdir(parents=True, exist_ok=True)
    published = (
        output / "captions.parquet",
        output / "failures.json",
        output / "frame_enrichment.parquet",
        output / "manifest.json",
    )
    staged = tuple(
        path.with_name(f".{path.name}.staged") for path in published
    )
    try:
        atomic_write(
            staged[0],
            lambda path: write_parquet(frame_table, path, index=False),
        )
        atomic_write(staged[1], lambda path: write_json(failure_rows, path))
        atomic_write(
            staged[2],
            lambda path: write_parquet(projection_table, path, index=False),
        )
        atomic_write(staged[3], lambda path: write_json(manifest, path))

        for data in staged_records(
            staged[0],
            CaptionEvidence.model_fields,
            "Caption evidence",
            expected_order=order,
        ):
            CaptionEvidence.model_validate(data)
        for data in staged_records(
            staged[2],
            FrameEnrichment.model_fields,
            "Caption projection",
            expected_order=order,
        ):
            FrameEnrichment.model_validate(data)
        if read_json(staged[1]) != failure_rows:
            raise ValueError("staged Caption failures failed validation")
        if read_json(staged[3]) != manifest:
            raise ValueError("staged Caption manifest failed validation")

        validate_bundle_lineage(rows.values(), manifest, "Caption")
        publish_staged_bundle(staged, published)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
