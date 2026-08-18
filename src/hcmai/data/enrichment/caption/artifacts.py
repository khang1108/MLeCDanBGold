"""Typed caption artifacts and their temporary legacy projection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from hcmai.common.schemas import CaptionEvidence, FrameEnrichment, ProcessingStatus
from hcmai.common.utils.io import atomic_write, write_json, write_parquet


def _null_scalar(value: object) -> bool:
    return value is None or value is pd.NA or (
        isinstance(value, float) and math.isnan(value)
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
        values = dict(data)
        for field in (
            "text",
            "frame_store_id",
            "model_revision",
            "error_code",
            "error_message",
        ):
            if _null_scalar(values.get(field)):
                values[field] = None
        row = CaptionEvidence.model_validate(values)
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
) -> None:
    """Atomically write typed source evidence and its compatibility view."""

    evidence = [rows[frame_id].model_dump(mode="json") for frame_id in order]
    projection = [
        _legacy_projection(rows[frame_id]).model_dump(mode="json")
        for frame_id in order
    ]
    atomic_write(
        output / "captions.parquet",
        lambda path: write_parquet(pd.DataFrame(evidence), path, index=False),
    )
    atomic_write(
        output / "failures.json",
        lambda path: write_json(
            [failures[frame_id] for frame_id in order if frame_id in failures], path
        ),
    )
    atomic_write(
        output / "frame_enrichment.parquet",
        lambda path: write_parquet(pd.DataFrame(projection), path, index=False),
    )
