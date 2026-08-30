"""Typed caption artifacts and their temporary legacy projection."""

from __future__ import annotations

import math
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

from hcmai.common.schemas import CaptionEvidence, FrameEnrichment, ProcessingStatus
from hcmai.common.utils.io import (
    atomic_write,
    read_json,
    write_json,
    write_parquet,
)
from offline.enrichment.bundle import publish_staged_bundle


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
        if (
            not isinstance(data.get("frame_id"), str)
            or not data["frame_id"]
            or data["frame_id"].strip() != data["frame_id"]
            or not isinstance(data.get("video_id"), str)
            or not data["video_id"]
            or data["video_id"].strip() != data["video_id"]
            or isinstance(data.get("frame_idx"), bool)
            or not isinstance(data.get("frame_idx"), Integral)
            or isinstance(data.get("timestamp_ms"), bool)
            or not isinstance(data.get("timestamp_ms"), Integral)
        ):
            return None
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

        staged_frames = pd.read_parquet(staged[0])
        staged_projection = pd.read_parquet(staged[2])
        if staged_frames.columns.tolist() != list(CaptionEvidence.model_fields):
            raise ValueError("staged Caption evidence has an invalid schema")
        if staged_projection.columns.tolist() != list(FrameEnrichment.model_fields):
            raise ValueError("staged Caption projection has an invalid schema")
        if staged_frames["frame_id"].tolist() != order:
            raise ValueError("staged Caption rows changed canonical order")
        if staged_projection["frame_id"].tolist() != order:
            raise ValueError("staged Caption projection changed canonical order")

        for data in staged_frames.astype(object).where(
            staged_frames.notna(), None
        ).to_dict(orient="records"):
            CaptionEvidence.model_validate(data)
        for data in staged_projection.astype(object).where(
            staged_projection.notna(), None
        ).to_dict(orient="records"):
            objects = data.get("objects")
            to_list = getattr(objects, "tolist", None)
            if callable(to_list):
                data["objects"] = to_list()
            FrameEnrichment.model_validate(data)
        if read_json(staged[1]) != failure_rows:
            raise ValueError("staged Caption failures failed validation")
        if read_json(staged[3]) != manifest:
            raise ValueError("staged Caption manifest failed validation")

        versions = {row.artifact_version for row in rows.values()}
        lineages = {row.frame_store_id for row in rows.values()}
        if len(versions) > 1 or len(lineages) > 1:
            raise ValueError("Caption bundle has mixed version or lineage")
        if versions and manifest.get("artifact_version") not in versions:
            raise ValueError("Caption manifest artifact_version mismatch")
        if lineages and manifest.get("frame_store_id") not in lineages:
            raise ValueError("Caption manifest frame_store_id mismatch")

        publish_staged_bundle(staged, published)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
