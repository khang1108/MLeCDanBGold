"""Validation and persistence for caption enrichment artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from hcmai.common.schemas import FrameEnrichment, ProcessingStatus
from hcmai.common.utils.io import atomic_write, write_json, write_parquet
from hcmai.data.enrichment.caption.config import ENRICHMENT_VERSION


def _null_scalar(value: object) -> bool:
    return value is None or isinstance(value, float) and math.isnan(value)


def valid_caption(
    data: dict[str, Any], version: str
) -> FrameEnrichment | None:
    """Return a reusable completed caption row, if valid."""

    try:
        values, objects = dict(data), data.get("objects")
        to_list = getattr(objects, "tolist", None)
        values["objects"] = to_list() if callable(to_list) else objects or []
        nullable = (
            "caption",
            "detailed_caption",
            "ocr_text",
            "asr_text",
            ENRICHMENT_VERSION,
            "error_message",
        )
        values.update(
            {key: None for key in nullable if _null_scalar(values.get(key))}
        )
        row = FrameEnrichment.model_validate(values)
    except Exception:
        return None
    complete = row.status == ProcessingStatus.COMPLETED
    reusable = bool(row.caption and row.caption.strip()) and row.error_message is None
    return row if row.enrichment_version == version and complete and reusable else None


def write_caption_artifacts(
    output: Path,
    order: list[str],
    rows: dict[str, FrameEnrichment],
    failures: dict[str, dict[str, str]],
) -> None:
    """Atomically write caption rows and structured failures."""

    data = pd.DataFrame([rows[key].model_dump(mode="json") for key in order])
    atomic_write(
        output / "frame_enrichment.parquet",
        lambda path: write_parquet(data, path, index=False),
    )
    atomic_write(
        output / "failures.json",
        lambda path: write_json(
            [failures[key] for key in order if key in failures], path
        ),
    )
