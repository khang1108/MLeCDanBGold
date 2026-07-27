"""Resume validation for caption enrichment artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd

from hcmai.common.schemas import FrameEnrichment, ProcessingStatus
from hcmai.enrichment.caption.artifacts import valid_caption
from hcmai.enrichment.caption.config import CaptionConfig, ENRICHMENT_VERSION


def guard_resume(
    path: Path,
    old: dict[str, Any],
    config: CaptionConfig,
    root: Path,
    resolved_revision: str | None = None,
) -> None:
    if not path.exists():
        return
    if not old:
        raise ValueError("Cannot safely resume: manifest.json is missing")
    if old.get(ENRICHMENT_VERSION) != config.enrichment_version:
        return
    previous, current = old.get("effective_configuration"), asdict(config)
    if not isinstance(previous, dict):
        raise ValueError("Cannot safely resume: effective configuration is missing")
    changed = [key for key, value in current.items() if previous.get(key) != value]
    if old.get("dataset_root") != str(root):
        changed.append("dataset_root")
    if resolved_revision is not None:
        if old.get("resolved_model_revision") != resolved_revision:
            changed.append("resolved_model_revision")
    if changed:
        fields = ", ".join(sorted(set(changed)))
        raise ValueError(
            f"Cannot resume {config.enrichment_version!r}: changed {fields}; "
            "use a new enrichment_version or output directory"
        )


def resume_rows(
    frames: list[dict[str, Any]], path: Path, config: CaptionConfig
) -> tuple[dict[str, FrameEnrichment], list[dict[str, Any]], int, int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    if path.exists():
        try:
            prior = cast(
                list[dict[str, Any]],
                pd.read_parquet(path).to_dict(orient="records"),
            )
        except Exception as error:
            message = str(error).strip()[:200] or type(error).__name__
            raise RuntimeError(f"Cannot resume corrupted Parquet {path}: {message}") from error
        for data in prior:
            if data.get(ENRICHMENT_VERSION) == config.enrichment_version:
                groups.setdefault(str(data.get("frame_id")), []).append(data)

    rows: dict[str, FrameEnrichment] = {}
    todo: list[dict[str, Any]] = []
    skipped, retried = 0, 0
    for frame in frames:
        frame_id, old = frame["frame_id"], groups.get(frame["frame_id"], [])
        row = valid_caption(old[0], config.enrichment_version) if len(old) == 1 else None
        if row:
            rows[frame_id], skipped = row, skipped + 1
            continue
        retried += bool(old)
        rows[frame_id] = FrameEnrichment.model_validate(
            {
                "frame_id": frame_id,
                "model_name": config.model_checkpoint,
                "enrichment_version": config.enrichment_version,
                "status": ProcessingStatus.PENDING,
            }
        )
        todo.append(frame)
    return rows, todo, skipped, retried
