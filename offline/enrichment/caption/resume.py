"""Resume logic for source-of-truth caption evidence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd

from hcmai.common.schemas import CaptionEvidence, ProcessingStatus
from offline.enrichment.caption.artifacts import valid_caption
from offline.enrichment.caption.config import CaptionConfig, ENRICHMENT_VERSION


def guard_resume(
    path: Path,
    old: dict[str, Any],
    config: CaptionConfig,
    root: Path,
    resolved_revision: str | None = None,
    frame_store_id: str | None = None,
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
    throughput_only = {"batch_size", "write_interval"}
    changed = [
        key
        for key, value in current.items()
        if key not in throughput_only and previous.get(key) != value
    ]
    if old.get("dataset_root") != str(root):
        changed.append("dataset_root")
    if resolved_revision is not None and old.get("resolved_model_revision") != resolved_revision:
        changed.append("resolved_model_revision")
    if frame_store_id is not None and old.get("frame_store_id") != frame_store_id:
        changed.append("frame_store_id")
    if changed:
        fields = ", ".join(sorted(set(changed)))
        raise ValueError(
            f"Cannot resume {config.enrichment_version!r}: changed {fields}; "
            "use a new enrichment_version or output directory"
        )


def resume_rows(
    frames: list[dict[str, Any]],
    path: Path,
    config: CaptionConfig,
    frame_store_id: str | None = None,
) -> tuple[dict[str, CaptionEvidence], list[dict[str, Any]], int, int]:
    """Reuse only valid completed rows from ``captions.parquet``."""

    groups: dict[str, list[dict[str, Any]]] = {}
    if path.exists():
        try:
            prior = cast(
                list[dict[str, Any]], pd.read_parquet(path).to_dict(orient="records")
            )
        except Exception as error:
            message = str(error).strip()[:200] or type(error).__name__
            raise RuntimeError(f"Cannot resume corrupted Parquet {path}: {message}") from error
        for data in prior:
            groups.setdefault(str(data.get("frame_id")), []).append(data)

    rows: dict[str, CaptionEvidence] = {}
    todo: list[dict[str, Any]] = []
    skipped = retried = 0
    for frame in frames:
        frame_id = str(frame["frame_id"])
        old = groups.get(frame_id, [])
        row = (
            valid_caption(
                old[0],
                artifact_version=config.enrichment_version,
                model_name=config.model_checkpoint,
                frame_store_id=frame_store_id,
            )
            if len(old) == 1
            else None
        )
        if row is not None and (
            row.video_id != str(frame["video_id"])
            or row.frame_idx != int(frame["frame_idx"])
            or row.timestamp_ms != int(frame["timestamp_ms"])
        ):
            row = None
        if row is not None:
            rows[frame_id] = row
            skipped += 1
            continue
        retried += int(bool(old))
        rows[frame_id] = CaptionEvidence(
            frame_id=frame_id,
            video_id=str(frame["video_id"]),
            frame_idx=int(frame["frame_idx"]),
            timestamp_ms=int(frame["timestamp_ms"]),
            frame_store_id=frame_store_id,
            artifact_version=config.enrichment_version,
            model_name=config.model_checkpoint,
            status=ProcessingStatus.PENDING,
        )
        todo.append(frame)
    return rows, todo, skipped, retried
