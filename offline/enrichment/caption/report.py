"""Caption progress manifest and completion report."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from offline.enrichment.caption.models import CaptionEvidence
from offline.enrichment.models import ProcessingStatus
from offline.enrichment.caption.models.contracts import CaptionAdapter
from offline.enrichment.caption.config import CaptionConfig, ENRICHMENT_VERSION


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def build_manifest(
    config: CaptionConfig,
    frames_path: Path,
    root: Path,
    rows: dict[str, CaptionEvidence],
    captioner: CaptionAdapter,
    started: datetime,
    elapsed: float,
    latencies: list[float],
    skipped: int,
    retried: int,
    *,
    frame_store_id: str | None = None,
) -> dict[str, Any]:
    complete = sum(
        row.status == ProcessingStatus.COMPLETED for row in rows.values()
    )
    ordered = sorted(latencies)

    def percentile(part: float) -> float:
        return ordered[round((len(ordered) - 1) * part)] if ordered else 0.0

    return {
        # Rows and manifest share one version so downstream validation can
        # prove that the whole specialist bundle has matching lineage.
        "artifact_version": config.enrichment_version,
        "source_artifact": "captions.parquet",
        ENRICHMENT_VERSION: config.enrichment_version,
        "dataset_version": config.dataset_version,
        "input_parquet_path": str(frames_path),
        "dataset_root": str(root),
        "frame_store_id": frame_store_id,
        "model_checkpoint": config.model_checkpoint,
        "resolved_model_revision": captioner.resolved_revision,
        "prompt": config.prompt,
        "decoding": config.decoding,
        "device": config.device,
        "precision": config.precision,
        "dtype": config.dtype,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "input_frame_count": len(rows),
        "completed_count": complete,
        "failed_count": len(rows) - complete,
        "skipped_count": skipped,
        "retried_count": retried,
        "start_time": started.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "elapsed_time_sec": elapsed,
        "throughput_images_per_sec": (complete - skipped) / elapsed if elapsed else 0.0,
        "batch_latency_ms_p50": percentile(0.5),
        "batch_latency_ms_p95": percentile(0.95),
        "effective_configuration": asdict(config),
        "git_commit": _git_commit(),
    }
