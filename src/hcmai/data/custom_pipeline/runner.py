"""Compose preflight, rolling archive processing, status, and finalization.

This module is the local pipeline's top-level orchestration. It never
invents the exact per-batch specialist/embedding stage commands itself: the
caller supplies a ``produce_batch_artifacts`` callback that is expected to
run the real local stages (Task 7's :func:`run_batch_stages`) and read their
outputs back into the tables/vectors :mod:`shards` needs, including advancing
each video's state through ``captioned``/``ocr_complete``/... up to
``embeddings_complete``. This runner owns archive/batch download, extraction,
canonical grouping, shard splitting, index building, local commit, ephemeral
cleanup, and the final ``local_complete``/``committed``/``cleaned``
transitions.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hcmai.common.utils.io import write_parquet
from hcmai.common.utils.logging import get_logger
from hcmai.data.custom_pipeline.archive import (
    ArchiveInventory,
    download_archive,
    extract_archive_atomically,
    inspect_archive,
    plan_archive_batches,
    stage_archive_source_links,
)
from hcmai.data.custom_pipeline.asr import ASRReuseBundle, require_asr_video_coverage
from hcmai.data.custom_pipeline.commit import (
    build_batch_inventory,
    cleanup_ephemeral_batch,
    commit_local_batch,
    validate_local_batch,
)
from hcmai.data.custom_pipeline.config import (
    ArchivePlan,
    ArchivePlanEntry,
    ArchiveWorkWindow,
    DiskBudgetConfig,
    SchedulingConfig,
)
from hcmai.data.custom_pipeline.disk import DiskSnapshot, snapshot_disk
from hcmai.data.custom_pipeline.finalize import finalize_corpus
from hcmai.data.custom_pipeline.shards import build_batch_index_bundle, split_batch_artifacts_by_video, write_video_shard
from hcmai.data.custom_pipeline.state import (
    ArchiveStage,
    BatchStage,
    PipelineStateStore,
    VideoStage,
    compute_batch_id,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class RunnerContext:
    """Local roots and resource configuration shared by every runner operation."""

    run_root: Path
    artifacts_root: Path
    native_executable: Path
    disk_budget: DiskBudgetConfig
    scheduling: SchedulingConfig

    @property
    def active_root(self) -> Path:
        return self.run_root / "active"


@dataclass(frozen=True)
class PreflightReport:
    """Local readiness result; never downloads or extracts an archive."""

    native_executable_found: bool
    ffmpeg_found: bool
    curl_found: bool
    measured_cpus: int
    disk: DiskSnapshot
    archive_plan_size: int
    work_window: dict[str, object]
    ok: bool
    problems: tuple[str, ...]


@dataclass(frozen=True)
class BatchArtifacts:
    """Batch-scoped tables and vectors ready for :mod:`shards`, from real stages."""

    frames_table: pd.DataFrame
    frame_native_tables: dict[str, pd.DataFrame]
    child_tables: dict[str, pd.DataFrame]
    visual_vectors: np.ndarray
    visual_mapping: pd.DataFrame
    context_vectors: np.ndarray
    context_mapping: pd.DataFrame


ProduceBatchArtifacts = Callable[[str, Sequence[str], Sequence[Path]], BatchArtifacts]
ASRBundleFactory = Callable[[Sequence[str]], ASRReuseBundle]


def preflight_pipeline(
    context: RunnerContext, plan: ArchivePlan, window: ArchiveWorkWindow
) -> PreflightReport:
    """Validate local prerequisites and the requested work window.

    Performs no archive download. Aggregates every problem instead of failing
    on the first one so an operator sees the complete local readiness gap.
    """

    problems: list[str] = []

    native_found = Path(context.native_executable).is_file()
    if not native_found:
        problems.append(f"native executable not found: {context.native_executable}")

    ffmpeg_found = shutil.which("ffmpeg") is not None
    if not ffmpeg_found:
        problems.append("ffmpeg not found on PATH")

    curl_found = shutil.which("curl") is not None
    if not curl_found:
        problems.append("curl not found on PATH")

    import os

    measured_cpus = os.cpu_count() or 0
    if measured_cpus < context.scheduling.available_cpus:
        problems.append(
            f"measured CPU count ({measured_cpus}) is below configured "
            f"available_cpus ({context.scheduling.available_cpus})"
        )

    try:
        window.select(plan)
    except ValueError as error:
        problems.append(str(error))

    context.active_root.mkdir(parents=True, exist_ok=True)
    disk = snapshot_disk(context.run_root, context.active_root)
    if disk.free_bytes < context.disk_budget.min_free_bytes:
        problems.append(
            f"free disk ({disk.free_bytes}) is below min_free_bytes "
            f"({context.disk_budget.min_free_bytes})"
        )

    report = PreflightReport(
        native_executable_found=native_found,
        ffmpeg_found=ffmpeg_found,
        curl_found=curl_found,
        measured_cpus=measured_cpus,
        disk=disk,
        archive_plan_size=len(plan.entries),
        work_window={"offset": window.offset, "limit": window.limit},
        ok=not problems,
        problems=tuple(problems),
    )
    logger.info("preflight %s: %s", "OK" if report.ok else "FAILED", problems or "no problems found")
    return report


def process_archive(
    context: RunnerContext,
    state_store: PipelineStateStore,
    archive_entry: ArchivePlanEntry,
    produce_batch_artifacts: ProduceBatchArtifacts,
    asr_bundle_factory: ASRBundleFactory,
    *,
    dataset_version: str,
    visual_model_name: str,
    context_model_name: str,
    batch_offset: int = 0,
    batch_limit: int | None = None,
) -> list[str]:
    """Resume one contiguous slice of an archive's batches through commit and cleanup.

    Returns:
        The ordered list of committed ``batch_id`` values for this archive.
    """

    archive_id = archive_entry.archive_id
    state_store.ensure_archive(archive_id, archive_entry.position)
    archive_active_root = context.active_root / "archives" / archive_id
    zip_path = context.active_root / "archives" / f"{archive_id}.zip"

    record = state_store.get_archive(archive_id)
    assert record is not None

    if record.stage in (ArchiveStage.PENDING, ArchiveStage.DOWNLOADING):
        if record.stage == ArchiveStage.PENDING:
            state_store.advance_archive(archive_id, ArchiveStage.DOWNLOADING)
        download_archive(
            archive_entry.url,
            zip_path,
            budget=context.disk_budget,
            run_root=context.run_root,
            active_root=context.active_root,
        )
        state_store.advance_archive(archive_id, ArchiveStage.DOWNLOADED)
        record = state_store.get_archive(archive_id)
        assert record is not None

    inventory: ArchiveInventory
    if record.stage == ArchiveStage.DOWNLOADED:
        inventory = inspect_archive(zip_path, budget=context.disk_budget)
        extract_archive_atomically(
            zip_path,
            inventory,
            archive_active_root,
            budget=context.disk_budget,
            run_root=context.run_root,
            active_root=context.active_root,
        )
        state_store.advance_archive(archive_id, ArchiveStage.EXTRACTED)
        state_store.advance_archive(archive_id, ArchiveStage.PROCESSING)
    else:
        inventory = _reload_inventory_from_manifest(archive_active_root)

    batch_ids: list[str] = []
    groups = plan_archive_batches(inventory, batch_size=context.scheduling.max_videos_per_batch)
    stop = len(groups) if batch_limit is None else batch_offset + batch_limit
    for batch_index, video_group in list(enumerate(groups))[batch_offset:stop]:
        batch_id = compute_batch_id(archive_id, batch_index)
        batch_ids.append(batch_id)
        _process_one_batch(
            context,
            state_store,
            archive_id,
            archive_active_root,
            batch_id,
            list(video_group),
            produce_batch_artifacts,
            asr_bundle_factory,
            dataset_version=dataset_version,
            visual_model_name=visual_model_name,
            context_model_name=context_model_name,
        )

    state_store.advance_archive(archive_id, ArchiveStage.COMPLETE)
    if archive_active_root.exists():
        shutil.rmtree(archive_active_root)
    state_store.advance_archive(archive_id, ArchiveStage.CLEANED)
    logger.info("archive %s complete: %d batch(es) committed", archive_id, len(batch_ids))
    return batch_ids


def _reload_inventory_from_manifest(archive_active_root: Path) -> ArchiveInventory:
    """Rebuild an :class:`ArchiveInventory` from an already-extracted archive."""

    from hcmai.common.utils.io import read_json
    from hcmai.data.custom_pipeline.archive import ArchiveMember

    manifest = read_json(archive_active_root / "archive_manifest.json")
    members = tuple(
        ArchiveMember(
            video_id=row["video_id"],
            member_name=row["member_name"],
            declared_size=row["declared_size"],
        )
        for row in manifest["members"]
    )
    return ArchiveInventory(archive_id=manifest["archive_id"], members=members)


def _process_one_batch(
    context: RunnerContext,
    state_store: PipelineStateStore,
    archive_id: str,
    archive_active_root: Path,
    batch_id: str,
    video_ids: list[str],
    produce_batch_artifacts: ProduceBatchArtifacts,
    asr_bundle_factory: ASRBundleFactory,
    *,
    dataset_version: str,
    visual_model_name: str,
    context_model_name: str,
) -> None:
    """Run one canonical batch from source staging through commit and cleanup."""

    existing = state_store.get_batch(batch_id)
    if existing is not None and existing.stage in (
        BatchStage.COMMITTED,
        BatchStage.EPHEMERAL_CLEANED,
    ):
        logger.info("batch %s already committed; skipping", batch_id)
        return

    state_store.ensure_batch(batch_id, archive_id, video_ids)
    for video_id in video_ids:
        state_store.ensure_video(video_id, batch_id)

    native_source_root = context.active_root / "native" / batch_id
    source_paths = stage_archive_source_links(archive_active_root, video_ids, native_source_root)
    state_store.advance_batch(batch_id, BatchStage.EXTRACTED)

    artifacts = produce_batch_artifacts(batch_id, video_ids, list(source_paths))
    shards = split_batch_artifacts_by_video(
        video_ids,
        artifacts.frames_table,
        artifacts.frame_native_tables,
        artifacts.child_tables,
        artifacts.visual_vectors,
        artifacts.visual_mapping,
        artifacts.context_vectors,
        artifacts.context_mapping,
    )
    staging_root = context.active_root / "batch" / batch_id
    for video_id in video_ids:
        write_video_shard(shards[video_id], staging_root)
    write_parquet(artifacts.frames_table, staging_root / "frames.parquet", index=False)
    state_store.advance_batch(batch_id, BatchStage.ARTIFACTS_COMPLETE)

    asr_bundle = asr_bundle_factory(video_ids)
    require_asr_video_coverage(asr_bundle, video_ids)
    build_batch_index_bundle(
        batch_id,
        video_ids,
        shards,
        asr_bundle,
        staging_root,
        dataset_version=dataset_version,
        visual_model_name=visual_model_name,
        context_model_name=context_model_name,
    )
    state_store.advance_batch(batch_id, BatchStage.INDEXES_COMPLETE)

    inventory = build_batch_inventory(staging_root, batch_id, video_ids)
    validate_local_batch(batch_id, video_ids, staging_root, inventory)
    final_batch_root = context.artifacts_root / "batches" / archive_id / batch_id
    commit_local_batch(staging_root, final_batch_root, inventory)
    state_store.advance_batch(batch_id, BatchStage.COMMITTED)

    for video_id in video_ids:
        state_store.advance_video(video_id, VideoStage.LOCAL_COMPLETE)

    cleanup_ephemeral_batch(
        state_store,
        batch_id,
        [native_source_root],
        allowed_root=context.active_root,
    )


def pipeline_status(
    context: RunnerContext, state_store: PipelineStateStore, plan: ArchivePlan
) -> dict[str, object]:
    """Report read-only local state: per-archive stages and next offset.

    Never mutates state and never treats file existence alone as completion;
    every reported stage comes from the persisted state store.
    """

    archive_stages: dict[str, str] = {}
    for entry in plan.entries:
        record = state_store.get_archive(entry.archive_id)
        archive_stages[entry.archive_id] = record.stage.value if record is not None else "pending"

    cleaned_positions = [
        entry.position
        for entry in plan.entries
        if archive_stages[entry.archive_id] == ArchiveStage.CLEANED.value
    ]
    recommended_next_offset = (max(cleaned_positions) + 1) if cleaned_positions else 0
    complete_corpus = len(cleaned_positions) == len(plan.entries)

    disk = snapshot_disk(context.run_root, context.active_root)
    return {
        "archives": archive_stages,
        "recommended_next_offset": recommended_next_offset,
        "complete_corpus": complete_corpus,
        "free_bytes": disk.free_bytes,
        "active_bytes": disk.active_bytes,
    }


def finalize_pipeline(
    context: RunnerContext,
    state_store: PipelineStateStore,
    plan: ArchivePlan,
    batches_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    dataset_version: str,
) -> dict[str, object]:
    """Finalize the corpus once every archive in ``plan`` is cleaned.

    Delegates entirely to :func:`hcmai.data.custom_pipeline.finalize.finalize_corpus`.
    """

    archive_ids = [entry.archive_id for entry in plan.entries]
    return finalize_corpus(
        state_store,
        archive_ids,
        batches_root,
        dataset_root,
        output_root,
        dataset_version=dataset_version,
    )


__all__ = [
    "BatchArtifacts",
    "PreflightReport",
    "RunnerContext",
    "finalize_pipeline",
    "pipeline_status",
    "preflight_pipeline",
    "process_archive",
]
