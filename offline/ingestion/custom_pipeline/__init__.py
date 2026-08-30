"""Local A6000 / 100 GB custom corpus pipeline package.

This package owns the resumable local pipeline that turns organizer archive
ZIPs into deterministic frames, specialist evidence, embeddings, and loadable
retrieval indexes on one bounded local machine. It does not perform cloud
upload/sync; that remains an operator-owned, out-of-band step.
"""

from __future__ import annotations

from offline.ingestion.custom_pipeline.config import (
    ArchivePlan,
    ArchivePlanEntry,
    ArchiveWorkWindow,
    CustomPipelineConfig,
    DiskBudgetConfig,
    SchedulingConfig,
    StageBatchConfig,
)
from offline.ingestion.custom_pipeline.asr import (
    ASRReuseBundle,
    require_asr_video_coverage,
    validate_asr_source,
)
from offline.ingestion.custom_pipeline.commit import (
    BatchInventory,
    BatchValidationError,
    FileInventoryEntry,
    build_batch_inventory,
    cleanup_ephemeral_batch,
    commit_local_batch,
    validate_local_batch,
)
from offline.ingestion.custom_pipeline.archive import (
    ArchiveInventory,
    ArchiveMember,
    ArchiveSafetyError,
    download_archive,
    extract_archive_atomically,
    inspect_archive,
    plan_archive_batches,
    stage_archive_source_links,
)
from offline.ingestion.custom_pipeline.contracts import RunIdentity
from offline.ingestion.custom_pipeline.disk import (
    DiskAdmissionError,
    DiskSnapshot,
    measure_tree_bytes,
    require_write_capacity,
    snapshot_disk,
)
from offline.ingestion.custom_pipeline.finalize import (
    BatchManifest,
    FinalizeError,
    build_dense_index_from_precomputed,
    build_segment_index_from_precomputed,
    compact_batch_embeddings,
    compact_frame_metadata,
    compact_specialist_shards,
    discover_committed_batches,
    finalize_corpus,
    require_full_plan_cleaned,
)
from offline.ingestion.custom_pipeline.runner import (
    BatchArtifacts,
    PreflightReport,
    RunnerContext,
    finalize_pipeline,
    pipeline_status,
    preflight_pipeline,
    process_archive,
)
from offline.ingestion.custom_pipeline.shards import (
    BatchIndexInventory,
    IndexArtifactSummary,
    VideoShard,
    VideoShardError,
    build_batch_index_bundle,
    split_batch_artifacts_by_video,
    validate_video_shard,
    write_video_shard,
)
from offline.ingestion.custom_pipeline.state import (
    ArchiveRecord,
    ArchiveStage,
    BatchRecord,
    BatchStage,
    PipelineStateStore,
    VideoRecord,
    VideoStage,
    compute_batch_id,
)

__all__ = [
    "ASRReuseBundle",
    "ArchiveInventory",
    "ArchiveMember",
    "ArchivePlan",
    "ArchivePlanEntry",
    "ArchiveRecord",
    "ArchiveSafetyError",
    "ArchiveStage",
    "ArchiveWorkWindow",
    "BatchIndexInventory",
    "BatchInventory",
    "BatchManifest",
    "BatchRecord",
    "BatchStage",
    "BatchValidationError",
    "CustomPipelineConfig",
    "DiskAdmissionError",
    "DiskBudgetConfig",
    "DiskSnapshot",
    "FileInventoryEntry",
    "FinalizeError",
    "IndexArtifactSummary",
    "PipelineStateStore",
    "RunIdentity",
    "SchedulingConfig",
    "StageBatchConfig",
    "VideoRecord",
    "VideoShard",
    "VideoShardError",
    "VideoStage",
    "compute_batch_id",
    "download_archive",
    "extract_archive_atomically",
    "inspect_archive",
    "measure_tree_bytes",
    "plan_archive_batches",
    "require_asr_video_coverage",
    "require_write_capacity",
    "snapshot_disk",
    "stage_archive_source_links",
    "validate_asr_source",
    "build_batch_index_bundle",
    "split_batch_artifacts_by_video",
    "validate_video_shard",
    "write_video_shard",
    "build_batch_inventory",
    "cleanup_ephemeral_batch",
    "commit_local_batch",
    "validate_local_batch",
    "build_dense_index_from_precomputed",
    "build_segment_index_from_precomputed",
    "compact_batch_embeddings",
    "compact_frame_metadata",
    "compact_specialist_shards",
    "discover_committed_batches",
    "finalize_corpus",
    "require_full_plan_cleaned",
    "BatchArtifacts",
    "PreflightReport",
    "RunnerContext",
    "finalize_pipeline",
    "pipeline_status",
    "preflight_pipeline",
    "process_archive",
]
