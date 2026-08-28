"""Local A6000 / 100 GB custom corpus pipeline package.

This package owns the resumable local pipeline that turns organizer archive
ZIPs into deterministic frames, specialist evidence, embeddings, and loadable
retrieval indexes on one bounded local machine. It does not perform cloud
upload/sync; that remains an operator-owned, out-of-band step.
"""

from __future__ import annotations

from hcmai.data.custom_pipeline.config import (
    ArchivePlan,
    ArchivePlanEntry,
    ArchiveWorkWindow,
    CustomPipelineConfig,
    DiskBudgetConfig,
    SchedulingConfig,
    StageBatchConfig,
)
from hcmai.data.custom_pipeline.asr import (
    ASRReuseBundle,
    require_asr_video_coverage,
    validate_asr_source,
)
from hcmai.data.custom_pipeline.contracts import RunIdentity
from hcmai.data.custom_pipeline.state import (
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
    "ArchivePlan",
    "ArchivePlanEntry",
    "ArchiveRecord",
    "ArchiveStage",
    "ArchiveWorkWindow",
    "BatchRecord",
    "BatchStage",
    "CustomPipelineConfig",
    "DiskBudgetConfig",
    "PipelineStateStore",
    "RunIdentity",
    "SchedulingConfig",
    "StageBatchConfig",
    "VideoRecord",
    "VideoStage",
    "compute_batch_id",
    "require_asr_video_coverage",
    "validate_asr_source",
]
