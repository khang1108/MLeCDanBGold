from __future__ import annotations

from dataclasses import dataclass

from hcmai.common.schemas.enum import TaskType
from hcmai.common.schemas.search import SearchFilters
from hcmai.temporal.evidence import EvidenceStore
from hcmai.temporal.models import QueryUnit
from hcmai.temporal.retrieval import (
    SparseEvidenceBatchRequest,
    SparseEvidenceProvider,
    SparseEvidenceRequest,
)


@dataclass(frozen=True, slots=True)
class LazyBackfillRequest:
    old_units: tuple[QueryUnit, ...]
    rescued_video_ids: tuple[str, ...]
    top_k: int = 100
    filters: SearchFilters | None = None
    task_type: TaskType = TaskType.KIS


@dataclass(frozen=True, slots=True)
class BackfilledPair:
    unit_id: str
    video_id: str


@dataclass(frozen=True, slots=True)
class LazyBackfillResult:
    updated_pairs: tuple[BackfilledPair, ...]
    warnings: tuple[str, ...] = ()


def backfill_rescued_videos(
    provider: SparseEvidenceProvider,
    store: EvidenceStore,
    request: LazyBackfillRequest,
) -> LazyBackfillResult:
    units_by_id = {unit.unit_id: unit for unit in request.old_units}
    unknown_by_video = {
        video_id: unit_ids
        for video_id in dict.fromkeys(request.rescued_video_ids)
        if (unit_ids := store.unknown_units(units_by_id, video_id))
    }
    if not unknown_by_video:
        return LazyBackfillResult(updated_pairs=())

    # Every rescued video is searched in one batch, so top_k is shared across them.
    unit_ids = tuple(
        dict.fromkeys(
            unit_id for video_units in unknown_by_video.values() for unit_id in video_units
        )
    )
    responses = provider.retrieve_batch(
        SparseEvidenceBatchRequest(
            requests=tuple(
                SparseEvidenceRequest(
                    unit_id=unit_id,
                    query=units_by_id[unit_id].text,
                    top_k=request.top_k * len(unknown_by_video),
                    filters=request.filters,
                    task_type=request.task_type,
                )
                for unit_id in unit_ids
            ),
            video_ids=tuple(unknown_by_video),
        )
    )
    if not responses:
        return LazyBackfillResult(updated_pairs=())

    by_unit = dict(zip(unit_ids, responses, strict=True))
    searched = frozenset(responses[0].searched_video_ids)
    updated: list[BackfilledPair] = []
    warnings: list[str] = []
    for video_id, video_units in unknown_by_video.items():
        if video_id not in searched:
            continue
        for unit_id in video_units:
            response = by_unit[unit_id]
            if response.warnings:
                warnings.extend(
                    f"backfill unit={unit_id} video={video_id}: {warning}"
                    for warning in response.warnings
                )
                continue
            store.record(
                unit_id,
                video_id,
                tuple(point for point in response.points if point.video_id == video_id),
            )
            updated.append(BackfilledPair(unit_id, video_id))

    return LazyBackfillResult(updated_pairs=tuple(updated), warnings=tuple(warnings))
