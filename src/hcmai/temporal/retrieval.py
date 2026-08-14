from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from hcmai.common.schemas.enum import TaskType
from hcmai.common.schemas.frame import FrameLookup, FrameRecord
from hcmai.common.schemas.retrieval import RetrievalCandidate, RetrievalResult
from hcmai.common.schemas.search import SearchFilters
from hcmai.common.schemas.telemetry import RetrievalTrace
from hcmai.retriever.models.contracts import Retriever
from hcmai.temporal.evidence import EvidenceStore
from hcmai.temporal.models import EvidencePoint


@dataclass(frozen=True, slots=True)
class SparseEvidenceRequest:
    unit_id: str
    query: str
    top_k: int = 100
    filters: SearchFilters | None = None
    task_type: TaskType = TaskType.KIS


@dataclass(frozen=True, slots=True)
class SparseEvidenceBatchRequest:
    requests: tuple[SparseEvidenceRequest, ...]
    video_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SparseEvidenceResponse:
    points: tuple[EvidencePoint, ...]
    trace: RetrievalTrace
    warnings: tuple[str, ...] = ()
    searched_video_ids: tuple[str, ...] = ()


class SparseEvidenceProvider:
    def __init__(self, retrieval: Retriever, frames: FrameLookup) -> None:
        self._retrieval = retrieval
        self._frames = frames

    def retrieve(self, request: SparseEvidenceRequest) -> SparseEvidenceResponse:
        result = self._retrieval.search(
            request.query,
            request.top_k,
            request.filters,
            request.task_type,
        )
        return self._convert(request.unit_id, result, branch="global")

    def retrieve_local(
        self,
        request: SparseEvidenceRequest,
        video_ids: tuple[str, ...],
    ) -> SparseEvidenceResponse:
        filters = _intersect_filters(request.filters, video_ids)
        if filters is None:
            return SparseEvidenceResponse(points=(), trace=RetrievalTrace())
        result = self._retrieval.search(
            request.query,
            request.top_k,
            filters,
            request.task_type,
        )
        return self._convert(
            request.unit_id,
            result,
            branch="local",
            searched_video_ids=tuple(filters.video_ids),
        )

    def retrieve_batch(
        self, request: SparseEvidenceBatchRequest
    ) -> tuple[SparseEvidenceResponse, ...]:
        if not request.requests:
            return ()
        first = request.requests[0]
        filters = _intersect_filters(first.filters, request.video_ids)
        if filters is None:
            return ()
        results = self._retrieval.search_batch(
            [unit.query for unit in request.requests],
            first.top_k,
            filters,
            first.task_type,
        )
        searched_video_ids = tuple(filters.video_ids)
        return tuple(
            self._convert(
                unit.unit_id,
                result,
                branch="backfill",
                searched_video_ids=searched_video_ids,
            )
            for unit, result in zip(request.requests, results, strict=True)
        )

    def _convert(
        self,
        unit_id: str,
        result: RetrievalResult,
        *,
        branch: str,
        searched_video_ids: tuple[str, ...] = (),
    ) -> SparseEvidenceResponse:
        points = tuple(
            _candidate_to_point(
                unit_id,
                candidate,
                self._frames.get_frame(candidate.frame_id),
                rank,
                branch,
            )
            for rank, candidate in enumerate(result.candidates, start=1)
        )
        return SparseEvidenceResponse(
            points=points,
            trace=RetrievalTrace().merged(result.trace, prefix=branch),
            warnings=tuple(result.warnings),
            searched_video_ids=searched_video_ids,
        )


@dataclass(frozen=True, slots=True)
class GatheredEvidence:
    points: tuple[EvidencePoint, ...]
    rescued_video_ids: tuple[str, ...]
    trace: RetrievalTrace
    warnings: tuple[str, ...] = ()


def gather_evidence(
    provider: SparseEvidenceProvider,
    store: EvidenceStore,
    request: SparseEvidenceRequest,
    *,
    global_quota: int = 100,
    local_quota: int = 100,
) -> GatheredEvidence:
    if global_quota < 1 or local_quota < 1:
        raise ValueError("quotas must be at least 1")

    known_video_ids = store.video_ids()
    if known_video_ids:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hcmai-global") as executor:
            future = executor.submit(provider.retrieve, request)
            local = provider.retrieve_local(request, known_video_ids)
        global_ = future.result()
    else:
        global_ = provider.retrieve(request)
        local = SparseEvidenceResponse(points=(), trace=RetrievalTrace())

    kept = _apply_quotas(
        _merge_points(global_.points, local.points), global_quota, local_quota
    )
    points_by_video: dict[str, list[EvidencePoint]] = {}
    for point in kept:
        points_by_video.setdefault(point.video_id, []).append(point)

    # Locally searched videos without a match are evaluated, not unknown.
    for video_id in dict.fromkeys((*points_by_video, *local.searched_video_ids)):
        store.record(request.unit_id, video_id, points_by_video.get(video_id, ()))

    return GatheredEvidence(
        points=kept,
        rescued_video_ids=tuple(
            video_id for video_id in points_by_video if video_id not in known_video_ids
        ),
        trace=global_.trace.merged(local.trace),
        warnings=(*global_.warnings, *local.warnings),
    )


def _merge_points(
    global_points: tuple[EvidencePoint, ...],
    local_points: tuple[EvidencePoint, ...],
) -> tuple[EvidencePoint, ...]:
    merged: dict[tuple[str, str, int], EvidencePoint] = {}
    for point in (*global_points, *local_points):
        kept = merged.get(point.canonical_identity)
        if kept is None:
            merged[point.canonical_identity] = point
            continue
        best = point if point.relevance_score > kept.relevance_score else kept
        merged[point.canonical_identity] = replace(
            best,
            source_scores={**kept.source_scores, **point.source_scores},
            provenance=tuple(dict.fromkeys((*kept.provenance, *point.provenance))),
        )
    return tuple(merged.values())


def _apply_quotas(
    points: tuple[EvidencePoint, ...],
    global_quota: int,
    local_quota: int,
) -> tuple[EvidencePoint, ...]:
    ranked = sorted(
        points,
        key=lambda point: (-point.relevance_score, point.frame_idx, point.frame_id),
    )
    from_global = [point for point in ranked if "global" in point.provenance]
    from_local = [point for point in ranked if "global" not in point.provenance]
    kept = {
        point.canonical_identity
        for point in (*from_global[:global_quota], *from_local[:local_quota])
    }
    return tuple(point for point in ranked if point.canonical_identity in kept)


def _intersect_filters(
    filters: SearchFilters | None,
    video_ids: tuple[str, ...],
) -> SearchFilters | None:
    local_video_ids = list(dict.fromkeys(video_ids))
    if filters is None:
        return SearchFilters(video_ids=local_video_ids) if local_video_ids else None

    if filters.video_ids:
        allowed = set(local_video_ids)
        local_video_ids = [
            video_id for video_id in filters.video_ids if video_id in allowed
        ]
    if not local_video_ids:
        return None
    return filters.model_copy(update={"video_ids": local_video_ids})


def _candidate_to_point(
    unit_id: str,
    candidate: RetrievalCandidate,
    frame: FrameRecord,
    rank: int,
    branch: str,
) -> EvidencePoint:
    sources = sorted(candidate.source_scores.items(), key=lambda item: item[0].value)
    source_scores = {
        source.value: score for source, score in sources if 0.0 <= score <= 1.0
    }
    source_names = tuple(source.value for source, _ in sources)
    fusion = candidate.fusion_score
    return EvidencePoint(
        unit_id=unit_id,
        video_id=frame.video_id,
        frame_id=frame.frame_id,
        frame_idx=frame.frame_idx,
        timestamp_ms=frame.timestamp_ms,
        # No usable fusion score falls back to the best valid source, then to rank decay.
        relevance_score=(
            fusion
            if fusion is not None and 0.0 <= fusion <= 1.0
            else max(source_scores.values(), default=1 / rank)
        ),
        source_scores=source_scores,
        provenance=(branch, *source_names),
    )
