"""Adapt shared temporal scenes into the existing single-frame VQA contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import UUID

from hcmai.common.config import VQAProfileConfig
from hcmai.common.schemas import (
    FrameRecord,
    RetrievalSource,
    TaskType,
    VQARetrievalEvidence,
)
from hcmai.common.schemas.search import SearchFilters
from hcmai.data.pipeline import DataService
from hcmai.retriever.pipeline import RetrievalService
from hcmai.temporal.engine import TemporalEvidenceEngine
from hcmai.temporal.models import EvidencePoint, SceneCandidate
from hcmai.vqa.evidence import build_evidence_bundle
from hcmai.vqa.models import BranchCandidate, LocalizedWindow, ParsedVQAQuery, TemporalWindow


@dataclass(frozen=True, slots=True)
class TemporalVQAResult:
    localized: list[LocalizedWindow]
    evidence_candidates: list[VQARetrievalEvidence]
    warnings: list[str]
    search_id: UUID


def localize_with_temporal_core(
    engine: TemporalEvidenceEngine,
    parsed: ParsedVQAQuery,
    search_id: UUID | None,
    filters: SearchFilters | None,
    top_k: int,
    data: DataService,
    retrieval: RetrievalService,
    profile: VQAProfileConfig,
) -> TemporalVQAResult:
    configured = replace(
        engine,
        top_k=profile.candidates_per_branch,
        max_total=profile.max_windows,
    )
    result = configured.search(
        parsed.retrieval_query,
        task_type=TaskType.VQA,
        search_id=search_id,
        filters=filters,
        allow_missing_state_fallback=True,
    )
    if result.commit_required:
        configured.states.commit(result.state, expected_version=result.state.version)
    localized, evidence = scenes_to_vqa_localized(
        result.scenes,
        data,
        question=parsed.question,
        retrieval=retrieval,
        max_frames_per_window=profile.max_frames_per_window,
        max_evidence_items=profile.max_evidence_items,
        top_k=top_k,
    )
    return TemporalVQAResult(
        localized, evidence, list(result.warnings), result.state.search_id
    )


def scenes_to_vqa_localized(
    scenes: tuple[SceneCandidate, ...],
    data: DataService,
    *,
    question: str,
    retrieval: RetrievalService,
    max_frames_per_window: int,
    max_evidence_items: int,
    top_k: int,
) -> tuple[list[LocalizedWindow], list[VQARetrievalEvidence]]:
    localized: list[LocalizedWindow] = []
    evidence: list[VQARetrievalEvidence] = []
    scene_videos = list(dict.fromkeys(scene.video_id for scene in scenes))
    # One question search for the whole request; each scene keeps only its own hits.
    hits = (
        tuple(
            data.get_frame(candidate.frame_id)
            for candidate in retrieval.search(
                question,
                max_frames_per_window * len(scenes),
                SearchFilters(video_ids=scene_videos),
                TaskType.VQA,
            ).candidates
        )
        if scene_videos
        else ()
    )
    for scene in scenes:
        points = _scene_points(scene, data)
        if not points:
            continue
        # Keep the localization frame, add question hits, then spread the leftover budget.
        anchor = max(points, key=lambda pair: pair[0].relevance_score)[1]
        chosen = {anchor.frame_id: anchor}
        for hit in hits:
            if len(chosen) >= max_frames_per_window:
                break
            if (
                hit.video_id == scene.video_id
                and scene.start_ms <= hit.timestamp_ms <= scene.end_ms
            ):
                chosen.setdefault(hit.frame_id, hit)
        remaining = [frame for _, frame in points if frame.frame_id not in chosen]
        coverage = min(max_frames_per_window - len(chosen), len(remaining))
        # Evenly spaced so coverage frames span the window instead of clustering at its head.
        step = len(remaining) / coverage if coverage > 0 else 1.0
        for index in range(coverage):
            spread = remaining[int(index * step)]
            chosen[spread.frame_id] = spread
        frames = sorted(
            chosen.values(),
            key=lambda frame: (frame.timestamp_ms, frame.frame_idx, frame.frame_id),
        )
        candidates = tuple(_branch_candidate(point, frame) for point, frame in points)
        window = TemporalWindow(
            window_id=f"temporal:{scene.video_id}:{scene.start_ms}:{scene.end_ms}",
            video_id=scene.video_id,
            start_ms=scene.start_ms,
            end_ms=scene.end_ms,
            source_frames=candidates,
            sampled_frames=tuple(frames),
            score=scene.final_score,
        )
        bundle = build_evidence_bundle(window, data, max_items=max_evidence_items)
        labels = ("temporal_core", *(name for point, _ in points for name in point.provenance))
        localized.append(
            LocalizedWindow(bundle, scene.final_score, tuple(dict.fromkeys(labels)))
        )
        if len(evidence) < top_k:
            frame = frames[0]
            evidence.append(
                VQARetrievalEvidence(
                    rank=len(evidence) + 1,
                    video_id=frame.video_id,
                    frame_id=frame.frame_id,
                    frame_idx=frame.frame_idx,
                    timestamp_ms=frame.timestamp_ms,
                    retrieval_score=scene.final_score,
                )
            )
    return localized, evidence


def _scene_points(
    scene: SceneCandidate, data: DataService
) -> tuple[tuple[EvidencePoint, FrameRecord], ...]:
    by_frame: dict[str, tuple[EvidencePoint, FrameRecord]] = {}
    for evidence_set in scene.evidence_by_unit.values():
        for point in evidence_set.points:
            frame = data.get_frame(point.frame_id)
            kept = by_frame.get(frame.frame_id)
            if kept is None or point.relevance_score > kept[0].relevance_score:
                by_frame[frame.frame_id] = (point, frame)
    return tuple(
        sorted(
            by_frame.values(),
            key=lambda pair: (pair[1].timestamp_ms, pair[1].frame_idx, pair[1].frame_id),
        )
    )


def _branch_candidate(point: EvidencePoint, frame: FrameRecord) -> BranchCandidate:
    # Provider keys source_scores by RetrievalSource.value, so the lookup cannot fail.
    source_scores = {
        RetrievalSource(source): score for source, score in point.source_scores.items()
    }
    return BranchCandidate(
        frame=frame,
        branch_scores={"temporal_core": point.relevance_score},
        source_scores=source_scores,
        source_ranks=dict.fromkeys(source_scores, 1),
        score=point.relevance_score,
        provenance=point.provenance,
    )
