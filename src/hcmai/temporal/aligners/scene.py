"""Bounded scene clustering and scoring for progressive KIS/VQA evidence."""

from __future__ import annotations

from hcmai.common.config import ProgressiveSearchConfig
from hcmai.common.schemas import (
    FrameEvidence,
    SceneCandidate,
    TemporalAlignmentMode,
    TemporalQueryPlan,
)

from ..evidence import ProgressiveEvidenceState
from ..scoring import rank_scenes, score_scene, unit_score_bounds


class ProgressiveSceneAligner:
    """Align sparse canonical evidence into ranked bounded scenes."""

    def __init__(self, config: ProgressiveSearchConfig) -> None:
        self.config = config

    def align(
        self,
        plan: TemporalQueryPlan,
        evidence: ProgressiveEvidenceState,
    ) -> tuple[SceneCandidate, ...]:
        if plan.alignment_mode is not TemporalAlignmentMode.PROGRESSIVE_SCENE:
            raise ValueError("scene alignment requires a progressive-scene plan")
        video_ids = sorted({video_id for _, video_id in evidence.evaluated_keys})
        bounds = unit_score_bounds(evidence, set(video_ids))
        all_scenes: list[SceneCandidate] = []
        for video_id in video_ids:
            items = [
                item
                for (_, candidate_video), retained in evidence.evidence.items()
                if candidate_video == video_id
                for item in retained
            ]
            scenes = cluster_video_evidence(
                video_id,
                items,
                max_gap_ms=self.config.scene_max_gap_ms,
                max_span_ms=self.config.scene_max_span_ms,
            )
            scored = [
                score_scene(
                    scene,
                    list(plan.units),
                    evidence,
                    list(plan.constraints),
                    self.config,
                    coherence_window_ms=self.config.scene_coherence_ms,
                    unit_score_bounds=bounds,
                )
                for scene in scenes
            ]
            all_scenes.extend(
                rank_scenes(scored)[: self.config.scene_top_b_per_video]
            )
        return tuple(
            rank_scenes(all_scenes)[: self.config.scene_top_p_global]
        )


def cluster_video_evidence(
    video_id: str,
    evidence: list[FrameEvidence],
    *,
    max_gap_ms: int,
    max_span_ms: int,
) -> list[SceneCandidate]:
    """Split canonical evidence on both adjacent gap and total scene span."""

    unique: dict[str, FrameEvidence] = {}
    for item in evidence:
        prior = unique.get(item.frame.frame_id)
        if prior is None:
            unique[item.frame.frame_id] = item
            continue
        source_scores = {**prior.source_scores, **item.source_scores}
        source_ranks = {**prior.source_ranks, **item.source_ranks}
        unique[item.frame.frame_id] = prior.model_copy(update={
            "unit_scores": {**prior.unit_scores, **item.unit_scores},
            "source_scores": {
                source: max(prior.source_scores.get(source, score), score)
                for source, score in source_scores.items()
            },
            "source_ranks": {
                source: min(prior.source_ranks.get(source, rank), rank)
                for source, rank in source_ranks.items()
            },
            "score": max(prior.score, item.score),
            "provenance": tuple(
                dict.fromkeys((*prior.provenance, *item.provenance))
            ),
        })
    ordered = sorted(
        unique.values(),
        key=lambda item: (item.frame.timestamp_ms, item.frame.frame_id),
    )
    if not ordered:
        return []
    clusters: list[list[FrameEvidence]] = [[ordered[0]]]
    for item in ordered[1:]:
        start = clusters[-1][0].frame.timestamp_ms
        previous = clusters[-1][-1].frame.timestamp_ms
        if (
            item.frame.timestamp_ms - previous <= max_gap_ms
            and item.frame.timestamp_ms - start <= max_span_ms
        ):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return [
        SceneCandidate(
            scene_id=f"{video_id}:{cluster[0].frame.timestamp_ms}-"
            f"{cluster[-1].frame.timestamp_ms}",
            video_id=video_id,
            start_ms=cluster[0].frame.timestamp_ms,
            end_ms=cluster[-1].frame.timestamp_ms,
            evidence=tuple(cluster),
            reason_labels=("bounded_temporal_cluster",),
        )
        for cluster in clusters
    ]
