"""One late reranking stage that nudges assembled scenes without replacing their score."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import RetrievalCandidate, StageStatus, StageTrace
from hcmai.observability import PipelineStage
from hcmai.observability.tracing import StageTimer
from hcmai.orchestration.pipelines.kis.representative import RepresentativeFrameSelector
from hcmai.orchestration.ranking import reranker_backend
from hcmai.reranking.pipeline import RerankingError, RerankingService
from hcmai.temporal.models import SceneCandidate


@dataclass(frozen=True, slots=True)
class SceneRerank:
    """Reordered scenes plus the stage trace carrying probe count and latency."""

    scenes: tuple[SceneCandidate, ...]
    trace: StageTrace
    warnings: tuple[str, ...] = ()


def rerank_scenes(
    query: str,
    scenes: Sequence[SceneCandidate],
    reranking: RerankingService,
    selector: RepresentativeFrameSelector,
    config: SearchConfig,
) -> SceneRerank:
    """Score a few evidence frames per Top-P scene and blend that signal into the order."""
    timer = StageTimer(PipelineStage.RERANK.value)
    head = scenes[: config.scene_rerank_top_p]
    probes_by_scene = [
        tuple(
            candidate.frame_id
            for candidate in selector.rank(scene)[: config.scene_rerank_frames_per_scene]
        )
        for scene in head
    ]
    probes = tuple(
        dict.fromkeys(frame_id for frame_ids in probes_by_scene for frame_id in frame_ids)
    )
    if not probes:
        return SceneRerank(
            scenes=tuple(scenes),
            trace=timer.finish(
                status=StageStatus.SKIPPED,
                attempt_count=0,
                input_count=0,
                output_count=len(scenes),
            ),
        )

    try:
        ranked = reranking.rerank(
            query, [RetrievalCandidate(frame_id=frame_id) for frame_id in probes]
        )
    except RerankingError as error:
        if reranking.config.required:
            raise
        return SceneRerank(
            scenes=tuple(scenes),
            trace=timer.finish(
                status=StageStatus.PARTIAL,
                error_category=error.category,
                input_count=len(probes),
                output_count=len(scenes),
                backend=reranker_backend(reranking),
                fallback_used=True,
            ),
            warnings=(f"scene reranking fallback ({error.category})",),
        )

    signal = _normalized(
        {
            candidate.frame_id: candidate.reranker_score
            for candidate in ranked
            if candidate.reranker_score is not None
        }
    )
    # Only the Top-P prefix is reordered; deeper scenes keep their position behind it.
    reordered = sorted(
        zip(head, probes_by_scene, strict=True),
        key=lambda pair: (
            -_blended(pair[0], pair[1], signal, config.scene_rerank_weight),
            pair[0].start_ms,
            pair[0].end_ms,
            pair[0].video_id,
        ),
    )
    return SceneRerank(
        scenes=(
            *(scene for scene, _ in reordered),
            *scenes[len(head) :],
        ),
        trace=timer.finish(
            input_count=len(probes),
            output_count=len(scenes),
            backend=reranker_backend(reranking),
        ),
    )


def _blended(
    scene: SceneCandidate,
    frame_ids: tuple[str, ...],
    signal: dict[str, float],
    weight: float,
) -> float:
    scores = [signal[frame_id] for frame_id in frame_ids if frame_id in signal]
    if not scores:
        return scene.final_score
    return (1.0 - weight) * scene.final_score + weight * max(scores)


def _normalized(scores: dict[str, float]) -> dict[str, float]:
    """Reranker scale is model-specific, so probes are only ever compared to each other."""
    if not scores:
        return {}
    lowest = min(scores.values())
    spread = max(scores.values()) - lowest
    if spread <= 0.0:
        return dict.fromkeys(scores, 1.0)
    return {
        frame_id: (score - lowest) / spread for frame_id, score in scores.items()
    }
