"""Executable training-free competition VQA pipeline."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha1
from time import perf_counter
from uuid import uuid4

from hcmai.common.config import VQAConfig
from hcmai.common.schemas import (
    PipelineTrace,
    StageStatus,
    TaskType,
    VQARetrievalEvidence,
    VQARequest,
    VQAResponse,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.llm.pipeline import LLMService
from hcmai.common.observability.tracing import StageTimer, log_stage
from hcmai.orchestration.workflows.base import (
    TaskPipelineDependencyError,
    TaskPipelineRequestError,
)
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.pipelines.vqa.domain.models import EvidenceBundle
from hcmai.pipelines.vqa.legacy_localization.candidates import retrieve_candidates
from hcmai.pipelines.vqa.legacy_localization.localizer import SimilarityLocalizer
from hcmai.pipelines.vqa.legacy_localization.video_aggregation import aggregate_videos
from hcmai.pipelines.vqa.legacy_localization.windows import build_windows
from hcmai.pipelines.vqa.output.ranking import rank_grounded_answers
from hcmai.pipelines.vqa.output.submission import materialize_submissions
from hcmai.pipelines.vqa.query.parser import parse_vqa_query
from hcmai.pipelines.vqa.reasoning.answerer import answer_windows
from hcmai.pipelines.vqa.reasoning.evidence import (
    build_evidence_bundle,
    select_question_evidence,
)
from hcmai.temporal import TemporalEvidenceCore

logger = get_logger(__name__)


class VQAPipeline:
    """Retrieve, localize, answer, and rank grounded VQA candidates."""

    task_type = TaskType.VQA

    def __init__(
        self,
        data: DataService | None,
        retrieval: RetrievalService | None,
        llm: LLMService | None,
        config: VQAConfig,
        temporal_core: TemporalEvidenceCore | None = None,
    ) -> None:
        """Initialize the VQA head and optional shared temporal core."""

        self.data = data
        self.retrieval = retrieval
        self.llm = llm
        self.config = config
        self.temporal_core = temporal_core

    def execute(self, request: VQARequest) -> VQAResponse:
        """Localize from hints, select question evidence, and answer scenes."""

        if (
            not isinstance(request, VQARequest)
            or request.query_type is not TaskType.VQA
        ):
            raise TaskPipelineRequestError(
                "VQAPipeline requires a competition VQARequest"
            )
        if self.data is None:
            raise TaskPipelineDependencyError("Frame store not loaded")
        if self.retrieval is None:
            raise TaskPipelineDependencyError("Retriever not loaded")

        started = perf_counter()
        request_id = _request_id(request)
        profile = self.config.profiles[request.baseline_profile]
        traces = []

        timer = StageTimer("parse")
        parsed = parse_vqa_query(request)
        traces.append(timer.finish(input_count=1, output_count=1, backend="rules"))

        timer = StageTimer("search")
        progressive = None
        if self.temporal_core is not None:
            progressive = self.temporal_core.localize(
                request.event_description,
                search_id=request.search_id,
                filters=request.filters,
                task_type=TaskType.VQA,
                session_fingerprint=_question_fingerprint(request.question),
            )
            scenes = list(progressive.scenes[: profile.max_windows])
            candidates = [item for scene in scenes for item in scene.evidence]
            warnings = list(progressive.warnings)
            evidence_candidates = _scene_retrieval_evidence(scenes, request.top_k)
            windows = [
                _scene_bundle(scene, self.data, profile.max_frames_per_window)
                for scene in scenes
            ]
            budgets = progressive.diagnostics
            search_backend = (
                f"TemporalEvidenceCore:{progressive.diff.mode.value};"
                f"candidate_pool_size={budgets['candidate_pool_size']};"
                f"top_m_evidence={budgets['top_m_evidence']};"
                f"scene_top_p_global={budgets['scene_top_p_global']}"
            )
        else:
            candidates, warnings = retrieve_candidates(
                self.retrieval,
                self.data,
                parsed,
                top_k=profile.candidates_per_branch,
                filters=request.filters,
            )
            evidence_candidates = _retrieval_evidence(candidates, request.top_k)
            videos = aggregate_videos(candidates, top_videos=profile.candidate_videos)
            windows = build_windows(
                videos,
                self.data,
                duration_ms=profile.window_ms,
                max_frames=profile.max_frames_per_window,
            )[: profile.max_windows]
            search_backend = type(self.retrieval).__name__
        traces.append(timer.finish(
            input_count=1,
            output_count=len(candidates),
            backend=search_backend,
        ))

        timer = StageTimer("video_aggregation")
        traces.append(timer.finish(
            input_count=len(candidates),
            output_count=len(scenes) if progressive is not None else len(videos),
            backend="shared_scenes" if progressive is not None else "coverage",
        ))

        timer = StageTimer("window_construction")
        traces.append(timer.finish(
            input_count=len(candidates), output_count=len(windows),
            backend="shared_scenes" if progressive is not None else "neighbors",
        ))

        timer = StageTimer("evidence_construction")
        # The question first influences processing here, after the shared core
        # has already selected scene boundaries from event hints alone.
        bundles = [
            select_question_evidence(
                build_evidence_bundle(
                    window,
                    self.data,
                    max_items=profile.max_evidence_items,
                ),
                parsed,
                max_items=profile.max_evidence_items,
            )
            for window in windows
        ]
        traces.append(timer.finish(
            input_count=len(windows), output_count=len(bundles), backend="local_stores"
        ))

        timer = StageTimer("localization")
        if progressive is not None:
            localized = bundles
        elif profile.localizer_enabled:
            localized = SimilarityLocalizer().localize(
                parsed, bundles, limit=profile.max_windows
            )
        else:
            localized = [
                replace(
                    bundle,
                    scene=bundle.scene.model_copy(update={
                        "reason_labels": ("retrieval_order",),
                    }),
                )
                for bundle in bundles
            ]
        traces.append(timer.finish(
            input_count=len(bundles), output_count=len(localized),
            backend=(
                "shared_temporal_core" if progressive is not None
                else "similarity" if profile.localizer_enabled else "retrieval"
            ),
        ))

        timer = StageTimer("answer")
        answer_warnings: list[str] = []
        answers = []
        if self.llm is not None and profile.max_vlm_calls > 0:
            answers, answer_warnings = answer_windows(
                localized,
                parsed,
                self.llm,
                max_calls=profile.max_vlm_calls,
                data=self.data,
            )
            answer_status = (
                StageStatus.PARTIAL if answer_warnings else StageStatus.SUCCESS
            )
            fallback_used = bool(answer_warnings)
        else:
            answer_warnings = [
                "vqa_answerer_unavailable; returning retrieval evidence"
            ]
            answer_status = StageStatus.SKIPPED
            fallback_used = True
        traces.append(timer.finish(
            status=answer_status,
            attempt_count=min(profile.max_vlm_calls, len(localized)) if self.llm else 0,
            input_count=len(localized),
            output_count=len(answers),
            backend=type(self.llm.adapter).__name__ if self.llm is not None else None,
            fallback_used=fallback_used,
        ))

        timer = StageTimer("joint_ranking")
        ranked = rank_grounded_answers(answers)
        traces.append(timer.finish(
            input_count=len(answers), output_count=len(ranked), backend="linear"
        ))

        timer = StageTimer("materialization")
        submissions = materialize_submissions(ranked, self.data, top_k=request.top_k)
        traces.append(timer.finish(
            input_count=len(ranked), output_count=len(submissions),
            backend="canonical_frame_store",
        ))

        trace = PipelineTrace(stages={item.stage: item for item in traces})
        if progressive is not None:
            trace = trace.merged(progressive.trace, prefix="temporal_retrieval")
        for item in traces:
            log_stage(
                logger,
                request_id=request_id,
                task_type=TaskType.VQA,
                trace=item,
            )
        return VQAResponse(
            request_id=request_id,
            search_id=(
                progressive.search_id
                if progressive is not None
                else request.search_id
            ),
            event_description=request.event_description,
            question=request.question,
            top_k=request.top_k,
            total_results=len(submissions),
            submissions=submissions,
            evidence_candidates=evidence_candidates,
            warnings=list(dict.fromkeys([*warnings, *answer_warnings])),
            latency_ms=max(0, int((perf_counter() - started) * 1_000)),
            trace=trace,
        )


def _request_id(request: VQARequest) -> str:
    """Build a unique identifier for one VQA pipeline invocation."""

    return f"request-{uuid4().hex}"


def _question_fingerprint(question: str) -> str:
    """Bind progressive scene state to one normalized VQA question."""

    normalized = " ".join(question.split()).casefold().encode()
    return sha1(normalized).hexdigest()


def _retrieval_evidence(candidates, top_k: int) -> list[VQARetrievalEvidence]:
    """Materialize canonical fallback evidence from legacy frame candidates."""

    return [
        VQARetrievalEvidence(
            rank=rank,
            video_id=candidate.frame.video_id,
            frame_id=candidate.frame.frame_id,
            frame_idx=candidate.frame.frame_idx,
            timestamp_ms=candidate.frame.timestamp_ms,
            retrieval_score=candidate.score,
        )
        for rank, candidate in enumerate(candidates[:top_k], start=1)
    ]


def _scene_retrieval_evidence(scenes, top_k: int) -> list[VQARetrievalEvidence]:
    """Expose bounded canonical frame evidence from ranked shared scenes."""

    rows = []
    seen = set()
    for scene in scenes:
        for item in sorted(scene.evidence, key=lambda value: -value.score):
            if item.frame.frame_id in seen:
                continue
            seen.add(item.frame.frame_id)
            rows.append(VQARetrievalEvidence(
                rank=len(rows) + 1,
                video_id=item.frame.video_id,
                frame_id=item.frame.frame_id,
                frame_idx=item.frame.frame_idx,
                timestamp_ms=item.frame.timestamp_ms,
                retrieval_score=scene.final_score,
            ))
            if len(rows) >= top_k:
                return rows
    return rows


def _scene_bundle(scene, data: DataService, max_frames: int) -> EvidenceBundle:
    """Materialize bounded chronological images strictly inside a ranked scene."""

    by_id = {item.frame.frame_id: item.frame for item in scene.evidence}
    for item in scene.evidence:
        for frame in data.neighbors(
            item.frame.frame_id,
            window_ms=max(0, scene.end_ms - scene.start_ms),
            include_self=True,
        ):
            if (
                frame.video_id == scene.video_id
                and scene.start_ms <= frame.timestamp_ms <= scene.end_ms
            ):
                by_id[frame.frame_id] = frame
    ordered = sorted(
        by_id.values(),
        key=lambda frame: (frame.timestamp_ms, frame.frame_idx, frame.frame_id),
    )
    if len(ordered) > max_frames:
        positions = {
            round(index * (len(ordered) - 1) / (max_frames - 1))
            for index in range(max_frames)
        } if max_frames > 1 else {len(ordered) // 2}
        ordered = [ordered[index] for index in sorted(positions)]
    return EvidenceBundle(scene=scene, image_frames=tuple(ordered))
