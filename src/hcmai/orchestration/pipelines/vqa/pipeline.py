"""Executable training-free competition VQA pipeline."""

from __future__ import annotations

from hashlib import sha1
from time import perf_counter

from hcmai.common.config import SearchConfig, VQAConfig
from hcmai.common.schemas import (
    PipelineTrace,
    StageStatus,
    TaskRequest,
    TaskType,
    VQARetrievalEvidence,
    VQARequest,
    VQAResponse,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.llm.pipeline import LLMService
from hcmai.observability.tracing import StageTimer, log_stage
from hcmai.orchestration.pipelines.base import (
    TaskPipelineDependencyError,
    TaskPipelineRequestError,
)
from hcmai.orchestration.pipelines.vqa.temporal import localize_with_temporal_core
from hcmai.retriever.pipeline import RetrievalService
from hcmai.temporal.engine import TemporalEvidenceEngine
from hcmai.temporal.retrieval import SparseEvidenceProvider
from hcmai.temporal.settings import TemporalSettings
from hcmai.temporal.state import ProgressiveStateStore
from hcmai.vqa.answerer import answer_windows
from hcmai.vqa.candidates import retrieve_candidates
from hcmai.vqa.evidence import build_evidence_bundle
from hcmai.vqa.localizer import SimilarityLocalizer
from hcmai.vqa.models import GroundedAnswerCandidate, LocalizedWindow
from hcmai.vqa.parser import parse_vqa_query
from hcmai.vqa.ranking import rank_grounded_answers
from hcmai.vqa.submission import materialize_submissions
from hcmai.vqa.video_aggregation import aggregate_videos
from hcmai.vqa.windows import build_windows

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
        search_config: SearchConfig | None = None,
        temporal_settings: TemporalSettings | None = None,
    ) -> None:
        self.data = data
        self.retrieval = retrieval
        self.llm = llm
        self.config = config
        self.search_config = search_config or SearchConfig()
        self.temporal_settings = temporal_settings or TemporalSettings()
        self.temporal_engine: TemporalEvidenceEngine | None = None
        if (
            any(profile.temporal_core_enabled for profile in config.profiles.values())
            and data is not None
            and retrieval is not None
        ):
            self.temporal_engine = self.temporal_settings.engine(
                SparseEvidenceProvider(retrieval, data),
                ProgressiveStateStore(
                    ttl_seconds=self.search_config.progressive_state_ttl_seconds,
                    max_states=self.search_config.progressive_max_states,
                ),
            )

    def execute(self, request: TaskRequest) -> VQAResponse:
        if not isinstance(request, VQARequest) or request.query_type is not TaskType.VQA:
            raise TaskPipelineRequestError(
                "VQAPipeline requires a competition VQARequest"
            )
        if self.data is None:
            raise TaskPipelineDependencyError("Frame store not loaded")
        if self.retrieval is None:
            raise TaskPipelineDependencyError("Retriever not loaded")

        started = perf_counter()
        payload = f"vqa\0{request.event_description}\0{request.question}\0{request.top_k}"
        request_id = f"vqa-{sha1(payload.encode()).hexdigest()[:12]}"
        profile = self.config.profiles[request.baseline_profile]
        traces = []

        timer = StageTimer("parse")
        parsed = parse_vqa_query(request)
        traces.append(timer.finish(input_count=1, output_count=1, backend="rules"))

        search_id = None
        if profile.temporal_core_enabled and self.temporal_engine is not None:
            timer = StageTimer("localization")
            temporal = localize_with_temporal_core(
                self.temporal_engine,
                parsed,
                request.search_id,
                request.filters,
                request.top_k,
                self.data,
                self.retrieval,
                profile,
            )
            localized = temporal.localized
            evidence_candidates = temporal.evidence_candidates
            warnings = temporal.warnings
            search_id = temporal.search_id
            traces.append(timer.finish(
                input_count=1, output_count=len(localized), backend="temporal_core"
            ))
        else:
            timer = StageTimer("search")
            candidates, warnings = retrieve_candidates(
                self.retrieval,
                self.data,
                parsed,
                top_k=profile.candidates_per_branch,
                filters=request.filters,
            )
            traces.append(timer.finish(
                input_count=2 + len(parsed.clue_queries),
                output_count=len(candidates),
                backend=type(self.retrieval).__name__,
            ))
            evidence_candidates = [
                VQARetrievalEvidence(
                    rank=rank,
                    video_id=candidate.frame.video_id,
                    frame_id=candidate.frame.frame_id,
                    frame_idx=candidate.frame.frame_idx,
                    timestamp_ms=candidate.frame.timestamp_ms,
                    retrieval_score=candidate.score,
                )
                for rank, candidate in enumerate(candidates[: request.top_k], start=1)
            ]

            timer = StageTimer("video_aggregation")
            videos = aggregate_videos(candidates, top_videos=profile.candidate_videos)
            traces.append(timer.finish(
                input_count=len(candidates), output_count=len(videos), backend="coverage"
            ))

            timer = StageTimer("window_construction")
            windows = build_windows(
                videos,
                self.data,
                duration_ms=profile.window_ms,
                max_frames=profile.max_frames_per_window,
            )[: profile.max_windows]
            traces.append(timer.finish(
                input_count=len(videos), output_count=len(windows), backend="neighbors"
            ))

            timer = StageTimer("evidence_construction")
            bundles = [
                build_evidence_bundle(
                    window, self.data, max_items=profile.max_evidence_items
                )
                for window in windows
            ]
            traces.append(timer.finish(
                input_count=len(windows), output_count=len(bundles), backend="local_stores"
            ))

            timer = StageTimer("localization")
            if profile.localizer_enabled:
                localized = SimilarityLocalizer().localize(
                    parsed, bundles, limit=profile.max_windows
                )
            else:
                localized = [
                    LocalizedWindow(bundle, bundle.window.score, ("retrieval_order",))
                    for bundle in bundles
                ]
            traces.append(timer.finish(
                input_count=len(bundles), output_count=len(localized),
                backend="similarity" if profile.localizer_enabled else "retrieval",
            ))

        timer = StageTimer("answer")
        answer_warnings: list[str] = []
        answers: list[GroundedAnswerCandidate] = []
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
        for item in traces:
            log_stage(
                logger,
                request_id=request_id,
                task_type=TaskType.VQA,
                trace=item,
            )
        return VQAResponse(
            request_id=request_id,
            event_description=request.event_description,
            question=request.question,
            top_k=request.top_k,
            total_results=len(submissions),
            submissions=submissions,
            evidence_candidates=evidence_candidates,
            warnings=list(dict.fromkeys([*warnings, *answer_warnings])),
            latency_ms=max(0, int((perf_counter() - started) * 1_000)),
            trace=trace,
            search_id=search_id,
        )


