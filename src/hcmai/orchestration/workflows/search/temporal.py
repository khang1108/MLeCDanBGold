"""Timed orchestration for shared ordered temporal search.

This module owns the task-agnostic runtime facade that scores event text,
validates retrieval metadata against canonical frame records, and materializes
canonical aligned paths. It does not shape KIS or TRAKE HTTP responses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol
import numpy as np

from hcmai.common.config import AlignmentConfig, DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.corpus import Corpus
from hcmai.orchestration.utils.materializer import SearchMaterializer
from hcmai.retrieval.evidence.components import TemporalScoreComponent
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
from hcmai.retrieval.plan import KISRetrievalPlan
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.events import normalize_event_texts
from hcmai.temporal.dp import AlignedPath, DPPath, align_video, align_video_conditioned, rank_paths


@dataclass(frozen=True, slots=True)
class EventConditionedPath:
    """Decoded temporal path conditioned on one focused event position."""

    path: AlignedPath
    focus_frame_position: int
    focus_timestamp_ms: int
    score: float

@dataclass(frozen=True, slots=True)
class TemporalSearchResult:
    """Shared temporal-search output and stage timings for one request."""

    paths: tuple[AlignedPath, ...]
    retrieval_ms: float
    alignment_ms: float


@dataclass(frozen=True, slots=True)
class DecoderConfigSnapshot:
    """Immutable values that determine selected-video decoder semantics."""

    lambda_gap: float
    event_power: float
    cluster_delta: float
    path_min_separation_ms: int


@dataclass(frozen=True, slots=True)
class SelectedVideoScoreResult:
    """Scored frames and decoder settings for one selected video."""

    video: VideoEventScores
    retrieval_ms: float
    decoder_config: DecoderConfigSnapshot
    scoring_revision: str | None = None


@dataclass(frozen=True, slots=True)
class TemporalSearchArtifact:
    """Shared score matrix and paths, optionally tagged with a serving revision."""

    result: TemporalSearchResult
    video_scores: tuple[VideoEventScores, ...]
    decoder_config: DecoderConfigSnapshot
    scoring_revision: str | None = None


class TemporalSearchGateway(Protocol):
    """High-level gateway protocol for temporal retrieval and alignment."""

    corpus: Corpus

    def search(
        self,
        original_events: Sequence[str],
        *,
        top_k: int,
        retrieval_events: Sequence[str] | None = None,
        caption_events: Sequence[str] | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> TemporalSearchResult: ...

    def search_plan_artifact(
        self,
        plan: KISRetrievalPlan,
        *,
        image_component: TemporalScoreComponent | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
        top_k: int = 20,
    ) -> TemporalSearchArtifact: ...

    def score_video(
        self,
        plan: KISRetrievalPlan,
        *,
        video_id: str,
        image_component: TemporalScoreComponent | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> SelectedVideoScoreResult: ...

    def decode_video(
        self,
        video: VideoEventScores,
        *,
        allowed: np.ndarray,
        decoder_config: DecoderConfigSnapshot | None = None,
    ) -> tuple[AlignedPath, ...]: ...



class TemporalSearchService:
    """Score ordered events, decode monotonic paths, and preserve identity."""

    def __init__(
        self,
        corpus: Corpus,
        evidence: TemporalEvidenceScorer,
        config: AlignmentConfig,
        max_temporal_event_count: int = DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    ) -> None:
        """Bind canonical data access, retrieval scoring, and DP settings."""

        self.corpus = corpus
        self.materializer = SearchMaterializer(corpus)
        self.evidence = evidence
        self.config = config
        self.max_temporal_event_count = max_temporal_event_count

    def search(
        self,
        original_events: Sequence[str],
        *,
        top_k: int,
        retrieval_events: Sequence[str] | None = None,
        caption_events: Sequence[str] | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> TemporalSearchResult:
        """Return canonical aligned paths for one ordered event sequence."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        scores, retrieval_ms = self.score_videos(
            original_events,
            retrieval_events=retrieval_events,
            caption_events=caption_events,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )
        score_by_video = {video.video_id: video for video in scores}

        alignment_started = perf_counter()
        rows = rank_paths(
            scores,
            lambda_gap=self.config.lambda_gap,
            max_rows=top_k,
            event_power=self.config.event_power,
            cluster_delta=self.config.cluster_delta,
            paths_per_video=self.config.paths_per_video,
            path_min_separation_ms=self.config.path_min_separation_ms,
        )

        paths = tuple(
            self._materialize_aligned_path(row, score_by_video[row.video_id]) for row in rows
        )
        alignment_ms = (perf_counter() - alignment_started) * 1_000
        return TemporalSearchResult(
            paths=paths,
            retrieval_ms=retrieval_ms,
            alignment_ms=alignment_ms,
        )

    def search_plan(
        self,
        plan: KISRetrievalPlan,
        *,
        image_component: TemporalScoreComponent | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
        top_k: int = 20,
    ) -> TemporalSearchResult:
        """Return canonical aligned paths for a multimodal retrieval plan."""
        return self.search_plan_artifact(
            plan,
            image_component=image_component,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=top_k,
        ).result

    def search_plan_artifact(
        self,
        plan: KISRetrievalPlan,
        *,
        image_component: TemporalScoreComponent | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
        top_k: int = 20,
    ) -> TemporalSearchArtifact:
        """Return canonical aligned paths and the exact score matrix artifact."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        scores, retrieval_ms = self.score_plan(
            plan,
            image_component=image_component,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )
        score_by_video = {video.video_id: video for video in scores}

        alignment_started = perf_counter()
        rows = rank_paths(
            scores,
            lambda_gap=self.config.lambda_gap,
            max_rows=top_k,
            event_power=self.config.event_power,
            cluster_delta=self.config.cluster_delta,
            paths_per_video=self.config.paths_per_video,
            path_min_separation_ms=self.config.path_min_separation_ms,
        )

        paths = tuple(
            self._materialize_aligned_path(row, score_by_video[row.video_id]) for row in rows
        )
        alignment_ms = (perf_counter() - alignment_started) * 1_000
        result = TemporalSearchResult(
            paths=paths,
            retrieval_ms=retrieval_ms,
            alignment_ms=alignment_ms,
        )
        return TemporalSearchArtifact(
            result=result,
            video_scores=scores,
            decoder_config=self.snapshot_decoder_config(),
        )

    def score_plan(
        self,
        plan: KISRetrievalPlan,
        *,
        image_component: TemporalScoreComponent | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> tuple[tuple[VideoEventScores, ...], float]:
        """Score and validate every video for a multimodal retrieval plan."""
        if plan.event_count > self.max_temporal_event_count:
            raise ValueError(
                f"requests may contain at most {self.max_temporal_event_count} temporal events"
            )

        retrieval_started = perf_counter()
        scores = self.evidence.score_plan(
            plan,
            image_component=image_component,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1_000

        validated = tuple(scores)
        for video in validated:
            self._validate_video_scores(plan.event_count, video)
        return validated, retrieval_ms

    def score_videos(
        self,
        original_events: Sequence[str],
        *,
        retrieval_events: Sequence[str] | None = None,
        caption_events: Sequence[str] | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> tuple[tuple[VideoEventScores, ...], float]:
        """Score and validate every video for one normalized event sequence.

        Scoring time excludes canonical metadata validation so callers retain
        the existing retrieval latency semantics while safely reusing scores.
        """

        original = normalize_event_texts(original_events)
        if len(original) > self.max_temporal_event_count:
            raise ValueError(
                f"requests may contain at most {self.max_temporal_event_count} temporal events"
            )
        retrieval = (
            original
            if retrieval_events is None
            else normalize_event_texts(retrieval_events)
        )
        captions = (
            None
            if caption_events is None
            else normalize_event_texts(caption_events)
        )
        if len(retrieval) != len(original):
            raise ValueError("retrieval_events must match the original event count")
        if captions is not None and len(captions) != len(original):
            raise ValueError("caption_events must match the original event count")
        if not use_dense and not use_bm25:
            raise ValueError("at least one retrieval source must be enabled")

        retrieval_started = perf_counter()
        scores = self.evidence.score_events(
            original,
            retrieval,
            caption_events=captions,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1_000

        validated = tuple(scores)
        for video in validated:
            self._validate_video_scores(len(original), video)
        return validated, retrieval_ms

    def score_video(
        self,
        plan: KISRetrievalPlan,
        *,
        video_id: str,
        image_component: TemporalScoreComponent | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> SelectedVideoScoreResult:
        """Score a multimodal retrieval plan and return scores for one video."""
        scores, retrieval_ms = self.score_plan(
            plan,
            image_component=image_component,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )
        selected = next((v for v in scores if v.video_id == video_id), None)
        if selected is None:
            raise KeyError(f"video {video_id!r} was not found in scored plan")
        return SelectedVideoScoreResult(
            video=selected,
            retrieval_ms=retrieval_ms,
            decoder_config=self.snapshot_decoder_config(),
            scoring_revision=None,
        )

    def decode_video(
        self,
        video: VideoEventScores,
        *,
        allowed: np.ndarray,
        decoder_config: DecoderConfigSnapshot | None = None,
    ) -> tuple[AlignedPath, ...]:
        """Decode one scored video under a mask and optional config snapshot.

        Without a snapshot, existing callers continue to use the service's
        current alignment configuration.
        """

        config = (
            self.snapshot_decoder_config()
            if decoder_config is None
            else decoder_config
        )
        return decode_video_scores(
            self.corpus,
            video,
            allowed=allowed,
            decoder_config=config,
        )

    def decode_event_alternatives(
        self,
        video: VideoEventScores,
        *,
        allowed: np.ndarray,
        focus_event_index: int,
        max_paths: int,
        min_separation_ms: int,
        decoder_config: DecoderConfigSnapshot | None = None,
    ) -> tuple[EventConditionedPath, ...]:
        """Decode complete chronological paths conditioned on one focused event."""
        config = self.snapshot_decoder_config() if decoder_config is None else decoder_config
        conditioned = align_video_conditioned(
            video,
            focus_event_index,
            allowed=allowed,
            lambda_gap=config.lambda_gap,
            event_power=config.event_power,
            cluster_delta=config.cluster_delta,
            max_paths=max_paths,
            min_separation_ms=min_separation_ms,
        )
        return tuple(
            EventConditionedPath(
                path=self._materialize_aligned_path(item.path, video),
                focus_frame_position=item.focus_frame_position,
                focus_timestamp_ms=item.focus_timestamp_ms,
                score=item.score,
            )
            for item in conditioned
        )

    def snapshot_decoder_config(self) -> DecoderConfigSnapshot:
        """Copy all selected-video decoder settings into an immutable value."""

        return DecoderConfigSnapshot(
            lambda_gap=self.config.lambda_gap,
            event_power=self.config.event_power,
            cluster_delta=self.config.cluster_delta,
            path_min_separation_ms=self.config.path_min_separation_ms,
        )

    def _validate_video_scores(
        self,
        event_count: int,
        video: VideoEventScores,
    ) -> None:
        """Reject score metadata that conflicts with canonical frame records."""

        frame_count = len(video.frame_ids)
        if video.scores.shape != (event_count, frame_count):
            raise ValueError("temporal score matrix shape does not match event input")
        if not (len(video.frame_idx) == frame_count and len(video.timestamps_ms) == frame_count):
            raise ValueError("temporal score metadata arrays must have equal lengths")

        for position, frame_id in enumerate(video.frame_ids):
            canonical_frame_id = str(frame_id)
            frame = self.corpus.frame(canonical_frame_id)
            if frame.video_id != video.video_id:
                raise ValueError("temporal score frame has mixed canonical video identity")
            if frame.frame_id != canonical_frame_id:
                raise ValueError("temporal score frame_id conflicts with canonical data")
            if frame.frame_idx != int(video.frame_idx[position]):
                raise ValueError("temporal score frame_idx conflicts with canonical data")
            if frame.timestamp_ms != round(float(video.timestamps_ms[position])):
                raise ValueError("temporal score timestamp conflicts with canonical data")

    def _materialize_aligned_path(
        self,
        row: DPPath,
        video: VideoEventScores,
    ) -> AlignedPath:
        """Resolve one decoded DP row into canonical frame indices and times."""
        return _materialize_aligned_path(self.materializer, row, video)


def decode_video_scores(
    corpus: Corpus,
    video: VideoEventScores,
    *,
    allowed: np.ndarray,
    decoder_config: DecoderConfigSnapshot,
) -> tuple[AlignedPath, ...]:
    """Decode one scored video under a mask and explicit config snapshot."""
    rows = align_video(
        video,
        lambda_gap=decoder_config.lambda_gap,
        paths=1,
        event_power=decoder_config.event_power,
        cluster_delta=decoder_config.cluster_delta,
        min_separation_ms=decoder_config.path_min_separation_ms,
        allowed=allowed,
    )
    materializer = SearchMaterializer(corpus)
    return tuple(_materialize_aligned_path(materializer, row, video) for row in rows)


def _materialize_aligned_path(
    materializer: SearchMaterializer,
    row: DPPath,
    video: VideoEventScores,
) -> AlignedPath:
    """Resolve one decoded DP row into canonical frame indices and times."""
    positions = {str(frame_id): position for position, frame_id in enumerate(video.frame_ids)}
    frame_idxs: list[int] = []
    timestamps_ms: list[int] = []

    for frame_id, frame_idx in zip(row.frame_ids, row.frame_idx, strict=True):
        position = positions.get(frame_id)
        if position is None:
            raise ValueError("decoded path frame_id is missing from score metadata")
        if int(video.frame_idx[position]) != frame_idx:
            raise ValueError("decoded path frame_idx conflicts with score metadata")

        frame_idxs.append(frame_idx)
        timestamps_ms.append(round(float(video.timestamps_ms[position])))

    path = AlignedPath(
        video_id=row.video_id,
        score=row.score,
        frame_ids=row.frame_ids,
        frame_idxs=tuple(frame_idxs),
        timestamps_ms=tuple(timestamps_ms),
    )
    materializer.validate_aligned_path(path)
    return path
