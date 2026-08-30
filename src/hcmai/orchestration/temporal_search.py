"""Timed orchestration for shared ordered temporal search.

This module owns the task-agnostic runtime facade that scores event text,
validates retrieval metadata against canonical frame records, and materializes
canonical aligned paths. It does not shape KIS or TRAKE HTTP responses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from hcmai.common.config import AlignmentConfig
from hcmai.data.pipeline import DataService
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import AlignedPath, DPPath, rank_paths

if TYPE_CHECKING:
    from hcmai.retrieval.retriever.pipeline import RetrievalService


@dataclass(frozen=True, slots=True)
class TemporalSearchResult:
    """Shared temporal-search output and stage timings for one request."""

    paths: tuple[AlignedPath, ...]
    retrieval_ms: float
    alignment_ms: float


class TemporalSearchService:
    """Score ordered events, decode monotonic paths, and preserve identity."""

    def __init__(
        self,
        data: DataService,
        retrieval: RetrievalService,
        config: AlignmentConfig,
    ) -> None:
        """Bind canonical data access, retrieval scoring, and DP settings."""

        self.data = data
        self.retrieval = retrieval
        self.config = config

    def search(
        self,
        events: Sequence[str],
        *,
        top_k: int,
    ) -> TemporalSearchResult:
        """Return canonical aligned paths for one ordered event sequence."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        normalized = tuple(
            " ".join(event.split()) for event in events if event.strip()
        )
        if not normalized:
            raise ValueError("events must not be empty")

        retrieval_started = perf_counter()
        scores = self.retrieval.score_event_videos(
            normalized,
            chunk_size=self.config.chunk_size,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1_000

        score_by_video: dict[str, VideoEventScores] = {}
        for video in scores:
            self._validate_video_scores(len(normalized), video)
            score_by_video[video.video_id] = video

        alignment_started = perf_counter()
        rows = rank_paths(
            scores,
            lambda_gap=self.config.lambda_gap,
            max_rows=top_k,
            event_power=self.config.event_power,
            cluster_delta=self.config.cluster_delta,
        )
        paths = tuple(
            self._materialize_aligned_path(row, score_by_video[row.video_id])
            for row in rows
        )
        alignment_ms = (perf_counter() - alignment_started) * 1_000
        return TemporalSearchResult(
            paths=paths,
            retrieval_ms=retrieval_ms,
            alignment_ms=alignment_ms,
        )

    def _validate_video_scores(
        self,
        event_count: int,
        video: VideoEventScores,
    ) -> None:
        """Reject score metadata that conflicts with canonical frame records."""

        frame_count = len(video.frame_ids)
        if video.scores.shape != (event_count, frame_count):
            raise ValueError("dense score matrix shape does not match event input")
        if not (
            len(video.frame_idx) == frame_count
            and len(video.timestamps_ms) == frame_count
        ):
            raise ValueError("dense score metadata arrays must have equal lengths")

        for position, frame_id in enumerate(video.frame_ids):
            canonical_frame_id = str(frame_id)
            frame = self.data.get_frame(canonical_frame_id)
            if frame.video_id != video.video_id:
                raise ValueError("dense score frame has mixed canonical video identity")
            if frame.frame_id != canonical_frame_id:
                raise ValueError("dense score frame_id conflicts with canonical data")
            if frame.frame_idx != int(video.frame_idx[position]):
                raise ValueError("dense score frame_idx conflicts with canonical data")
            if frame.timestamp_ms != round(float(video.timestamps_ms[position])):
                raise ValueError("dense score timestamp conflicts with canonical data")

    def _materialize_aligned_path(
        self,
        row: DPPath,
        video: VideoEventScores,
    ) -> AlignedPath:
        """Resolve one decoded DP row into canonical frame indices and times."""

        positions = {
            str(frame_id): position
            for position, frame_id in enumerate(video.frame_ids)
        }
        frame_idxs: list[int] = []
        timestamps_ms: list[int] = []

        for frame_id, frame_idx in zip(row.frame_ids, row.frame_idx, strict=True):
            position = positions.get(frame_id)
            if position is None:
                raise ValueError("decoded path frame_id is missing from score metadata")
            if int(video.frame_idx[position]) != frame_idx:
                raise ValueError("decoded path frame_idx conflicts with score metadata")

            frame = self.data.get_frame(frame_id)
            timestamp_ms = round(float(video.timestamps_ms[position]))
            if frame.video_id != row.video_id:
                raise ValueError("decoded path frame has mixed canonical video identity")
            if frame.frame_id != frame_id:
                raise ValueError("decoded path frame_id conflicts with canonical data")
            if frame.frame_idx != frame_idx:
                raise ValueError("decoded path frame_idx conflicts with canonical data")
            if frame.timestamp_ms != timestamp_ms:
                raise ValueError("decoded path timestamp conflicts with canonical data")

            frame_idxs.append(frame.frame_idx)
            timestamps_ms.append(frame.timestamp_ms)

        return AlignedPath(
            video_id=row.video_id,
            score=row.score,
            frame_ids=row.frame_ids,
            frame_idxs=tuple(frame_idxs),
            timestamps_ms=tuple(timestamps_ms),
        )
