"""Stateless orchestration of event scoring, DP decoding, and canonical paths.

This module is the shared temporal facade for KIS and TRAKE. It validates that
retrieval score metadata agrees with canonical frame records before exposing an
``AlignmentPath``. It does not own HTTP response shaping or model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

from hcmai.common.config import AlignmentConfig
from hcmai.common.schemas import AlignmentPath, AlignmentPlan
from hcmai.data.pipeline import DataService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.video_scores import VideoEventScores

from .dp import rank_paths


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Canonical paths returned for one validated ordered alignment plan."""

    plan: AlignmentPlan
    paths: tuple[AlignmentPath, ...]
    candidate_video_count: int


class TemporalAlignmentService:
    """Run shared visual event alignment without request or session state."""

    def __init__(
        self,
        data: DataService,
        retrieval: RetrievalService,
        config: AlignmentConfig,
    ) -> None:
        """Bind canonical data, visual scoring, and explicit DP configuration."""

        self.data = data
        self.retrieval = retrieval
        self.config = config

    def align(self, plan: AlignmentPlan, *, max_paths: int) -> AlignmentResult:
        """Score plan events, decode paths, and materialize canonical frames.

        Every score-matrix column is checked against ``DataService`` before DP
        output becomes public. This prevents an index artifact from silently
        rewriting frame identity at the task boundary.
        """

        if max_paths <= 0:
            raise ValueError("max_paths must be greater than zero")
        if plan.filters is not None and plan.filters.min_score is not None:
            raise ValueError(
                "min_score is not supported for ordered event alignment"
            )

        scores = self.retrieval.score_event_videos(
            [event.text for event in plan.events],
            filters=plan.filters,
            top_k=self.config.top_k,
            max_videos=self.config.max_videos,
            rrf_k=self.config.rrf_k,
            chunk_size=self.config.chunk_size,
        )
        for video in scores:
            self._validate_video_scores(plan, video)

        rows = rank_paths(
            scores,
            lambda_gap=self.config.lambda_gap,
            max_rows=max_paths,
            event_power=self.config.event_power,
            cluster_delta=self.config.cluster_delta,
        )
        event_ids = tuple(event.event_id for event in plan.events)
        paths = tuple(
            AlignmentPath(
                path_id=self._path_id(row.video_id, row.frame_ids),
                video_id=row.video_id,
                frames=tuple(
                    self.data.get_frame(frame_id) for frame_id in row.frame_ids
                ),
                event_ids=event_ids,
                score=row.score,
            )
            for row in rows
        )
        return AlignmentResult(
            plan=plan,
            paths=paths,
            candidate_video_count=len(scores),
        )

    @staticmethod
    def _path_id(video_id: str, frame_ids: tuple[str, ...]) -> str:
        """Create a stable path identifier without deriving frame identity."""

        digest = sha1(f"{video_id}\0{'|'.join(frame_ids)}".encode()).hexdigest()[:16]
        return f"path-{digest}"

    def _validate_video_scores(
        self,
        plan: AlignmentPlan,
        video: VideoEventScores,
    ) -> None:
        """Reject score metadata that conflicts with canonical frame records."""

        frame_count = len(video.frame_ids)
        if video.scores.shape != (len(plan.events), frame_count):
            raise ValueError("dense score matrix shape does not match alignment plan")
        if not (
            len(video.frame_idx) == frame_count
            and len(video.timestamps_ms) == frame_count
        ):
            raise ValueError("dense score metadata arrays must have equal lengths")

        for position, frame_id in enumerate(video.frame_ids):
            frame = self.data.get_frame(str(frame_id))
            if frame.video_id != video.video_id:
                raise ValueError("dense score frame has mixed canonical video identity")
            if frame.frame_idx != int(video.frame_idx[position]):
                raise ValueError("dense score frame_idx conflicts with canonical data")
            if frame.timestamp_ms != round(float(video.timestamps_ms[position])):
                raise ValueError("dense score timestamp conflicts with canonical data")
