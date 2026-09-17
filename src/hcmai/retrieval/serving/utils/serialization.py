"""Conversion helpers between internal domain objects and wire schemas."""

from __future__ import annotations

import numpy as np

from hcmai.corpus import Corpus
from hcmai.orchestration.workflows.temporal_search import (
    DecoderConfigSnapshot,
    TemporalSearchArtifact,
    TemporalSearchResult,
)
from hcmai.retrieval.plan import (
    KISImageRef,
    KISRetrievalEvent,
    KISRetrievalPlan,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.retrieval.serving.schemas import (
    AlignedPathSchema,
    DecoderConfigSchema,
    RetrievalEventSchema,
    SearchPlanRequestSchema,
    TemporalSearchArtifactSchema,
    TemporalSearchResultSchema,
    VideoScoresSchema,
)
from hcmai.temporal.dp import AlignedPath

_TRANSPORT_IMAGE_CONTENT_TYPE = "image/jpeg"


def plan_to_schema(
    plan: KISRetrievalPlan,
    *,
    use_dense: bool = True,
    use_bm25: bool = False,
    top_k: int = 20,
) -> SearchPlanRequestSchema:
    """Convert domain KISRetrievalPlan to serializable request schema."""
    if not isinstance(plan, KISRetrievalPlan):
        raise ValueError("plan must be a KISRetrievalPlan")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    plan.validate_text_sources(use_dense=use_dense, use_bm25=use_bm25)

    events = [
        RetrievalEventSchema(
            event_id=event.event_id,
            canonical_text=event.canonical_text,
            dense_text=event.dense_text,
            bm25_text=event.bm25_text,
            image_asset_ids=[ref.asset_id for ref in event.image_refs],
        )
        for event in plan.events
    ]
    return SearchPlanRequestSchema(
        events=events,
        use_dense=use_dense,
        use_bm25=use_bm25,
        top_k=top_k,
    )


def schema_to_plan(schema: SearchPlanRequestSchema) -> KISRetrievalPlan:
    """Convert request schema back to domain KISRetrievalPlan."""
    events = tuple(
        KISRetrievalEvent(
            event_id=e.event_id,
            canonical_text=e.canonical_text,
            dense_text=e.dense_text,
            bm25_text=e.bm25_text,
            image_refs=tuple(
                KISImageRef(asset_id=aid, content_type=_TRANSPORT_IMAGE_CONTENT_TYPE)
                for aid in e.image_asset_ids
            ),
        )
        for e in schema.events
    )
    plan = KISRetrievalPlan(events=events)
    plan.validate_text_sources(use_dense=schema.use_dense, use_bm25=schema.use_bm25)
    return plan


def video_scores_to_schema(video: VideoEventScores) -> VideoScoresSchema:
    """Convert domain VideoEventScores to serializable schema."""
    scores_array = np.asarray(video.scores, dtype=np.float32)
    return VideoScoresSchema(
        video_id=video.video_id,
        frame_ids=[str(fid) for fid in video.frame_ids],
        frame_idx=[int(idx) for idx in video.frame_idx],
        timestamps_ms=[int(ts) for ts in video.timestamps_ms],
        scores=scores_array.tolist(),
    )


def schema_to_video_scores(
    schema: VideoScoresSchema,
    corpus: Corpus | None = None,
) -> VideoEventScores:
    """Convert VideoScoresSchema back to domain VideoEventScores."""
    frame_ids = np.asarray([str(fid) for fid in schema.frame_ids], dtype=object)
    frame_idx = np.asarray(schema.frame_idx, dtype=np.int64)
    timestamps_ms = np.asarray(schema.timestamps_ms, dtype=np.int64)
    scores = np.asarray(schema.scores, dtype=np.float32)

    if corpus is not None:
        for fid, fidx, ts in zip(frame_ids, frame_idx, timestamps_ms, strict=True):
            frame = corpus.frame(str(fid))
            if frame.video_id != schema.video_id:
                raise ValueError("score video_id conflicts with corpus")
            if frame.frame_idx != int(fidx):
                raise ValueError("score frame_idx conflicts with corpus")
            if frame.timestamp_ms != int(ts):
                raise ValueError("score timestamp conflicts with corpus")

    for arr in (frame_ids, frame_idx, timestamps_ms, scores):
        arr.setflags(write=False)

    return VideoEventScores(
        video_id=schema.video_id,
        frame_ids=frame_ids,
        frame_idx=frame_idx,
        timestamps_ms=timestamps_ms,
        scores=scores,
    )


def path_to_schema(path: AlignedPath) -> AlignedPathSchema:
    """Convert domain AlignedPath to schema."""
    return AlignedPathSchema(
        video_id=path.video_id,
        score=path.score,
        frame_ids=[str(fid) for fid in path.frame_ids],
        frame_idxs=[int(fidx) for fidx in path.frame_idxs],
        timestamps_ms=[int(ts) for ts in path.timestamps_ms],
    )


def schema_to_path(schema: AlignedPathSchema) -> AlignedPath:
    """Convert AlignedPathSchema back to domain AlignedPath."""
    return AlignedPath(
        video_id=schema.video_id,
        score=schema.score,
        frame_ids=tuple(str(fid) for fid in schema.frame_ids),
        frame_idxs=tuple(int(fidx) for fidx in schema.frame_idxs),
        timestamps_ms=tuple(int(ts) for ts in schema.timestamps_ms),
    )


def artifact_to_schema(artifact: TemporalSearchArtifact) -> TemporalSearchArtifactSchema:
    """Convert domain TemporalSearchArtifact to schema."""
    return TemporalSearchArtifactSchema(
        result=TemporalSearchResultSchema(
            paths=[path_to_schema(p) for p in artifact.result.paths],
            retrieval_ms=artifact.result.retrieval_ms,
            alignment_ms=artifact.result.alignment_ms,
        ),
        video_scores=[video_scores_to_schema(v) for v in artifact.video_scores],
        decoder_config=DecoderConfigSchema(
            lambda_gap=artifact.decoder_config.lambda_gap,
            event_power=artifact.decoder_config.event_power,
            cluster_delta=artifact.decoder_config.cluster_delta,
            path_min_separation_ms=artifact.decoder_config.path_min_separation_ms,
        ),
        scoring_revision=artifact.scoring_revision,
    )


def schema_to_artifact(
    schema: TemporalSearchArtifactSchema,
    corpus: Corpus | None = None,
) -> TemporalSearchArtifact:
    """Convert TemporalSearchArtifactSchema back to domain TemporalSearchArtifact."""
    paths = tuple(schema_to_path(p) for p in schema.result.paths)
    result = TemporalSearchResult(
        paths=paths,
        retrieval_ms=schema.result.retrieval_ms,
        alignment_ms=schema.result.alignment_ms,
    )
    video_scores = tuple(schema_to_video_scores(v, corpus) for v in schema.video_scores)
    decoder_config = DecoderConfigSnapshot(
        lambda_gap=schema.decoder_config.lambda_gap,
        event_power=schema.decoder_config.event_power,
        cluster_delta=schema.decoder_config.cluster_delta,
        path_min_separation_ms=schema.decoder_config.path_min_separation_ms,
    )
    return TemporalSearchArtifact(
        result=result,
        video_scores=video_scores,
        decoder_config=decoder_config,
        scoring_revision=schema.scoring_revision,
    )
