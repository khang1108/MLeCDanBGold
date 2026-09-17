"""Convert retrieval protobufs to validated HCMAI temporal domain values.

This module owns transport serialization and strict protocol validation.  It
does not score, align, materialize, or repair canonical corpus identity.
"""

from __future__ import annotations

import numpy as np

from hcmai.corpus import Corpus
from hcmai.kis.models import KISImageRef
from hcmai.orchestration.workflows.temporal_search import (
    DecoderConfigSnapshot,
    TemporalSearchArtifact,
    TemporalSearchResult,
)
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.retrieval_service.proto import retrieval_pb2
from hcmai.temporal.dp import AlignedPath


VIDEO_SCORES_ENCODING_VERSION = 1
_I64_LE = np.dtype("<i8")
_F32_LE = np.dtype("<f4")
_TRANSPORT_IMAGE_CONTENT_TYPE = "image/jpeg"


def encode_plan(
    plan: KISRetrievalPlan,
    *,
    use_dense: bool,
    use_bm25: bool,
    top_k: int,
) -> retrieval_pb2.SearchPlanRequest:
    """Encode a complete ordered KIS plan without flattening its text views."""

    if not isinstance(plan, KISRetrievalPlan):
        raise ValueError("plan must be a KISRetrievalPlan")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    plan.validate_text_sources(use_dense=use_dense, use_bm25=use_bm25)

    events = []
    for event in plan.events:
        message = retrieval_pb2.RetrievalEvent(
            event_id=event.event_id,
            image_asset_ids=[ref.asset_id for ref in event.image_refs],
        )
        if event.canonical_text is not None:
            message.canonical_text = event.canonical_text
        if event.dense_text is not None:
            message.dense_text = event.dense_text
        if event.bm25_text is not None:
            message.bm25_text = event.bm25_text
        events.append(message)
    return retrieval_pb2.SearchPlanRequest(
        events=events,
        use_dense=use_dense,
        use_bm25=use_bm25,
        top_k=top_k,
    )


def decode_plan(message: retrieval_pb2.SearchPlanRequest) -> KISRetrievalPlan:
    """Decode a plan while preserving optional string absence and E1..En order.

    The transport contract intentionally contains only image asset IDs.  The
    retrieval process opens those immutable assets itself, so a stable accepted
    media type is used solely to satisfy the existing domain image-reference
    contract; it is not used to interpret the asset bytes.
    """

    if message.top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    events = tuple(
        KISRetrievalEvent(
            event_id=event.event_id,
            canonical_text=(event.canonical_text if event.HasField("canonical_text") else None),
            dense_text=event.dense_text if event.HasField("dense_text") else None,
            bm25_text=event.bm25_text if event.HasField("bm25_text") else None,
            image_refs=tuple(
                KISImageRef(
                    asset_id=asset_id,
                    content_type=_TRANSPORT_IMAGE_CONTENT_TYPE,
                )
                for asset_id in event.image_asset_ids
            ),
        )
        for event in message.events
    )
    plan = KISRetrievalPlan(events=events)
    plan.validate_text_sources(use_dense=message.use_dense, use_bm25=message.use_bm25)
    return plan


def encode_video_scores(video: VideoEventScores) -> retrieval_pb2.VideoScores:
    """Encode one event-by-frame score matrix with explicit little-endian buffers."""

    frame_idxs = np.asarray(video.frame_idx, dtype=_I64_LE)
    timestamps = np.asarray(video.timestamps_ms, dtype=_I64_LE)
    scores = np.asarray(video.scores, dtype=_F32_LE, order="C")
    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional event-by-frame matrix")
    event_count, frame_count = scores.shape
    if len(video.frame_ids) != frame_count:
        raise ValueError("frame_ids must match score frame count")
    if len(frame_idxs) != frame_count or len(timestamps) != frame_count:
        raise ValueError("score metadata arrays must match score frame count")
    return retrieval_pb2.VideoScores(
        encoding_version=VIDEO_SCORES_ENCODING_VERSION,
        video_id=video.video_id,
        frame_ids=[str(value) for value in video.frame_ids],
        frame_idxs_i64_le=frame_idxs.tobytes(order="C"),
        timestamps_ms_i64_le=timestamps.tobytes(order="C"),
        scores_f32_le=scores.tobytes(order="C"),
        event_count=event_count,
        frame_count=frame_count,
    )


def decode_video_scores(
    message: retrieval_pb2.VideoScores,
    *,
    expected_event_count: int,
    corpus: Corpus,
) -> VideoEventScores:
    """Decode and validate immutable score buffers against canonical Corpus frames."""

    if message.encoding_version != VIDEO_SCORES_ENCODING_VERSION:
        raise ValueError("unsupported video score encoding version")
    if message.event_count != expected_event_count:
        raise ValueError("video score event count does not match the request")
    if len(message.frame_ids) != message.frame_count:
        raise ValueError("video score frame count does not match frame_ids")
    _require_byte_length(
        message.frame_idxs_i64_le,
        message.frame_count * _I64_LE.itemsize,
        "frame_idx byte length",
    )
    _require_byte_length(
        message.timestamps_ms_i64_le,
        message.frame_count * _I64_LE.itemsize,
        "timestamp byte length",
    )
    _require_byte_length(
        message.scores_f32_le,
        message.event_count * message.frame_count * _F32_LE.itemsize,
        "score byte length",
    )

    frame_ids = np.asarray(tuple(message.frame_ids))
    frame_idxs = np.frombuffer(message.frame_idxs_i64_le, dtype=_I64_LE).copy()
    timestamps = np.frombuffer(message.timestamps_ms_i64_le, dtype=_I64_LE).copy()
    scores = np.frombuffer(message.scores_f32_le, dtype=_F32_LE).copy().reshape(
        message.event_count, message.frame_count
    )
    _validate_video_identity(message.video_id, frame_ids, frame_idxs, timestamps, corpus)
    for values in (frame_ids, frame_idxs, timestamps, scores):
        values.setflags(write=False)
    return VideoEventScores(
        video_id=message.video_id,
        frame_ids=frame_ids,
        frame_idx=frame_idxs,
        timestamps_ms=timestamps,
        scores=scores,
    )


def encode_aligned_path(path: AlignedPath) -> retrieval_pb2.AlignedPath:
    """Encode one already-materialized canonical temporal path."""

    _validate_path_cardinality(path.frame_ids, path.frame_idxs, path.timestamps_ms)
    return retrieval_pb2.AlignedPath(
        video_id=path.video_id,
        score=path.score,
        frame_ids=path.frame_ids,
        frame_idxs=path.frame_idxs,
        timestamps_ms=path.timestamps_ms,
    )


def decode_aligned_path(
    message: retrieval_pb2.AlignedPath,
    *,
    corpus: Corpus,
) -> AlignedPath:
    """Decode one path only when every coordinate matches the Corpus authority."""

    frame_ids = tuple(message.frame_ids)
    frame_idxs = tuple(message.frame_idxs)
    timestamps = tuple(message.timestamps_ms)
    _validate_path_cardinality(frame_ids, frame_idxs, timestamps)
    _validate_video_identity(
        message.video_id,
        np.asarray(frame_ids),
        np.asarray(frame_idxs, dtype=np.int64),
        np.asarray(timestamps, dtype=np.int64),
        corpus,
    )
    return AlignedPath(
        video_id=message.video_id,
        score=message.score,
        frame_ids=frame_ids,
        frame_idxs=frame_idxs,
        timestamps_ms=timestamps,
    )


def encode_decoder_config(
    config: DecoderConfigSnapshot,
) -> retrieval_pb2.DecoderConfig:
    """Encode the immutable settings that determine selected-video decoding."""

    return retrieval_pb2.DecoderConfig(
        lambda_gap=config.lambda_gap,
        event_power=config.event_power,
        cluster_delta=config.cluster_delta,
        path_min_separation_ms=config.path_min_separation_ms,
    )


def decode_decoder_config(message: retrieval_pb2.DecoderConfig) -> DecoderConfigSnapshot:
    """Decode a selected-video decoder configuration without applying defaults."""

    return DecoderConfigSnapshot(
        lambda_gap=message.lambda_gap,
        event_power=message.event_power,
        cluster_delta=message.cluster_delta,
        path_min_separation_ms=message.path_min_separation_ms,
    )


def encode_temporal_search_result(
    result: TemporalSearchResult,
    *,
    scoring_revision: str | None = None,
) -> retrieval_pb2.TemporalSearchResult:
    """Encode aligned paths and timings, optionally attaching serving generation."""

    return retrieval_pb2.TemporalSearchResult(
        paths=[encode_aligned_path(path) for path in result.paths],
        retrieval_ms=result.retrieval_ms,
        alignment_ms=result.alignment_ms,
        scoring_revision=scoring_revision or "",
    )


def decode_temporal_search_result(
    message: retrieval_pb2.TemporalSearchResult,
    *,
    corpus: Corpus,
    expected_event_count: int | None = None,
) -> TemporalSearchResult:
    """Decode validated paths and timings from one temporal search response."""

    paths = tuple(decode_aligned_path(path, corpus=corpus) for path in message.paths)
    if expected_event_count is not None and any(
        len(path.frame_ids) != expected_event_count for path in paths
    ):
        raise ValueError("aligned path cardinality does not match event count")
    return TemporalSearchResult(
        paths=paths,
        retrieval_ms=message.retrieval_ms,
        alignment_ms=message.alignment_ms,
    )


def encode_temporal_search_artifact(
    artifact: TemporalSearchArtifact,
) -> retrieval_pb2.TemporalSearchArtifact:
    """Encode bounded score snapshots alongside their paths and decoder settings."""

    return retrieval_pb2.TemporalSearchArtifact(
        result=encode_temporal_search_result(
            artifact.result,
            scoring_revision=artifact.scoring_revision,
        ),
        video_scores=[encode_video_scores(video) for video in artifact.video_scores],
        decoder_config=encode_decoder_config(artifact.decoder_config),
    )


def decode_temporal_search_artifact(
    message: retrieval_pb2.TemporalSearchArtifact,
    *,
    expected_event_count: int,
    corpus: Corpus,
) -> TemporalSearchArtifact:
    """Decode a complete bounded artifact while retaining its serving generation."""

    if not message.HasField("result"):
        raise ValueError("temporal search artifact is missing result")
    if not message.HasField("decoder_config"):
        raise ValueError("temporal search artifact is missing decoder_config")
    return TemporalSearchArtifact(
        result=decode_temporal_search_result(
            message.result,
            corpus=corpus,
            expected_event_count=expected_event_count,
        ),
        video_scores=tuple(
            decode_video_scores(
                video,
                expected_event_count=expected_event_count,
                corpus=corpus,
            )
            for video in message.video_scores
        ),
        decoder_config=decode_decoder_config(message.decoder_config),
        scoring_revision=message.result.scoring_revision or None,
    )


def _require_byte_length(payload: bytes, expected: int, name: str) -> None:
    """Reject binary buffers whose declared shape cannot safely consume them."""

    if len(payload) != expected:
        raise ValueError(f"{name} does not match declared shape")


def _validate_path_cardinality(
    frame_ids: tuple[str, ...],
    frame_idxs: tuple[int, ...],
    timestamps_ms: tuple[int, ...],
) -> None:
    """Require one explicit canonical coordinate triple for every path event."""

    if not (len(frame_ids) == len(frame_idxs) == len(timestamps_ms)):
        raise ValueError("aligned path canonical field cardinality mismatch")


def _validate_video_identity(
    video_id: str,
    frame_ids: np.ndarray,
    frame_idxs: np.ndarray,
    timestamps_ms: np.ndarray,
    corpus: Corpus,
) -> None:
    """Validate every supplied coordinate rather than inferring metadata by position."""

    for position, raw_frame_id in enumerate(frame_ids):
        frame_id = str(raw_frame_id)
        frame = corpus.frame(frame_id)
        if frame.video_id != video_id:
            raise ValueError("frame has mixed canonical video identity")
        if frame.frame_id != frame_id:
            raise ValueError("frame_id conflicts with canonical data")
        if frame.frame_idx != int(frame_idxs[position]):
            raise ValueError("frame_idx conflicts with canonical data")
        if frame.timestamp_ms != int(timestamps_ms[position]):
            raise ValueError("timestamp conflicts with canonical data")


# Short aliases keep client and servicer call sites transport-oriented.
encode_result = encode_temporal_search_result
decode_result = decode_temporal_search_result
encode_artifact = encode_temporal_search_artifact
decode_artifact = decode_temporal_search_artifact
