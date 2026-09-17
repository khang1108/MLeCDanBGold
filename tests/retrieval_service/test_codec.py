"""Regression tests for strict retrieval protobuf and NumPy codecs."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from hcmai.corpus.models import Frame
from hcmai.kis.models import KISImageRef
from hcmai.orchestration.workflows.temporal_search import (
    DecoderConfigSnapshot,
    TemporalSearchArtifact,
    TemporalSearchResult,
)
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.retrieval_service.codec import (
    VIDEO_SCORES_ENCODING_VERSION,
    decode_aligned_path,
    decode_plan,
    decode_temporal_search_artifact,
    decode_video_scores,
    encode_aligned_path,
    encode_plan,
    encode_temporal_search_artifact,
    encode_video_scores,
)
from hcmai.retrieval_service.proto import retrieval_pb2
from hcmai.temporal.dp import AlignedPath


@pytest.fixture
def corpus() -> Mock:
    """Provide canonical frames with identities unrelated to array positions."""

    frames = {
        "kf_alpha": Frame("kf_alpha", "L01_V001", 17, 680, "/tmp/alpha.jpg"),
        "kf_omega": Frame("kf_omega", "L01_V001", 931, 37_240, "/tmp/omega.jpg"),
    }
    value = Mock()
    value.frame.side_effect = frames.__getitem__
    return value


def _video_scores() -> VideoEventScores:
    """Build a small score matrix whose metadata cannot be position-derived."""

    return VideoEventScores(
        video_id="L01_V001",
        frame_ids=np.array(["kf_alpha", "kf_omega"]),
        frame_idx=np.array([17, 931], dtype=np.int64),
        timestamps_ms=np.array([680, 37_240], dtype=np.int64),
        scores=np.array([[0.25, 0.75], [0.5, 0.125]], dtype=np.float32),
    )


def _plan() -> KISRetrievalPlan:
    """Build a two-event plan with independently optional text views."""

    image = KISImageRef(asset_id="sha256-image", content_type="image/png")
    return KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "canonical one", "dense one", None, (image,)),
            KISRetrievalEvent("E2", "canonical two", "dense two", "bm25 two"),
        )
    )


def _path() -> AlignedPath:
    """Build a path retaining the same non-positional frame identities."""

    return AlignedPath(
        video_id="L01_V001",
        score=1.25,
        frame_ids=("kf_alpha", "kf_omega"),
        frame_idxs=(17, 931),
        timestamps_ms=(680, 37_240),
    )


def test_REQ_009_video_score_round_trip_preserves_canonical_identity(corpus) -> None:
    """Binary score buffers retain all canonical coordinates exactly."""

    encoded = encode_video_scores(_video_scores())
    decoded = decode_video_scores(encoded, expected_event_count=2, corpus=corpus)

    assert decoded.video_id == "L01_V001"
    assert decoded.frame_ids.tolist() == ["kf_alpha", "kf_omega"]
    assert decoded.frame_idx.tolist() == [17, 931]
    assert decoded.timestamps_ms.tolist() == [680, 37_240]
    assert decoded.scores.dtype == np.float32
    assert decoded.scores.shape == (2, 2)
    assert not decoded.frame_ids.flags.writeable
    assert not decoded.frame_idx.flags.writeable
    assert not decoded.timestamps_ms.flags.writeable
    assert not decoded.scores.flags.writeable


def test_REQ_010_rejects_truncated_score_bytes(corpus) -> None:
    """Truncated float buffers are protocol failures rather than reshaped data."""

    encoded = encode_video_scores(_video_scores())
    encoded.scores_f32_le = encoded.scores_f32_le[:-1]

    with pytest.raises(ValueError, match="score byte length"):
        decode_video_scores(encoded, expected_event_count=2, corpus=corpus)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda message: setattr(message, "encoding_version", 99), "encoding version"),
        (lambda message: setattr(message, "event_count", 1), "event count"),
        (lambda message: setattr(message, "frame_count", 3), "frame count"),
        (lambda message: setattr(message, "frame_idxs_i64_le", b"bad"), "frame_idx byte length"),
        (lambda message: setattr(message, "timestamps_ms_i64_le", b"bad"), "timestamp byte length"),
    ],
)
def test_REQ_010_rejects_invalid_binary_score_metadata(corpus, mutate, expected) -> None:
    """Version, dimensions, and packed metadata must agree before decoding."""

    encoded = encode_video_scores(_video_scores())
    mutate(encoded)

    with pytest.raises(ValueError, match=expected):
        decode_video_scores(encoded, expected_event_count=2, corpus=corpus)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("video_id", "other_video", "mixed canonical video identity"),
        ("frame_idxs_i64_le", np.array([18, 931], dtype="<i8").tobytes(), "frame_idx"),
        ("timestamps_ms_i64_le", np.array([681, 37_240], dtype="<i8").tobytes(), "timestamp"),
    ],
)
def test_REQ_009_rejects_score_identity_mismatch(corpus, field, value, expected) -> None:
    """Wire metadata cannot rewrite canonical Corpus identity."""

    encoded = encode_video_scores(_video_scores())
    setattr(encoded, field, value)

    with pytest.raises(ValueError, match=expected):
        decode_video_scores(encoded, expected_event_count=2, corpus=corpus)


def test_REQ_005_plan_round_trip_retains_optional_text_and_image_assets() -> None:
    """Optional proto strings and image asset IDs survive the transport boundary."""

    encoded = encode_plan(_plan(), use_dense=True, use_bm25=False, top_k=7)
    decoded = decode_plan(encoded)

    assert encoded.events[0].HasField("bm25_text") is False
    assert decoded.event_ids == ("E1", "E2")
    assert decoded.events[0].canonical_text == "canonical one"
    assert decoded.events[0].dense_text == "dense one"
    assert decoded.events[0].bm25_text is None
    assert decoded.events[0].image_refs[0].asset_id == "sha256-image"
    assert decoded.events[1].canonical_text == "canonical two"
    assert decoded.events[1].dense_text == "dense two"
    assert decoded.events[1].bm25_text == "bm25 two"
    assert encoded.use_dense is True
    assert encoded.use_bm25 is False
    assert encoded.top_k == 7


@pytest.mark.parametrize(
    "event_ids",
    [("E2", "E1"), ("E1", "E3")],
)
def test_REQ_005_rejects_plan_event_ids_outside_exact_e_order(event_ids) -> None:
    """Transport plans must retain the server-owned E1..En event order."""

    message = retrieval_pb2.SearchPlanRequest(
        events=[
            retrieval_pb2.RetrievalEvent(event_id=event_id, canonical_text="text")
            for event_id in event_ids
        ],
        use_dense=True,
        top_k=1,
    )

    with pytest.raises(ValueError, match="event IDs must be sequential"):
        decode_plan(message)


def test_REQ_009_path_round_trip_validates_cardinality_and_canonical_identity(corpus) -> None:
    """Aligned paths preserve all explicit coordinates and reject row mismatch."""

    encoded = encode_aligned_path(_path())
    assert decode_aligned_path(encoded, corpus=corpus) == _path()

    encoded.frame_idxs.pop()
    with pytest.raises(ValueError, match="cardinality"):
        decode_aligned_path(encoded, corpus=corpus)


def test_REQ_006_artifact_round_trip_retains_remote_scoring_revision(corpus) -> None:
    """Remote artifacts retain their generation while local defaults stay optional."""

    artifact = TemporalSearchArtifact(
        result=TemporalSearchResult(paths=(_path(),), retrieval_ms=12.5, alignment_ms=3.0),
        video_scores=(_video_scores(),),
        decoder_config=DecoderConfigSnapshot(0.1, 1.5, 0.2, 250),
        scoring_revision="retrieval-gen-4",
    )

    decoded = decode_temporal_search_artifact(
        encode_temporal_search_artifact(artifact), expected_event_count=2, corpus=corpus
    )

    assert decoded.scoring_revision == "retrieval-gen-4"
    assert decoded.result == artifact.result
    assert decoded.decoder_config == artifact.decoder_config
    assert decoded.video_scores[0].frame_ids.tolist() == ["kf_alpha", "kf_omega"]
