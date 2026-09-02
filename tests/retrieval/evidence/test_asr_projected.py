"""Identity and timeline projection tests for segment-projected ASR Dense evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from hcmai.corpus.models import Frame
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
from hcmai.retrieval.retriever.segment.projector import SegmentFrameProjector


VISUAL_FRAME_IDS = np.asarray(["v1-f0", "v1-f1", "v2-f0", "v2-f1"])
VISUAL_VIDEO_IDS = np.asarray(["v1", "v1", "v2", "v2"])
VISUAL_FRAME_IDX = np.asarray([0, 1, 0, 1], dtype=np.int64)
VISUAL_TIMESTAMPS = np.asarray([0, 1_000, 0, 2_000], dtype=np.int64)
EXPECTED_POSITION = 2
QUERY_VECTORS = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
ALL_FRAME_POSITIONS = np.arange(len(VISUAL_FRAME_IDS), dtype=np.int64)
COLLISION_FRAME = 1
NO_ASR_FRAME = 0
EXPECTED_MAX_SCORE = 1.0


def _canonical_index() -> DenseIndex:
    """Build a tiny canonical visual index in the expected competition order."""

    mapping = pd.DataFrame(
        {
            "embedding_index": np.arange(len(VISUAL_FRAME_IDS), dtype=np.int64),
            "frame_id": VISUAL_FRAME_IDS,
            "video_id": VISUAL_VIDEO_IDS,
            "frame_idx": VISUAL_FRAME_IDX,
            "timestamp_ms": VISUAL_TIMESTAMPS,
        }
    )
    return DenseIndex.build(
        np.eye(len(VISUAL_FRAME_IDS), dtype=np.float32),
        mapping,
        dataset_version="test",
        model_name="test-model",
    )


def _segment_index() -> SegmentDenseIndex:
    """Build segments covering inside, midpoint-fallback, and unmapped cases."""

    mapping = pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": "s-inside",
                "video_id": "v1",
                "segment_index": 0,
                "start_ms": 900,
                "end_ms": 1_100,
            },
            {
                "embedding_index": 1,
                "segment_id": "s-gap",
                "video_id": "v2",
                "segment_index": 0,
                "start_ms": 800,
                "end_ms": 1_200,
            },
            {
                "embedding_index": 2,
                "segment_id": "s-far",
                "video_id": "v1",
                "segment_index": 1,
                "start_ms": 9_000,
                "end_ms": 10_000,
            },
        ]
    )
    return SegmentDenseIndex.build(
        np.eye(len(mapping), dtype=np.float32),
        mapping,
        dataset_version="test",
        model_name="test-model",
    )


def _collision_segment_index() -> SegmentDenseIndex:
    """Build normalized segments where two rows project to one frame."""

    mapping = pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": "s-collision-low",
                "video_id": "v1",
                "segment_index": 0,
                "start_ms": 900,
                "end_ms": 1_100,
            },
            {
                "embedding_index": 1,
                "segment_id": "s-collision-high",
                "video_id": "v1",
                "segment_index": 1,
                "start_ms": 950,
                "end_ms": 1_050,
            },
            {
                "embedding_index": 2,
                "segment_id": "s-covered",
                "video_id": "v2",
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 1,
            },
            {
                "embedding_index": 3,
                "segment_id": "s-unmapped",
                "video_id": "v1",
                "segment_index": 2,
                "start_ms": 9_000,
                "end_ms": 10_000,
            },
        ]
    )
    embeddings = np.asarray(
        [
            [0.5, np.sqrt(3.0) / 2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return SegmentDenseIndex.build(
        embeddings,
        mapping,
        dataset_version="test",
        model_name="test-model",
    )


def _unmapped_segment_index() -> SegmentDenseIndex:
    """Build valid ASR vectors whose video has no canonical frames."""

    mapping = pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": "s-unmapped-0",
                "video_id": "missing-video",
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 1,
            },
            {
                "embedding_index": 1,
                "segment_id": "s-unmapped-1",
                "video_id": "missing-video",
                "segment_index": 1,
                "start_ms": 1,
                "end_ms": 2,
            },
        ]
    )
    return SegmentDenseIndex.build(
        np.eye(2, 3, dtype=np.float32),
        mapping,
        dataset_version="test",
        model_name="test-model",
    )


def _projector(max_projection_gap_ms: int = 5_000) -> SegmentFrameProjector:
    """Build the real timeline projector from canonical runtime frames."""

    frames = [
        Frame("v1-f0", "v1", 0, 0, "/frames/v1-f0.jpg"),
        Frame("v1-f1", "v1", 1, 1_000, "/frames/v1-f1.jpg"),
        Frame("v2-f0", "v2", 0, 0, "/frames/v2-f0.jpg"),
        Frame("v2-f1", "v2", 1, 2_000, "/frames/v2-f1.jpg"),
    ]
    return SegmentFrameProjector(frames, max_projection_gap_ms=max_projection_gap_ms)


def make_projected_asr(max_projection_gap_ms: int = 5_000):
    """Construct the adapter under test with real segment/frame indexes."""

    from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex

    return SegmentProjectedASRIndex(
        segment_index=_segment_index(),
        canonical_index=_canonical_index(),
        projector=_projector(max_projection_gap_ms),
    )


def make_collision_projected_asr():
    """Construct a projected adapter with a deliberate frame collision."""

    from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex

    return SegmentProjectedASRIndex(
        segment_index=_collision_segment_index(),
        canonical_index=_canonical_index(),
        projector=_projector(max_projection_gap_ms=100),
    )


def make_unmapped_projected_asr():
    """Construct an adapter whose complete segment mapping has no matches."""

    from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex

    return SegmentProjectedASRIndex(
        segment_index=_unmapped_segment_index(),
        canonical_index=_canonical_index(),
        projector=_projector(),
    )


def test_projected_asr_mirrors_canonical_identity() -> None:
    """The adapter exposes canonical visual identity without rewriting it."""

    projected = make_projected_asr()

    np.testing.assert_array_equal(projected.frame_ids, VISUAL_FRAME_IDS)
    np.testing.assert_array_equal(projected.video_ids, VISUAL_VIDEO_IDS)
    np.testing.assert_array_equal(projected.frame_idx, VISUAL_FRAME_IDX)
    np.testing.assert_array_equal(projected.timestamps, VISUAL_TIMESTAMPS)
    assert projected.metadata.embedding_dim == 3


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [("video_id", "stale-video"), ("frame_idx", 99), ("timestamp_ms", 99_000)],
)
def test_projected_asr_rejects_stale_projector_identity(
    field: str,
    stale_value: object,
) -> None:
    """A known frame_id must carry the canonical projection identity tuple."""

    identity = {
        "frame_id": "v1-f1",
        "video_id": "v1",
        "frame_idx": 1,
        "timestamp_ms": 1_000,
    }
    identity[field] = stale_value
    projector = SimpleNamespace(
        project=lambda *args, **kwargs: SimpleNamespace(**identity)
    )

    with pytest.raises(ValueError, match="identity conflicts with canonical index"):
        from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex

        SegmentProjectedASRIndex(
            segment_index=_segment_index(),
            canonical_index=_canonical_index(),
            projector=projector,
        )


def test_segment_inside_interval_maps_to_existing_canonical_frame() -> None:
    """A frame inside the half-open segment interval is selected first."""

    projected = make_projected_asr()

    assert projected.segment_frame_positions[0] == 1


def test_segment_without_frame_inside_uses_nearest_midpoint_within_gap() -> None:
    """A segment gap falls back to the nearest midpoint frame within the limit."""

    projected = make_projected_asr(max_projection_gap_ms=5_000)

    assert projected.segment_frame_positions[1] == EXPECTED_POSITION


def test_segment_outside_projection_gap_is_unmapped() -> None:
    """A segment beyond the configured projection gap remains unmapped."""

    projected = make_projected_asr(max_projection_gap_ms=100)

    assert projected.segment_frame_positions[2] == -1


def test_score_subset_scores_all_segments_then_scatter_maxes_collisions() -> None:
    """All segment vectors contribute, with maximum score winning collisions."""

    projected = make_collision_projected_asr()

    scores = projected.score_subset(
        QUERY_VECTORS,
        ALL_FRAME_POSITIONS,
        chunk_size=2,
    )

    assert scores.shape == (len(QUERY_VECTORS), len(ALL_FRAME_POSITIONS))
    assert scores[0, COLLISION_FRAME] == pytest.approx(EXPECTED_MAX_SCORE)


def test_uncovered_frames_receive_event_floor() -> None:
    """Canonical frames without projected ASR receive each event's floor."""

    projected = make_collision_projected_asr()

    scores = projected.score_subset(QUERY_VECTORS, ALL_FRAME_POSITIONS)

    assert scores[0, NO_ASR_FRAME] == pytest.approx(scores[0].min())


def test_no_valid_projected_segments_returns_constant_zero_row() -> None:
    """No valid segment projection produces a zero row for each event."""

    projected_without_valid_segments = make_unmapped_projected_asr()

    scores = projected_without_valid_segments.score_subset(
        QUERY_VECTORS,
        ALL_FRAME_POSITIONS,
    )

    np.testing.assert_array_equal(scores, np.zeros_like(scores))


def test_score_subset_honors_requested_canonical_positions() -> None:
    """Only requested canonical positions are returned in requested order."""

    projected = make_collision_projected_asr()
    subset = np.array([3, 1], dtype=np.int64)

    scores = projected.score_subset(QUERY_VECTORS, subset)

    assert scores.shape == (len(QUERY_VECTORS), 2)
    np.testing.assert_allclose(
        scores,
        np.asarray(
            [
                [0.0, 1.0],
                [np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0],
            ],
            dtype=np.float32,
        ),
    )


def test_score_subset_does_not_call_segment_top_k_search(monkeypatch) -> None:
    """Full segment scoring reads vectors directly instead of searching top-k."""

    projected = make_collision_projected_asr()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("segment top-k search must not be called")

    monkeypatch.setattr(projected.segment_index, "search", fail_if_called)

    projected.score_subset(QUERY_VECTORS, ALL_FRAME_POSITIONS)


def test_score_subset_accepts_one_dimensional_query() -> None:
    """A single query vector is reshaped to one event row."""

    projected = make_collision_projected_asr()

    scores = projected.score_subset(QUERY_VECTORS[0], ALL_FRAME_POSITIONS)

    assert scores.shape == (1, len(ALL_FRAME_POSITIONS))


@pytest.mark.parametrize(
    "query_vectors",
    [
        np.ones(2, dtype=np.float32),
        np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[np.inf, 0.0, 0.0]], dtype=np.float32),
    ],
)
def test_score_subset_validates_query_vectors(query_vectors) -> None:
    """Wrong dimensions and non-finite query values are rejected."""

    projected = make_collision_projected_asr()

    with pytest.raises(ValueError):
        projected.score_subset(query_vectors, ALL_FRAME_POSITIONS)


@pytest.mark.parametrize(
    "positions",
    [
        np.asarray([1.0], dtype=np.float32),
        np.asarray([-1], dtype=np.int64),
        np.asarray([len(VISUAL_FRAME_IDS)], dtype=np.int64),
        np.asarray([[1]], dtype=np.int64),
    ],
)
def test_score_subset_validates_positions(positions) -> None:
    """Requested positions must be one-dimensional integer in-bounds values."""

    projected = make_collision_projected_asr()

    with pytest.raises(ValueError):
        projected.score_subset(QUERY_VECTORS, positions)


def test_evidence_package_star_import_binds_all_public_symbols() -> None:
    """Every name advertised by the evidence package is publicly importable."""

    namespace: dict[str, object] = {}
    exec("from hcmai.retrieval.evidence import *", namespace)

    import hcmai.retrieval.evidence as evidence

    assert set(evidence.__all__).issubset(namespace)
