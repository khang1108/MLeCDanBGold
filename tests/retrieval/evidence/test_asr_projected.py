"""Identity and timeline projection tests for segment-projected ASR Dense evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcmai.corpus.models import Frame
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
from hcmai.retrieval.retriever.segment.projector import SegmentFrameProjector


VISUAL_FRAME_IDS = np.asarray(["v1-f0", "v1-f1", "v2-f0", "v2-f1"])
VISUAL_VIDEO_IDS = np.asarray(["v1", "v1", "v2", "v2"])
VISUAL_FRAME_IDX = np.asarray([0, 1, 0, 1], dtype=np.int64)
VISUAL_TIMESTAMPS = np.asarray([0, 1_000, 0, 2_000], dtype=np.int64)
EXPECTED_POSITION = 2


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


def test_projected_asr_mirrors_canonical_identity() -> None:
    """The adapter exposes canonical visual identity without rewriting it."""

    projected = make_projected_asr()

    np.testing.assert_array_equal(projected.frame_ids, VISUAL_FRAME_IDS)
    np.testing.assert_array_equal(projected.video_ids, VISUAL_VIDEO_IDS)
    np.testing.assert_array_equal(projected.frame_idx, VISUAL_FRAME_IDX)
    np.testing.assert_array_equal(projected.timestamps, VISUAL_TIMESTAMPS)
    assert projected.metadata.embedding_dim == 3


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


def test_evidence_package_star_import_binds_all_public_symbols() -> None:
    """Every name advertised by the evidence package is publicly importable."""

    namespace: dict[str, object] = {}
    exec("from hcmai.retrieval.evidence import *", namespace)

    import hcmai.retrieval.evidence as evidence

    assert set(evidence.__all__).issubset(namespace)
