"""Tests for full-corpus multimodal Dense temporal scoring."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hcmai.corpus.models import Frame
from hcmai.common.config import DenseTemporalWeights
from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex
from hcmai.retrieval.evidence.dense import DenseTemporalScorer
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
from hcmai.retrieval.retriever.segment.projector import SegmentFrameProjector


class FakeEncoder:
    """Record batch calls and return deterministic query vectors."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[list[str]] = []

    def encode_text(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return np.asarray([[self.value, 1.0] for _ in texts], dtype=np.float32)


class FakeIndex:
    """Expose canonical identity and exact subset scoring."""

    def __init__(self, scale: float = 1.0) -> None:
        self.frame_ids = np.asarray(["f1", "f2", "f3"])
        self.video_ids = np.asarray(["v1", "v1", "v2"])
        self.frame_idx = np.asarray([1, 2, 3], dtype=np.int64)
        self.timestamps = np.asarray([100, 200, 300], dtype=np.int64)
        self.scale = scale
        self.calls = 0
        self.metadata = SimpleNamespace(embedding_dim=2)

    def score_subset(
        self, query_vectors: np.ndarray, positions: np.ndarray, chunk_size: int
    ) -> np.ndarray:
        self.calls += 1
        base = np.asarray([[1.0, 2.0, 3.0]] * len(query_vectors), dtype=np.float32)
        return base * self.scale


FRAME_IDS = np.asarray(["v1-f0", "v1-f1", "v2-f0", "v2-f1"])
VIDEO_IDS = np.asarray(["v1", "v1", "v2", "v2"])
FRAME_IDX = np.asarray([0, 1, 0, 1], dtype=np.int64)
TIMESTAMPS = np.asarray([0, 1_000, 0, 2_000], dtype=np.int64)
FRAME_COUNT = len(FRAME_IDS)


class CountingEncoder:
    """Return one deterministic vector batch while recording each call."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)
        self.calls: list[list[str]] = []

    def encode_text(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.repeat(self.vector[None, :], len(texts), axis=0)


class RecordingFrameIndex:
    """Provide frame-native identity and record query vectors passed to it."""

    def __init__(
        self,
        identity_source: object,
        embedding_dim: int,
        scale: float,
    ) -> None:
        self.frame_ids = np.asarray(identity_source.frame_ids)
        self.video_ids = np.asarray(identity_source.video_ids)
        self.frame_idx = np.asarray(identity_source.frame_idx, dtype=np.int64)
        self.timestamps = np.asarray(identity_source.timestamps, dtype=np.int64)
        self.metadata = SimpleNamespace(embedding_dim=embedding_dim)
        self.scale = scale
        self.query_batches: list[np.ndarray] = []

    def score_subset(
        self, query_vectors: np.ndarray, positions: np.ndarray, chunk_size: int
    ) -> np.ndarray:
        self.query_batches.append(query_vectors)
        base = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        return np.repeat(base * self.scale, len(query_vectors), axis=0)[:, positions]


def _tiny_projected_asr() -> SegmentProjectedASRIndex:
    """Build a real projected ASR index with two segments and four frames."""

    canonical_mapping = pd.DataFrame(
        {
            "embedding_index": np.arange(FRAME_COUNT, dtype=np.int64),
            "frame_id": FRAME_IDS,
            "video_id": VIDEO_IDS,
            "frame_idx": FRAME_IDX,
            "timestamp_ms": TIMESTAMPS,
        }
    )
    canonical_index = DenseIndex.build(
        np.eye(FRAME_COUNT, dtype=np.float32),
        canonical_mapping,
        dataset_version="test",
        model_name="test-visual",
    )
    segment_mapping = pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": "s-v1",
                "video_id": "v1",
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 500,
            },
            {
                "embedding_index": 1,
                "segment_id": "s-v2",
                "video_id": "v2",
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 500,
            },
        ]
    )
    segment_index = SegmentDenseIndex.build(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        segment_mapping,
        dataset_version="test",
        model_name="test-text",
    )
    frames = [
        Frame("v1-f0", "v1", 0, 0, "/frames/v1-f0.jpg"),
        Frame("v1-f1", "v1", 1, 1_000, "/frames/v1-f1.jpg"),
        Frame("v2-f0", "v2", 0, 0, "/frames/v2-f0.jpg"),
        Frame("v2-f1", "v2", 1, 2_000, "/frames/v2-f1.jpg"),
    ]
    return SegmentProjectedASRIndex(
        segment_index=segment_index,
        canonical_index=canonical_index,
        projector=SegmentFrameProjector(frames),
    )


def test_dense_reuses_one_visual_and_one_text_encoding_batch() -> None:
    """Reuse BGE vectors for Context and frame-ASR scoring."""

    visual_encoder = FakeEncoder(1.0)
    text_encoder = FakeEncoder(2.0)
    indexes = [FakeIndex(1.0), FakeIndex(2.0), FakeIndex(3.0)]
    scorer = DenseTemporalScorer(
        visual_index=indexes[0],
        context_index=indexes[1],
        asr_index=indexes[2],
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        weights=DenseTemporalWeights(),
        chunk_size=2,
    )

    scores = scorer.score_events(("one", "two"))

    assert scores.shape == (2, 3)
    assert visual_encoder.calls == [["one", "two"]]
    assert text_encoder.calls == [["one", "two"]]
    assert [index.calls for index in indexes] == [1, 1, 1]
    np.testing.assert_allclose(scores[0], [0.0, 0.5, 1.0])


def test_dense_rejects_identity_mismatch() -> None:
    """Require all three indexes to use identical canonical frame order."""

    visual = FakeIndex()
    context = FakeIndex()
    asr = FakeIndex()
    asr.frame_idx[1] = 99

    with pytest.raises(ValueError, match="identity"):
        DenseTemporalScorer(
            visual_index=visual,
            context_index=context,
            asr_index=asr,
            visual_encoder=FakeEncoder(1.0),
            text_encoder=FakeEncoder(1.0),
            weights=DenseTemporalWeights(),
        )


def test_dense_accepts_real_projected_asr_with_one_shared_bge_batch(
    monkeypatch,
) -> None:
    """Fuse frame-native indexes with a real segment-to-frame ASR adapter."""

    projected_asr = _tiny_projected_asr()
    visual_index = RecordingFrameIndex(projected_asr, embedding_dim=2, scale=1.0)
    context_index = RecordingFrameIndex(projected_asr, embedding_dim=3, scale=2.0)
    visual_encoder = CountingEncoder([1.0, 0.0])
    text_encoder = CountingEncoder([1.0, 0.0, 0.0])
    asr_query_batches: list[np.ndarray] = []
    score_projected_asr = projected_asr.score_subset

    def record_projected_asr_batch(
        query_vectors: np.ndarray,
        positions: np.ndarray,
        chunk_size: int = 65_536,
    ) -> np.ndarray:
        asr_query_batches.append(query_vectors)
        return score_projected_asr(query_vectors, positions, chunk_size)

    monkeypatch.setattr(projected_asr, "score_subset", record_projected_asr_batch)
    scorer = DenseTemporalScorer(
        visual_index=visual_index,
        context_index=context_index,
        asr_index=projected_asr,
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        weights=DenseTemporalWeights(),
    )

    scores = scorer.score_events(["event one", "event two"])

    assert scores.shape == (2, FRAME_COUNT)
    assert visual_encoder.calls == [["event one", "event two"]]
    assert text_encoder.calls == [["event one", "event two"]]
    assert len(context_index.query_batches) == 1
    assert len(asr_query_batches) == 1
    assert context_index.query_batches[0] is asr_query_batches[0]
    np.testing.assert_array_equal(
        context_index.query_batches[0],
        np.repeat(text_encoder.vector[None, :], 2, axis=0),
    )


def test_dense_rejects_projected_asr_embedding_dimension_mismatch() -> None:
    """Reject Context and projected ASR indexes with different BGE dimensions."""

    projected_asr = _tiny_projected_asr()
    visual_index = RecordingFrameIndex(projected_asr, embedding_dim=2, scale=1.0)
    context_index = RecordingFrameIndex(projected_asr, embedding_dim=2, scale=2.0)

    with pytest.raises(
        ValueError,
        match="Context and ASR Dense index dimensions differ",
    ):
        DenseTemporalScorer(
            visual_index=visual_index,
            context_index=context_index,
            asr_index=projected_asr,
            visual_encoder=CountingEncoder([1.0, 0.0]),
            text_encoder=CountingEncoder([1.0, 0.0, 0.0]),
            weights=DenseTemporalWeights(),
        )
