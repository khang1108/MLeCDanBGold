"""Deterministic test fakes for temporal evidence index and encoder contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class FakeIndex:
    scores: np.ndarray
    embedding_dim: int = 2
    coverage_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        frame_count = self.scores.shape[1]
        self.frame_ids = np.asarray([f"f{i}" for i in range(frame_count)])
        self.video_ids = np.asarray(["v1"] * frame_count)
        self.frame_idx = np.arange(frame_count, dtype=np.int64)
        self.timestamps = np.arange(frame_count, dtype=np.int64) * 1000
        self.mapping = np.arange(frame_count)
        self.metadata = SimpleNamespace(embedding_dim=self.embedding_dim)

    def score_subset(
        self,
        query_vectors: np.ndarray,
        positions: np.ndarray,
        chunk_size: int = 65_536,
    ) -> np.ndarray:
        del query_vectors, chunk_size
        return np.asarray(self.scores[:, positions], dtype=np.float32)

    def video_positions(self, video_id: str) -> np.ndarray:
        return np.flatnonzero(self.video_ids == video_id)


class FakeEncoder:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.calls: list[tuple[str, ...]] = []

    def encode_text(self, events: Sequence[str]) -> np.ndarray:
        self.calls.append(tuple(events))
        return self.vectors[: len(events)]


class FakeSegmentIndex:
    def __init__(
        self,
        mapping: list[dict[str, Any]] | None = None,
        vectors: np.ndarray | None = None,
        embedding_dim: int = 2,
    ) -> None:
        if mapping is None:
            mapping = [{"video_id": "v1", "start_ms": 0, "end_ms": 1000}]
        self.mapping = pd.DataFrame(mapping)
        if vectors is None:
            vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.metadata = SimpleNamespace(embedding_dim=embedding_dim)


class FakeProjection:
    def __init__(
        self,
        video_id: str = "v1",
        frame_id: str = "f0",
        frame_idx: int = 0,
        timestamp_ms: int = 0,
    ) -> None:
        self.video_id = video_id
        self.frame_id = frame_id
        self.frame_idx = frame_idx
        self.timestamp_ms = timestamp_ms


class FakeProjector:
    def __init__(
        self,
        mapping: dict[tuple[str, int, int], FakeProjection | None] | None = None,
    ) -> None:
        self._mapping = mapping or {}

    def project(self, video_id: str, *, start_ms: int, end_ms: int) -> FakeProjection | None:
        key = (video_id, start_ms, end_ms)
        if key in self._mapping:
            return self._mapping[key]
        return FakeProjection(video_id=video_id, frame_id="f0", frame_idx=0, timestamp_ms=start_ms)


class FakeSourceRetriever:
    def __init__(self, source: Any, index: Any, encoder: Any) -> None:
        self.source = source
        self.index = index
        self.encoder = encoder


class FakeRetrievalService:
    def __init__(self, retrievers: Sequence[Any]) -> None:
        self.retrievers = {retriever.source: retriever for retriever in retrievers}

    def source_retriever(self, source: Any) -> Any:
        return self.retrievers.get(source)
