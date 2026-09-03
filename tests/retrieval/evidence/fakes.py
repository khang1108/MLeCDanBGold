"""Deterministic test fakes for temporal evidence index and encoder contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np


@dataclass
class FakeIndex:
    scores: np.ndarray
    embedding_dim: int = 2

    def __post_init__(self) -> None:
        frame_count = self.scores.shape[1]
        self.frame_ids = np.asarray([f"f{i}" for i in range(frame_count)])
        self.video_ids = np.asarray(["v1"] * frame_count)
        self.frame_idx = np.arange(frame_count, dtype=np.int64)
        self.timestamps = np.arange(frame_count, dtype=np.int64) * 1000
        self.metadata = SimpleNamespace(embedding_dim=self.embedding_dim)

    def score_subset(
        self,
        query_vectors: np.ndarray,
        positions: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray:
        del query_vectors, chunk_size
        return np.asarray(self.scores[:, positions], dtype=np.float32)

    def video_positions(self, video_id: str) -> np.ndarray:
        return np.flatnonzero(self.video_ids == video_id)


class FakeEncoder:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.calls: list[tuple[str, ...]] = []

    def encode_text(self, events: list[str]) -> np.ndarray:
        self.calls.append(tuple(events))
        return self.vectors[: len(events)]
