"""Remote GPU adapters that preserve local preprocessing semantics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from hcmai.common.schemas import BoundaryScoreResponse, EmbeddingResponse
from hcmai.data.preprocessing.video import FrameMeta


class PreprocessingClient(Protocol):
    """Định nghĩa interface cho kết nối tới các dịch vụ tiền xử lý (Preprocessing) remote."""
    def boundary_scores(
        self,
        frames: np.ndarray,
        *,
        request_id: str,
        source: str,
    ) -> BoundaryScoreResponse: ...

    def embed_images(
        self,
        images: Sequence[Image.Image],
        *,
        source: str = "visual",
        item_ids: list[str] | None = None,
    ) -> EmbeddingResponse: ...


def _request_id(prefix: str, value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    identity = (
        str(contiguous.dtype).encode(),
        repr(contiguous.shape).encode(),
        contiguous.tobytes(),
    )
    digest = hashlib.sha256(b"\0".join(identity)).hexdigest()
    return f"{prefix}-{digest}"


class RemoteTransNetDetector:
    """Gửi các frame video tới remote worker để chạy mô hình TransNetV2 (Shot Boundary Detection).
    Nhận về danh sách điểm số cắt cảnh ứng với từng frame.
    """

    def __init__(
        self,
        client: PreprocessingClient,
        *,
        model_name: str,
        revision: str | None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.revision = revision

    def score(self, _path: Path, frames: np.ndarray) -> np.ndarray:
        response = self.client.boundary_scores(
            frames,
            request_id=_request_id("shot", frames),
            source="shot",
        )
        _validate_model(response.model, response.revision, self.model_name, self.revision)
        values = np.asarray(response.scores, dtype=np.float32)
        if values.shape != (len(frames),) or not np.all(np.isfinite(values)):
            raise ValueError("remote TransNet returned invalid scores")
        return values


class RemoteDinoEncoder:
    """Gửi danh sách hình ảnh qua mạng tới remote worker để chạy mô hình DINO.
    Trả về bộ đặc trưng (embedding vectors) để sử dụng cho tác vụ KIS/TRAKE.
    """

    def __init__(
        self,
        client: PreprocessingClient,
        *,
        model_name: str,
        revision: str | None,
        dtype: str = "float32",
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.revision = revision
        self.dtype = dtype
        self.embedding_dim = 0

    def encode(self, images: list[Any]) -> np.ndarray:
        identifiers = [str(index) for index in range(len(images))]
        response = self.client.embed_images(
            images, source="dino", item_ids=identifiers
        )
        _validate_model(response.model, response.revision, self.model_name, self.revision)
        if response.item_ids != identifiers or not response.normalized:
            raise ValueError("remote DINO metadata mismatch")
        vectors = np.asarray(response.embeddings, dtype=self.dtype)
        if vectors.shape != (len(images), response.dimension):
            raise ValueError("remote DINO shape mismatch")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("remote DINO contains non-finite vectors")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
            raise ValueError("remote DINO vectors are not L2-normalized")
        self.embedding_dim = response.dimension
        return vectors


class RemoteEfficientGEBDDetector:
    """Giữ nguyên logic lấy mẫu (sampling) và gom nhóm sliding window tại máy local,
    nhưng gọi remote worker (Kaggle) để chấm điểm (scoring) EfficientGEBD cho từng window.
    Giúp offload phần tính toán mạng neural nặng nề nhưng vẫn đảm bảo đúng thuật toán GEBD.
    """

    def __init__(
        self,
        client: PreprocessingClient,
        *,
        model_name: str,
        revision: str | None,
        sample_fps: float = 10.0,
        resolution: int = 224,
        sequence_length: int = 100,
        overlap: int = 20,
    ) -> None:
        if sample_fps <= 0 or resolution <= 0 or sequence_length <= 0:
            raise ValueError("GEBD sampling settings must be positive")
        if overlap < 0 or overlap >= sequence_length:
            raise ValueError("GEBD overlap must be within the sequence length")
        self.client = client
        self.model_name = model_name
        self.revision = revision
        self.sample_fps = sample_fps
        self.resolution = resolution
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.start()

    def start(self) -> None:
        self.positions: list[int] = []
        self.totals: list[float] = []
        self.counts: list[int] = []
        self.window: list[tuple[int, np.ndarray]] = []
        self.pending = 0
        self.next_ms = 0.0

    def update(self, frame: FrameMeta, source: Any) -> None:
        if frame.timestamp_ms < self.next_ms:
            return
        image = source.to_image().convert("RGB").resize(
            (self.resolution, self.resolution)
        )
        tensor = np.asarray(image, dtype=np.uint8)
        image.close()
        self.positions.append(frame.decode_index)
        self.totals.append(0.0)
        self.counts.append(0)
        self.window.append((len(self.positions) - 1, tensor))
        self.pending += 1
        if len(self.window) == self.sequence_length:
            self._add_scores(self.window)
            self.window = self.window[self.sequence_length - self.overlap :]
            self.pending = 0
        interval = 1_000 / self.sample_fps
        while self.next_ms <= frame.timestamp_ms:
            self.next_ms += interval

    def scores(self, frame_count: int) -> np.ndarray:
        if not self.positions:
            return np.zeros(frame_count, dtype=np.float32)
        if self.window and (self.pending or not any(self.counts)):
            self._add_scores(self.window)
        sampled = np.asarray(self.totals) / np.maximum(self.counts, 1)
        return np.interp(
            np.arange(frame_count), self.positions, sampled
        ).astype(np.float32)

    def _add_scores(self, window: list[tuple[int, np.ndarray]]) -> None:
        frames = np.stack([tensor for _, tensor in window])
        response = self.client.boundary_scores(
            frames,
            request_id=_request_id("event", frames),
            source="event",
        )
        _validate_model(response.model, response.revision, self.model_name, self.revision)
        scores = np.asarray(response.scores, dtype=np.float32)
        if scores.shape != (len(window),) or not np.all(np.isfinite(scores)):
            raise ValueError("remote GEBD returned invalid scores")
        for (index, _), score in zip(window, scores):
            self.totals[index] += float(score)
            self.counts[index] += 1


def _validate_model(
    actual_name: str,
    actual_revision: str | None,
    expected_name: str,
    expected_revision: str | None,
) -> None:
    if actual_name != expected_name:
        raise ValueError("remote preprocessing checkpoint mismatch")
    if expected_revision is not None and actual_revision != expected_revision:
        raise ValueError("remote preprocessing revision mismatch")
