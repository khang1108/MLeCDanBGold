"""Focused tests for direct SigLIP2 image-query retrieval."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from hcmai.corpus import Frame
from hcmai.orchestration.image_search import (
    ImageQueryTooLargeError,
    ImageSearchService,
    InvalidImageQueryError,
)
from hcmai.retrieval.models import RetrievalCandidate, RetrievalResult, RetrievalSource


class FakeEncoder:
    """Return one normalized visual vector and retain decoded-image facts."""

    config = SimpleNamespace(model_name="google/siglip2-base-patch16-224")
    embedding_dim = 2

    def __init__(self) -> None:
        self.images: list[Image.Image] = []

    def encode_images(self, images: list[Image.Image], stats=None) -> np.ndarray:
        """Record the RGB input and return its deterministic query vector."""

        del stats
        self.images.extend(images)
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class FakeVisualRetriever:
    """Record the shared vector batch and return one canonical candidate."""

    def __init__(self) -> None:
        self.batch = None
        self.top_k = None

    def search_vectors(self, batch, top_k: int) -> list[RetrievalResult]:
        """Return a visual candidate without performing FAISS work."""

        self.batch = batch
        self.top_k = top_k
        return [
            RetrievalResult(
                candidates=[
                    RetrievalCandidate(
                        frame_id="f-7",
                        source_scores={RetrievalSource.VISUAL: 0.91},
                        source_ranks={RetrievalSource.VISUAL: 1},
                    )
                ]
            )
        ]


class FakeRetrieval:
    """Expose only the configured visual retriever lookup."""

    def __init__(self, visual: FakeVisualRetriever) -> None:
        self.visual = visual

    def source_retriever(self, source: RetrievalSource):
        """Return the visual retriever for its exact evidence source."""

        return self.visual if source is RetrievalSource.VISUAL else None


class FakeCorpus:
    """Resolve canonical identity and representative evidence for one frame."""

    @staticmethod
    def frame(frame_id: str) -> Frame:
        """Return the organizer-owned coordinates for the retrieved frame."""

        assert frame_id == "f-7"
        return Frame(
            frame_id="f-7",
            video_id="V01",
            frame_idx=700,
            timestamp_ms=7_000,
            image_path="f-7.jpg",
        )

    @staticmethod
    def title(video_id: str) -> str:
        """Return test title metadata."""

        assert video_id == "V01"
        return "Example"

    @staticmethod
    def caption(frame_id: str) -> str:
        """Return test caption metadata."""

        assert frame_id == "f-7"
        return "A red image"

    @staticmethod
    def ocr(frame_id: str) -> None:
        """Represent absent OCR without turning it into negative evidence."""

        assert frame_id == "f-7"
        return None

    @staticmethod
    def objects(frame_id: str) -> tuple[str, ...]:
        """Return retained object multiplicity."""

        assert frame_id == "f-7"
        return ("person", "person")

    @staticmethod
    def transcript(video_id: str, start_ms: int, end_ms: int) -> None:
        """Represent absent point-contained transcript evidence."""

        assert (video_id, start_ms, end_ms) == ("V01", 7_000, 7_001)
        return None


def _image_bytes(*, size: tuple[int, int] = (2, 2)) -> bytes:
    """Create a small in-memory JPEG fixture."""

    output = io.BytesIO()
    Image.new("RGB", size, "red").save(output, "JPEG")
    return output.getvalue()


def _service(*, max_upload_bytes: int = 1024, max_pixels: int = 100):
    """Build the service and expose its fakes for assertions."""

    visual = FakeVisualRetriever()
    encoder = FakeEncoder()
    service = ImageSearchService(
        FakeCorpus(),
        FakeRetrieval(visual),
        encoder,
        max_upload_bytes=max_upload_bytes,
        max_pixels=max_pixels,
    )
    return service, visual, encoder


def test_image_search_encodes_with_siglip_and_reuses_visual_search() -> None:
    """Return canonical results from the existing visual vector boundary."""

    service, visual, encoder = _service()

    response = service.search(
        _image_bytes(),
        content_type="image/jpeg",
        top_k=5,
    )

    assert visual.top_k == 5
    np.testing.assert_allclose(visual.batch.vectors, [[1.0, 0.0]])
    assert visual.batch.model_name == "google/siglip2-base-patch16-224"
    assert encoder.images[0].mode == "RGB"
    result = response.results[0]
    assert (result.frame_id, result.video_id) == ("f-7", "V01")
    assert (result.frame_idx, result.timestamp_ms) == (700, 7_000)
    assert result.frame_ids == ["f-7"]
    assert result.timestamps_ms == [7_000]
    assert result.score == pytest.approx(0.91)
    assert result.metadata.objects == ["person", "person"]


def test_image_search_rejects_invalid_and_oversized_payloads() -> None:
    """Bound upload bytes and reject undecodable content before inference."""

    service, _, encoder = _service(max_upload_bytes=10)

    with pytest.raises(ImageQueryTooLargeError):
        service.search(
            _image_bytes(),
            content_type="image/jpeg",
            top_k=1,
        )
    with pytest.raises(InvalidImageQueryError):
        service.search(b"bad", content_type="image/jpeg", top_k=1)

    assert encoder.images == []


def test_image_search_rejects_excessive_decoded_pixels() -> None:
    """Apply the pixel bound before SigLIP2 receives the decoded image."""

    service, _, encoder = _service(max_pixels=3)

    with pytest.raises(ImageQueryTooLargeError):
        service.search(
            _image_bytes(size=(2, 2)),
            content_type="image/jpeg",
            top_k=1,
        )

    assert encoder.images == []
