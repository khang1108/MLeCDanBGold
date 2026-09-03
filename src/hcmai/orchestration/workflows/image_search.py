"""Direct image-query search over the canonical SigLIP2 visual index.

This module owns bounded image decoding, visual query encoding, and canonical
result materialization. It does not perform text retrieval, BM25 fusion, or
temporal multi-event alignment.
"""

from __future__ import annotations

import warnings
from io import BytesIO
from time import perf_counter

from PIL import Image, UnidentifiedImageError

from hcmai.api.contracts import ImageSearchResponse, SearchLatency, SearchResult
from hcmai.corpus import Corpus
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.retrieval.embedding.models.contracts import ImageEmbeddingAdapter
from hcmai.retrieval.models import RetrievalCandidate, RetrievalSource
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.query_batch import encode_image_query_batch
from hcmai.temporal import AlignedPath


class InvalidImageQueryError(ValueError):
    """The uploaded payload is not one supported, safely bounded image."""


class ImageQueryTooLargeError(InvalidImageQueryError):
    """The uploaded image exceeds the configured byte or pixel limit."""


class ImageSearchService:
    """Encode one image with SigLIP2 and search the existing visual index."""

    SUPPORTED_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})

    def __init__(
        self,
        corpus: Corpus,
        retrieval: RetrievalService,
        encoder: ImageEmbeddingAdapter,
        *,
        max_upload_bytes: int,
        max_pixels: int,
    ) -> None:
        """Bind canonical data, visual retrieval, and bounded upload settings."""

        visual = retrieval.source_retriever(RetrievalSource.VISUAL)
        if visual is None:
            raise ValueError("visual retriever is required for image search")
        if max_upload_bytes <= 0 or max_pixels <= 0:
            raise ValueError("image upload limits must be positive")

        self.corpus = corpus
        self.visual = visual
        self.encoder = encoder
        self.max_upload_bytes = max_upload_bytes
        self.max_pixels = max_pixels
        self.materializer = SearchMaterializer(corpus)

    def search(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        top_k: int,
    ) -> ImageSearchResponse:
        """Decode, embed, and retrieve one uploaded visual query."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if content_type not in self.SUPPORTED_MEDIA_TYPES:
            raise InvalidImageQueryError(
                "image must use JPEG, PNG, or WebP media type"
            )
        if not payload:
            raise InvalidImageQueryError("image payload must not be empty")
        if len(payload) > self.max_upload_bytes:
            raise ImageQueryTooLargeError(
                f"image payload exceeds {self.max_upload_bytes} bytes"
            )

        started = perf_counter()
        image = self._decode(payload)

        batch = encode_image_query_batch([image], self.encoder)
        retrieved = self.visual.search_vectors(batch, top_k)[0]

        materialization_started = perf_counter()
        results = [self._materialize(candidate) for candidate in retrieved]
        materialization_ms = (perf_counter() - materialization_started) * 1_000
        total_ms = (perf_counter() - started) * 1_000

        return ImageSearchResponse(
            results=results,
            latency=SearchLatency(
                query_ms=batch.encoding_trace.duration_ms,
                retrieval_ms=retrieved.trace.duration_for("search"),
                materialization_ms=materialization_ms,
                total_ms=total_ms,
            ),
        )

    def _decode(self, payload: bytes) -> Image.Image:
        """Decode one static RGB image without accepting decompression bombs."""

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(payload)) as source:
                    width, height = source.size
                    if width <= 0 or height <= 0:
                        raise InvalidImageQueryError("image dimensions must be positive")
                    if width * height > self.max_pixels:
                        raise ImageQueryTooLargeError(
                            f"image exceeds {self.max_pixels} decoded pixels"
                        )
                    if getattr(source, "n_frames", 1) != 1:
                        raise InvalidImageQueryError("animated images are not supported")
                    return source.convert("RGB")
        except ImageQueryTooLargeError:
            raise
        except InvalidImageQueryError:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as error:
            raise ImageQueryTooLargeError(
                "image exceeds the safe decoded size"
            ) from error
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise InvalidImageQueryError("image payload cannot be decoded") from error

    def _materialize(self, candidate: RetrievalCandidate) -> SearchResult:
        """Resolve one retrieved frame through canonical Corpus identity."""

        frame = self.corpus.frame(candidate.frame_id)
        score = candidate.source_scores.get(RetrievalSource.VISUAL)
        if score is None:
            raise ValueError("visual candidate is missing its source score")

        return self.materializer.build_kis_result(
            AlignedPath(
                video_id=frame.video_id,
                score=score,
                frame_ids=(frame.frame_id,),
                frame_idxs=(frame.frame_idx,),
                timestamps_ms=(frame.timestamp_ms,),
            )
        )
