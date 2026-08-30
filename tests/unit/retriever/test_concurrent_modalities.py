"""Concurrent modality search and partial-failure behavior."""

from __future__ import annotations

from time import perf_counter, sleep

import numpy as np
import pytest

from hcmai.common.config import EncoderConfig, FusionConfig
from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    RetrievalTrace,
)
from hcmai.retrieval.retriever.concurrent import (
    ModalitySearchExecutor,
    RequiredModalitySearchError,
)
from hcmai.retrieval.retriever.fusion import RRFFusionRetriever
from hcmai.retrieval.retriever.query_batch import SourceFamily, encode_query_batch


class FixtureEncoder:
    def __init__(self, model_name: str) -> None:
        self.config = EncoderConfig(model_name=model_name)
        self.embedding_dim = 2

    def encode_text(self, texts, stats=None) -> np.ndarray:
        del stats
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class DelayedRetriever:
    def __init__(
        self,
        source: RetrievalSource,
        encoder: FixtureEncoder,
        *,
        delay_seconds: float = 0.0,
        failure: Exception | None = None,
    ) -> None:
        self.source = source
        self.encoder = encoder
        self.delay_seconds = delay_seconds
        self.failure = failure

    @property
    def source_family(self) -> SourceFamily:
        return "visual" if self.source is RetrievalSource.VISUAL else "text"

    def encode(self, queries):
        return encode_query_batch(queries, self.encoder, self.source_family)

    def search_vectors(
        self,
        query_batch,
        top_k,
    ):
        sleep(self.delay_seconds)
        if self.failure is not None:
            raise self.failure
        return [
            RetrievalResult(
                candidates=[
                    RetrievalCandidate(
                        frame_id=f"{self.source.value}-{index}",
                        source_scores={self.source: 1.0},
                        source_ranks={self.source: 1},
                    )
                ][:top_k],
                trace=RetrievalTrace(),
            )
            for index, _ in enumerate(query_batch.embeddings)
        ]


def _fusion(
    retrievers,
    *,
    required_sources: set[RetrievalSource] | None = None,
    max_workers: int = 4,
):
    config = FusionConfig(
        modality_max_workers=max_workers,
        required_sources=(
            {RetrievalSource.VISUAL}
            if required_sources is None
            else required_sources
        ),
    )
    executor = ModalitySearchExecutor(max_workers)
    return RRFFusionRetriever(retrievers, config, executor), executor


def test_modality_delays_overlap_for_every_source() -> None:
    visual_encoder = FixtureEncoder("fixture/visual")
    text_encoder = FixtureEncoder("fixture/text")
    retrievers = [
        DelayedRetriever(RetrievalSource.VISUAL, visual_encoder, delay_seconds=0.05),
        *[
            DelayedRetriever(source, text_encoder, delay_seconds=0.05)
            for source in (
                RetrievalSource.CAPTION,
                RetrievalSource.OCR,
                RetrievalSource.ASR,
            )
        ],
    ]
    fusion, executor = _fusion(retrievers)
    try:
        started = perf_counter()
        result = fusion.search("event")
        elapsed = perf_counter() - started
    finally:
        executor.close()

    assert len(result.candidates) == 4
    assert elapsed < 0.14


def test_optional_caption_failure_warns_and_normalizes_active_weights() -> None:
    visual = DelayedRetriever(
        RetrievalSource.VISUAL,
        FixtureEncoder("fixture/visual"),
    )
    caption = DelayedRetriever(
        RetrievalSource.CAPTION,
        FixtureEncoder("fixture/text"),
        failure=TimeoutError("private backend detail"),
    )
    fusion, executor = _fusion([visual, caption])
    try:
        result = fusion.search("event")
    finally:
        executor.close()

    assert [candidate.frame_id for candidate in result] == ["visual-0"]
    assert result[0].fusion_score == pytest.approx(2 / 61)
    assert result.warnings == ["caption retrieval unavailable (TimeoutError)"]
    assert result.trace.stages["caption.search"].error_category == "TimeoutError"
    assert "private backend detail" not in result.warnings[0]


def test_absent_asr_source_does_not_fail_or_warn() -> None:
    retrievers = [
        DelayedRetriever(
            RetrievalSource.VISUAL,
            FixtureEncoder("fixture/visual"),
        ),
        DelayedRetriever(
            RetrievalSource.CAPTION,
            FixtureEncoder("fixture/text"),
        ),
    ]
    fusion, executor = _fusion(retrievers)
    try:
        result = fusion.search("event")
    finally:
        executor.close()

    assert len(result.candidates) == 2
    assert result.warnings == []


def test_required_visual_failure_fails_request() -> None:
    visual = DelayedRetriever(
        RetrievalSource.VISUAL,
        FixtureEncoder("fixture/visual"),
        failure=RuntimeError("visual offline"),
    )
    caption = DelayedRetriever(
        RetrievalSource.CAPTION,
        FixtureEncoder("fixture/text"),
    )
    fusion, executor = _fusion([visual, caption])
    try:
        with pytest.raises(RequiredModalitySearchError) as caught:
            fusion.search("event")
    finally:
        executor.close()

    assert caught.value.source is RetrievalSource.VISUAL
    assert caught.value.category == "RuntimeError"
