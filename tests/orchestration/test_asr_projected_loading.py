"""Tests for projecting segment-native ASR into Dense temporal evidence.

These tests exercise startup composition only. They do not validate projection
algorithms, which belong to the retrieval evidence contract tests.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from hcmai.common.config import AppConfig
from hcmai.common.config import REPOSITORY_ROOT
from hcmai.corpus import Corpus
from hcmai.orchestration import setup
from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex
from hcmai.retrieval.models import RetrievalSource
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
from hcmai.retrieval.retriever.segment.retriever import ASRSegmentRetriever


_PRODUCTION_INDEX_PATHS = (
    Path("artifacts/indexes/visual"),
    Path("artifacts/indexes/context"),
    Path("artifacts/indexes/asr_segments"),
)


def _require_production_indexes() -> None:
    """Skip only when the required published Dense bundles are unavailable."""

    missing = [
        str(path)
        for path in _PRODUCTION_INDEX_PATHS
        if not (REPOSITORY_ROOT / path).is_dir()
    ]
    if missing:
        pytest.skip(
            "production Dense temporal artifacts are not mounted: "
            + ", ".join(missing)
        )


def _dense_bindings(
    *,
    context_dimension: int = 4,
    asr_dimension: int = 4,
) -> tuple[Any, Any, Any]:
    """Create small retriever bindings with distinct index dependencies."""

    visual = SimpleNamespace(index=object(), encoder=object())
    context = SimpleNamespace(
        index=SimpleNamespace(
            metadata=SimpleNamespace(embedding_dim=context_dimension),
        ),
        encoder=object(),
    )
    asr = SimpleNamespace(
        index=SimpleNamespace(
            metadata=SimpleNamespace(embedding_dim=asr_dimension),
        ),
        projector=object(),
    )
    return visual, context, asr


def _retrieval(*bindings: tuple[RetrievalSource, Any]) -> tuple[Any, list[RetrievalSource]]:
    """Build a source lookup that records the dependencies setup requests."""

    sources = dict(bindings)
    requested: list[RetrievalSource] = []

    def source_retriever(source: RetrievalSource) -> Any | None:
        requested.append(source)
        return sources.get(source)

    return SimpleNamespace(source_retriever=source_retriever), requested


def test_load_dense_temporal_reuses_asr_segment_retriever(monkeypatch: Any) -> None:
    """Build projected ASR from the existing fast-track retriever only."""

    visual, context, asr = _dense_bindings()
    retrieval, requested = _retrieval(
        (RetrievalSource.CONTEXT, context),
        (RetrievalSource.ASR, asr),
    )
    captured: dict[str, Any] = {}
    projected_asr = object()
    dense = object()

    def fail_legacy_asr_load(path: object) -> object:
        raise AssertionError(f"legacy frame-ASR index load attempted: {path}")

    def build_projected_asr(**kwargs: Any) -> object:
        captured["projected"] = kwargs
        return projected_asr

    def build_dense(**kwargs: Any) -> object:
        captured["dense"] = kwargs
        return dense

    monkeypatch.setattr(setup.RetrievalService, "load_index", fail_legacy_asr_load)
    monkeypatch.setattr(
        setup,
        "SegmentProjectedASRIndex",
        build_projected_asr,
        raising=False,
    )
    monkeypatch.setattr(setup, "DenseTemporalScorer", build_dense)

    scorer, context_ready, asr_ready = setup._load_dense_temporal(
        AppConfig(),
        cast(Any, retrieval),
        visual,
        [],
    )

    assert scorer is dense
    assert (context_ready, asr_ready) == (True, True)
    assert requested == [RetrievalSource.CONTEXT, RetrievalSource.ASR]
    assert captured["projected"] == {
        "segment_index": asr.index,
        "canonical_index": visual.index,
        "projector": asr.projector,
    }
    assert captured["dense"]["asr_index"] is projected_asr
    assert captured["dense"]["text_encoder"] is context.encoder


def test_load_dense_temporal_reports_missing_context() -> None:
    """Keep Dense unavailable when its frame-native Context evidence is absent."""

    visual, _, asr = _dense_bindings()
    retrieval, _ = _retrieval((RetrievalSource.ASR, asr))
    messages: list[str] = []

    scorer, context_ready, asr_ready = setup._load_dense_temporal(
        AppConfig(),
        cast(Any, retrieval),
        visual,
        messages,
    )

    assert scorer is None
    assert (context_ready, asr_ready) == (False, True)
    assert messages == ["Dense temporal evidence unavailable: Context retriever missing"]


def test_load_dense_temporal_reports_missing_asr_segment_retriever() -> None:
    """Keep Dense unavailable when projected segment-ASR cannot be supplied."""

    visual, context, _ = _dense_bindings()
    retrieval, _ = _retrieval((RetrievalSource.CONTEXT, context))
    messages: list[str] = []

    scorer, context_ready, asr_ready = setup._load_dense_temporal(
        AppConfig(),
        cast(Any, retrieval),
        visual,
        messages,
    )

    assert scorer is None
    assert (context_ready, asr_ready) == (True, False)
    assert messages == ["Dense temporal evidence unavailable: ASR segment retriever missing"]


def test_load_dense_temporal_marks_asr_unready_when_projection_fails(
    monkeypatch: Any,
) -> None:
    """Do not advertise projected ASR readiness after projection validation fails."""

    visual, context, asr = _dense_bindings()
    retrieval, _ = _retrieval(
        (RetrievalSource.CONTEXT, context),
        (RetrievalSource.ASR, asr),
    )
    messages: list[str] = []

    def fail_projection(**kwargs: Any) -> object:
        raise ValueError("projected frame identity conflicts")

    monkeypatch.setattr(
        setup,
        "SegmentProjectedASRIndex",
        fail_projection,
        raising=False,
    )

    scorer, context_ready, asr_ready = setup._load_dense_temporal(
        AppConfig(),
        cast(Any, retrieval),
        visual,
        messages,
    )

    assert scorer is None
    assert (context_ready, asr_ready) == (True, False)
    assert messages == [
        "Dense temporal ASR projection failed: ValueError: "
        "projected frame identity conflicts"
    ]


def test_load_dense_temporal_rejects_incompatible_context_and_asr_dimensions(
    monkeypatch: Any,
) -> None:
    """Disable Dense when the shared BGE encoder cannot serve both indexes."""

    visual, context, asr = _dense_bindings(context_dimension=4, asr_dimension=8)
    retrieval, _ = _retrieval(
        (RetrievalSource.CONTEXT, context),
        (RetrievalSource.ASR, asr),
    )
    messages: list[str] = []

    monkeypatch.setattr(
        setup,
        "SegmentProjectedASRIndex",
        lambda **kwargs: object(),
        raising=False,
    )
    monkeypatch.setattr(
        setup,
        "DenseTemporalScorer",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Dense was constructed")),
    )

    scorer, context_ready, asr_ready = setup._load_dense_temporal(
        AppConfig(),
        cast(Any, retrieval),
        visual,
        messages,
    )

    assert scorer is None
    assert (context_ready, asr_ready) == (True, True)
    assert messages == [
        "Dense temporal evidence identity validation failed: ValueError: "
        "Context and ASR segment index dimensions differ"
    ]


def test_production_artifacts_project_segment_asr_to_canonical_frames() -> None:
    """Validate mounted segment-ASR projection identity without reindexing.

    This test deliberately follows the fast-track runtime construction: it
    looks up the visual index's canonical IDs through ``Corpus`` and lets
    ``ASRSegmentRetriever`` construct ``SegmentFrameProjector`` with the
    configured gap. Production bundles are optional in ordinary test
    environments, so only their absence is a skip condition; invalid mounted
    bundles must fail visibly.
    """

    _require_production_indexes()

    visual = DenseIndex.load(REPOSITORY_ROOT / "artifacts/indexes/visual")
    context = DenseIndex.load(REPOSITORY_ROOT / "artifacts/indexes/context")
    asr_segments = SegmentDenseIndex.load(
        REPOSITORY_ROOT / "artifacts/indexes/asr_segments"
    )

    settings = setup._load_app_config()
    corpus = Corpus.open(REPOSITORY_ROOT / settings.dataset.frames_path)
    canonical_frames = tuple(
        corpus.frames(
            [str(frame_id) for frame_id in visual.mapping["frame_id"]]
        )
    )
    text_encoder = SimpleNamespace(
        config=SimpleNamespace(model_name=asr_segments.metadata.model_name),
        embedding_dim=asr_segments.metadata.embedding_dim,
    )
    asr_retriever = ASRSegmentRetriever(
        text_encoder,
        asr_segments,
        canonical_frames,
        max_projection_gap_ms=settings.index.asr_projection_max_gap_ms,
    )
    projected = SegmentProjectedASRIndex(
        segment_index=asr_segments,
        canonical_index=visual,
        projector=asr_retriever.projector,
    )

    assert len(projected.frame_ids) == len(visual.frame_ids)
    np.testing.assert_array_equal(projected.frame_ids, visual.frame_ids)
    assert projected.metadata.embedding_dim == context.metadata.embedding_dim
    assert np.any(projected.segment_frame_positions >= 0)
    assert np.all(
        (projected.segment_frame_positions == -1)
        | (
            (projected.segment_frame_positions >= 0)
            & (projected.segment_frame_positions < len(visual.frame_ids))
        )
    )


def test_production_artifacts_satisfy_fast_track_lineage() -> None:
    """Require mounted Context and ASR bundles to pass runtime validation.

    Projection identity alone is insufficient for an online Dense temporal
    capability: setup must also accept each published artifact's configured
    BGE model and revision. This exercises the exact startup loader instead of
    fabricating a compatible encoder from artifact metadata.
    """

    _require_production_indexes()

    settings = setup._load_app_config()
    models = setup._load_model_config()
    visual = setup.RetrievalService.load_index(
        REPOSITORY_ROOT / "artifacts/indexes/visual"
    )
    messages: list[str] = []
    encoder_config = models.resolved_evidence_embedding
    context = setup._load_fast_track_index(
        source=RetrievalSource.CONTEXT,
        path=REPOSITORY_ROOT / "artifacts/indexes/context",
        visual=visual,
        encoder_config=encoder_config,
        settings=settings,
        messages=messages,
    )
    asr_segments = setup._load_fast_track_index(
        source=RetrievalSource.ASR,
        path=REPOSITORY_ROOT / "artifacts/indexes/asr_segments",
        visual=visual,
        encoder_config=encoder_config,
        settings=settings,
        messages=messages,
    )

    assert context is not None and asr_segments is not None, "\n".join(messages)
