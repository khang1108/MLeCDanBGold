"""Composition tests for the explicit fast-track retrieval factory.

The fixtures use tiny exact indexes so these tests exercise only online
composition.  They do not build corpus artifacts or load production models.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest

from hcmai.common.config import (
    EncoderConfig,
    FusionConfig,
    RetrievalCacheConfig,
)
from hcmai.common.schemas import RetrievalSource
from hcmai.data.stores.frame import FrameStore
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex


class FakeEncoder:
    """Return deterministic normalized query vectors and record batch calls."""

    embedding_dim = 2
    resolved_revision = "a" * 40

    def __init__(
        self,
        model_name: str,
        backend: Literal["siglip", "bge_m3"],
    ) -> None:
        self.config = EncoderConfig(
            backend=backend,
            model_name=model_name,
            revision=self.resolved_revision,
        )
        self.calls: list[list[str]] = []

    def encode_text(self, texts, stats=None) -> np.ndarray:
        """Map cable-car queries to the first axis and all others to the second."""

        del stats
        self.calls.append(list(texts))
        return np.asarray(
            [
                [1.0, 0.0] if "cable" in text.lower() else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def _frame_mapping() -> pd.DataFrame:
    """Return two canonical frame rows shared by the frame-native indexes."""

    return pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 101,
                "timestamp_ms": 1_500,
            },
            {
                "embedding_index": 1,
                "frame_id": "f2",
                "video_id": "v1",
                "frame_idx": 202,
                "timestamp_ms": 5_000,
            },
        ]
    )


def _frame_store(tmp_path: Path) -> FrameStore:
    """Persist the canonical coordinates used for ASR segment projection."""

    pytest.importorskip("pyarrow")
    path = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 101,
                "timestamp_ms": 1_500,
                "image_path": "frames/f1.jpg",
                "width": 640,
                "height": 360,
            },
            {
                "frame_id": "f2",
                "video_id": "v1",
                "frame_idx": 202,
                "timestamp_ms": 5_000,
                "image_path": "frames/f2.jpg",
                "width": 640,
                "height": 360,
            },
        ]
    ).to_parquet(path, index=False)
    return FrameStore(path)


def _indexes() -> tuple[DenseIndex, DenseIndex, SegmentDenseIndex]:
    """Build visual, Context, and timeline ASR indexes with aligned fixtures."""

    pytest.importorskip("faiss")
    visual = DenseIndex.build(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        _frame_mapping(),
        dataset_version="test-v1",
        model_name="fake/siglip",
        show_progress=False,
    )
    context = DenseIndex.build(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        _frame_mapping(),
        dataset_version="test-v1",
        model_name="fake/bge",
        show_progress=False,
    )
    segments = SegmentDenseIndex.build(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        pd.DataFrame(
            [
                {
                    "embedding_index": 0,
                    "segment_id": "v1:strong",
                    "video_id": "v1",
                    "segment_index": 0,
                    "start_ms": 1_000,
                    "end_ms": 2_000,
                },
                {
                    "embedding_index": 1,
                    "segment_id": "v1:second",
                    "video_id": "v1",
                    "segment_index": 1,
                    "start_ms": 4_500,
                    "end_ms": 5_500,
                },
            ]
        ),
        dataset_version="test-v1",
        model_name="fake/bge",
    )
    return visual, context, segments


def _encoders() -> tuple[FakeEncoder, FakeEncoder]:
    """Return distinct visual-query and generic evidence-query encoders."""

    return (
        FakeEncoder("fake/siglip", "siglip"),
        FakeEncoder("fake/bge", "bge_m3"),
    )


def _service(
    tmp_path: Path,
    *,
    include_context: bool = True,
    include_asr: bool = True,
    cache_config: RetrievalCacheConfig | None = None,
) -> tuple[RetrievalService, FakeEncoder, FakeEncoder, DenseIndex]:
    """Compose the fast-track service with selected optional evidence indexes."""

    visual, context, segments = _indexes()
    visual_encoder, text_encoder = _encoders()
    service = RetrievalService.from_fast_track_indexes(
        visual_index=visual,
        visual_encoder=visual_encoder,
        context_index=context if include_context else None,
        asr_segment_index=segments if include_asr else None,
        text_encoder=text_encoder if include_context or include_asr else None,
        frame_store=_frame_store(tmp_path),
        fusion=FusionConfig(required_sources={RetrievalSource.VISUAL}),
        cache_config=cache_config,
        max_projection_gap_ms=1_000,
    )
    return service, visual_encoder, text_encoder, visual


def test_fast_track_service_returns_frame_native_fused_candidates(
    tmp_path: Path,
) -> None:
    """Fuse all modalities by canonical frame ID after ASR projection."""

    service, _, text_encoder, _ = _service(tmp_path)

    result = service.search("red cable car", top_k=10)

    assert service.active_sources == (
        RetrievalSource.VISUAL,
        RetrievalSource.CONTEXT,
        RetrievalSource.ASR,
    )
    assert tuple(
        type(retriever).__name__ for retriever in service._retriever.retrievers  # type: ignore[attr-defined]
    ) == ("DenseRetriever", "ContextRetriever", "ASRSegmentRetriever")
    assert [candidate.frame_id for candidate in result.candidates] == ["f1", "f2"]
    first = result.candidates[0]
    assert set(first.source_ranks) == {
        RetrievalSource.VISUAL,
        RetrievalSource.CONTEXT,
        RetrievalSource.ASR,
    }
    assert first.metadata["frame"] == {
        "frame_id": "f1",
        "video_id": "v1",
        "frame_idx": 101,
        "timestamp_ms": 1_500,
    }
    assert first.metadata["asr_segment"]["segment_id"] == "v1:strong"
    assert text_encoder.calls == [["red cable car"]]


@pytest.mark.parametrize(
    ("include_context", "include_asr", "expected_sources"),
    [
        (False, True, (RetrievalSource.VISUAL, RetrievalSource.ASR)),
        (True, False, (RetrievalSource.VISUAL, RetrievalSource.CONTEXT)),
        (False, False, (RetrievalSource.VISUAL,)),
    ],
)
def test_fast_track_optional_indexes_are_independent(
    tmp_path: Path,
    include_context: bool,
    include_asr: bool,
    expected_sources: tuple[RetrievalSource, ...],
) -> None:
    """Keep Visual required while either or both optional indexes are absent."""

    service, _, _, _ = _service(
        tmp_path,
        include_context=include_context,
        include_asr=include_asr,
    )

    assert service.active_sources == expected_sources
    assert service.search("red cable car", top_k=2).candidates
    if not include_context and not include_asr:
        assert not hasattr(service._retriever, "retrievers")


@pytest.mark.parametrize("include_context,include_asr", [(True, False), (False, True)])
def test_fast_track_requires_text_encoder_for_any_text_index(
    tmp_path: Path, include_context: bool, include_asr: bool
) -> None:
    """Reject an optional text index when no compatible query encoder exists."""

    visual, context, segments = _indexes()
    visual_encoder, _ = _encoders()

    with pytest.raises(ValueError, match="text_encoder"):
        RetrievalService.from_fast_track_indexes(
            visual_index=visual,
            visual_encoder=visual_encoder,
            context_index=context if include_context else None,
            asr_segment_index=segments if include_asr else None,
            text_encoder=None,
            frame_store=_frame_store(tmp_path),
            fusion=FusionConfig(required_sources={RetrievalSource.VISUAL}),
        )


def test_fast_track_retrievers_share_one_enabled_cache(tmp_path: Path) -> None:
    """Reuse one cache and prompt version across all composed retrievers."""

    cache_config = RetrievalCacheConfig(enabled=True, prompt_version="fast-track-v1")
    service, _, _, _ = _service(tmp_path, cache_config=cache_config)
    retrievers = service._retriever.retrievers  # type: ignore[attr-defined]

    caches = [retriever.embedding_cache for retriever in retrievers]
    assert caches[0] is not None
    assert all(cache is caches[0] for cache in caches)
    assert {retriever.prompt_version for retriever in retrievers} == {"fast-track-v1"}


def test_fast_track_visual_scoring_uses_visual_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protect TRAKE's visual-only video scoring path in multimodal services."""

    service, visual_encoder, _, visual_index = _service(tmp_path)
    captured: dict[str, Any] = {}

    def fake_score(index, vectors, *args):
        captured.update(index=index, vectors=vectors, args=args)
        return []

    monkeypatch.setattr(
        "hcmai.retrieval.retriever.pipeline.score_videos", fake_score
    )

    assert service.score_visual_videos(["red cable car"]) == []
    assert captured["index"] is visual_index
    assert visual_encoder.calls == [["red cable car"]]


def test_offline_index_cli_all_runs_strict_sequential_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build all corpora in GPU-safe order and validate only after publication."""

    from scripts import build_retrieval_indexes as workflow

    events: list[str] = []
    projected_frames = tmp_path / "projected.parquet"
    projected_frames.write_bytes(b"preflight-projection")
    config = SimpleNamespace(projected_frames_path=projected_frames)
    models = object()
    text_encoder = object()

    monkeypatch.setattr(workflow, "load_offline_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(workflow, "load_model_config", lambda *args, **kwargs: models)
    monkeypatch.setattr(
        workflow,
        "run_preflight",
        lambda received: events.append("preflight") or projected_frames,
    )
    monkeypatch.setattr(
        workflow,
        "build_visual",
        lambda received, received_models, frames: events.append("visual"),
    )
    monkeypatch.setattr(
        workflow,
        "create_text_encoder",
        lambda received_models: text_encoder,
    )
    monkeypatch.setattr(
        workflow,
        "build_context",
        lambda received, received_models, frames, encoder=None: events.append(
            "context"
        ),
    )
    monkeypatch.setattr(
        workflow,
        "build_asr",
        lambda received, received_models, encoder=None: events.append("asr"),
    )
    monkeypatch.setattr(
        workflow,
        "run_validate",
        lambda received, received_models: events.append("validate"),
    )
    monkeypatch.setattr(workflow, "release_gpu_memory", lambda: None)

    workflow.run(workflow.parse_args(["--stage", "all"]))

    assert events == ["preflight", "visual", "context", "asr", "validate"]


def test_offline_index_cli_context_does_not_build_other_modalities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permit one text-index rebuild without loading Visual or ASR builders."""

    from scripts import build_retrieval_indexes as workflow

    events: list[str] = []
    projected_frames = tmp_path / "projected.parquet"
    projected_frames.write_bytes(b"preflight-projection")
    config = SimpleNamespace(projected_frames_path=projected_frames)

    monkeypatch.setattr(workflow, "load_offline_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(workflow, "load_model_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(workflow, "build_visual", lambda *args, **kwargs: events.append("visual"))
    monkeypatch.setattr(workflow, "build_context", lambda *args, **kwargs: events.append("context"))
    monkeypatch.setattr(workflow, "build_asr", lambda *args, **kwargs: events.append("asr"))

    workflow.run(workflow.parse_args(["--stage", "context"]))

    assert events == ["context"]


def test_offline_projection_resolves_relative_keyframe_root_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a config-relative keyframe path directly readable by the builder."""

    from scripts import build_retrieval_indexes as workflow
    from hcmai.retrieval.embedding.artifacts import EmbeddingArtifactBuilder

    monkeypatch.chdir(tmp_path)
    keyframes_root = Path("data/keyframes")
    video_root = keyframes_root / "v1"
    video_root.mkdir(parents=True)
    image = video_root / "000001.jpg"
    image.write_bytes(b"fixture-image")
    frames = pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "video_id": "v1",
                "frame_idx": 10,
                "timestamp_ms": 400,
                "keyframe_order": 1,
                "image_path": "machine-specific/old.jpg",
            }
        ]
    )

    projected = workflow.project_staged_keyframes(frames, keyframes_root)
    builder = EmbeddingArtifactBuilder(
        frames_path=tmp_path / "unused.parquet",
        dataset_root=keyframes_root,
        output_dir=tmp_path / "output",
        encoder_config=EncoderConfig(),
        encoder=object(),  # type: ignore[arg-type]
    )

    assert Path(projected.iloc[0]["image_path"]).is_absolute()
    assert builder._resolve_image(projected.iloc[0]["image_path"]) == image.resolve()
