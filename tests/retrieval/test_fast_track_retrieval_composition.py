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
from hcmai.common.schemas import (
    InferenceCapabilities,
    InferenceReadiness,
    ModelStatus,
    RetrievalSource,
)
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


class FakeRemoteService:
    """Expose a checked remote client without making HTTP requests in CLI tests."""

    def __init__(self, readiness: InferenceReadiness, events: list[str]) -> None:
        self.adapter = object()
        self._readiness = readiness
        self._events = events
        self.closed = False

    def readiness(self) -> InferenceReadiness:
        """Return the fixed GPU-service capability snapshot."""

        self._events.append("remote-ready")
        return self._readiness

    def close(self) -> None:
        """Record client cleanup after the staged offline workflow finishes."""

        self.closed = True
        self._events.append("remote-close")


def _remote_models() -> SimpleNamespace:
    """Return pinned Visual and evidence configs used by remote CLI fixtures."""

    revision = "b" * 40
    return SimpleNamespace(
        visual_embedding=EncoderConfig(
            backend="siglip",
            model_name="fake/siglip",
            revision=revision,
        ),
        resolved_evidence_embedding=EncoderConfig(
            backend="bge_m3",
            model_name="fake/bge",
            revision=revision,
        ),
    )


def _remote_readiness(models: SimpleNamespace) -> InferenceReadiness:
    """Advertise the exact pinned SigLIP and BGE models for a test worker."""

    return InferenceReadiness(
        ready=True,
        models={
            "visual_embedding": ModelStatus(
                loaded=True,
                checkpoint=models.visual_embedding.model_name,
                revision=models.visual_embedding.revision,
            ),
            "caption_embedding": ModelStatus(
                loaded=True,
                checkpoint=models.resolved_evidence_embedding.model_name,
                revision=models.resolved_evidence_embedding.revision,
            ),
        },
        capabilities=InferenceCapabilities(
            embedding=True,
            image_embedding=True,
        ),
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
        "hcmai.retrieval.retriever.pipeline.score_all_videos", fake_score
    )

    assert service.score_event_videos(["red cable car"]) == []
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


def test_s3_index_cli_downloads_builds_validates_then_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish only after every local batch stage and validation succeeds."""

    from scripts import build_retrieval_indexes as workflow

    events: list[str] = []
    projection = tmp_path / "projected.parquet"
    projection.write_bytes(b"projection")
    config = SimpleNamespace(
        projected_frames_path=projection,
        output_root=tmp_path / "indexes",
    )
    client = object()
    text_encoder = object()

    monkeypatch.setattr(workflow, "load_offline_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(workflow, "load_model_config", lambda *args: object())
    monkeypatch.setattr(
        workflow,
        "_load_s3_transport",
        lambda path: (events.append("open") or client, "bucket"),
    )
    monkeypatch.setattr(
        workflow,
        "_download_s3_inputs",
        lambda *args: events.append("download"),
    )
    monkeypatch.setattr(
        workflow,
        "run_preflight",
        lambda received: events.append("preflight") or projection,
    )
    monkeypatch.setattr(
        workflow,
        "build_visual",
        lambda *args, **kwargs: events.append("visual"),
    )
    monkeypatch.setattr(
        workflow,
        "release_gpu_memory",
        lambda: events.append("release-gpu"),
    )
    monkeypatch.setattr(
        workflow,
        "create_text_encoder",
        lambda *args: events.append("load-text") or text_encoder,
    )
    monkeypatch.setattr(
        workflow,
        "build_context",
        lambda *args, **kwargs: events.append("context"),
    )
    monkeypatch.setattr(
        workflow,
        "build_asr",
        lambda *args, **kwargs: events.append("asr"),
    )
    monkeypatch.setattr(
        workflow,
        "run_validate",
        lambda *args: events.append("validate"),
    )
    monkeypatch.setattr(
        workflow,
        "_publish_s3_bundle",
        lambda *args: events.append("publish"),
    )
    monkeypatch.setattr(
        workflow,
        "_close_s3_transport",
        lambda received: events.append("close"),
    )

    args = workflow.parse_args(["--s3", "--stage", "all"])
    assert args.s3_sync_workers == 8
    workflow.run(args)

    assert events == [
        "open",
        "download",
        "preflight",
        "visual",
        "release-gpu",
        "load-text",
        "context",
        "asr",
        "validate",
        "publish",
        "close",
    ]


def test_offline_index_cli_all_uses_explicit_remote_embedding_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject one checked remote SigLIP+BGE pair only when the CLI URL is set."""

    from hcmai.common.config import InferenceConfig
    from thundercompute.pipeline import LLMService
    from hcmai.retrieval.embedding.pipeline import EmbeddingService
    from scripts import build_retrieval_indexes as workflow

    events: list[str] = []
    projected_frames = tmp_path / "projected.parquet"
    projected_frames.write_bytes(b"preflight-projection")
    config = SimpleNamespace(projected_frames_path=projected_frames)
    models = _remote_models()
    service = FakeRemoteService(_remote_readiness(models), events)
    remote_visual = object()
    remote_text = object()
    captured: dict[str, object] = {}

    monkeypatch.setenv("HCMAI_INFERENCE_BASE_URL", "https://must-not-be-used.test")
    monkeypatch.setattr(workflow, "load_offline_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(workflow, "load_model_config", lambda *args, **kwargs: models)

    def fake_remote(
        cls: type[LLMService], base_url: str, inference: InferenceConfig
    ) -> FakeRemoteService:
        del cls
        captured.update(base_url=base_url, inference=inference)
        return service

    monkeypatch.setattr(LLMService, "remote", classmethod(fake_remote))

    def create_remote_visual_adapter(client, encoder_config):
        captured.update(
            visual_client=client,
            visual_config=encoder_config,
        )
        return remote_visual

    def create_remote_adapter(client, encoder_config, embedding_dim, source):
        captured.update(
            text_client=client,
            text_config=encoder_config,
            text_dim=embedding_dim,
            text_source=source,
        )
        return remote_text

    monkeypatch.setattr(
        EmbeddingService,
        "create_remote_visual_adapter",
        create_remote_visual_adapter,
    )
    monkeypatch.setattr(
        EmbeddingService,
        "create_remote_adapter",
        create_remote_adapter,
    )
    monkeypatch.setattr(
        workflow,
        "run_preflight",
        lambda received: events.append("preflight") or projected_frames,
    )
    monkeypatch.setattr(
        workflow,
        "build_visual",
        lambda received, received_models, frames, *, encoder=None: captured.update(
            visual_encoder=encoder
        )
        or events.append("visual"),
    )
    monkeypatch.setattr(
        workflow,
        "release_gpu_memory",
        lambda: events.append("release-gpu"),
    )
    monkeypatch.setattr(
        workflow,
        "build_context",
        lambda received, received_models, frames, *, encoder=None: captured.update(
            context_encoder=encoder
        )
        or events.append("context"),
    )
    monkeypatch.setattr(
        workflow,
        "build_asr",
        lambda received, received_models, *, encoder=None: captured.update(
            asr_encoder=encoder
        )
        or events.append("asr"),
    )
    monkeypatch.setattr(
        workflow,
        "run_validate",
        lambda received, received_models: events.append("validate"),
    )
    monkeypatch.setattr(
        workflow,
        "create_text_encoder",
        lambda received: pytest.fail("remote mode must not load local BGE"),
    )

    workflow.run(
        workflow.parse_args(
            ["--stage", "all", "--inference-url", " https://gpu.test/ "]
        )
    )

    assert captured["base_url"] == "https://gpu.test/"
    assert isinstance(captured["inference"], InferenceConfig)
    assert captured["inference"].base_url == "https://gpu.test/"
    assert captured["visual_client"] is service.adapter
    assert captured["visual_config"] is models.visual_embedding
    assert captured["text_client"] is service.adapter
    assert captured["text_config"] is models.resolved_evidence_embedding
    assert captured["text_dim"] == 0
    assert captured["text_source"] == "text"
    assert captured["visual_encoder"] is remote_visual
    assert captured["context_encoder"] is remote_text
    assert captured["asr_encoder"] is remote_text
    assert events == [
        "remote-ready",
        "preflight",
        "visual",
        "release-gpu",
        "context",
        "asr",
        "validate",
        "remote-close",
    ]
    assert service.closed is True


def test_offline_index_cli_does_not_read_remote_url_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the no-flag workflow local even if another process set an endpoint."""

    from thundercompute.pipeline import LLMService
    from scripts import build_retrieval_indexes as workflow

    events: list[str] = []
    projected_frames = tmp_path / "projected.parquet"
    projected_frames.write_bytes(b"preflight-projection")
    config = SimpleNamespace(projected_frames_path=projected_frames)
    local_text = object()

    monkeypatch.setenv("HCMAI_INFERENCE_BASE_URL", "https://must-not-be-used.test")
    monkeypatch.setattr(workflow, "load_offline_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(workflow, "load_model_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        LLMService,
        "remote",
        classmethod(
            lambda cls, *args: pytest.fail(
                "local index workflow must not create a remote client"
            )
        ),
    )
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
    monkeypatch.setattr(workflow, "release_gpu_memory", lambda: None)
    monkeypatch.setattr(
        workflow,
        "create_text_encoder",
        lambda received: local_text,
    )
    monkeypatch.setattr(
        workflow,
        "build_context",
        lambda received, received_models, frames, *, encoder=None: events.append(
            "context"
        )
        if encoder is local_text
        else pytest.fail("local BGE adapter was not injected"),
    )
    monkeypatch.setattr(
        workflow,
        "build_asr",
        lambda received, received_models, *, encoder=None: events.append("asr")
        if encoder is local_text
        else pytest.fail("local BGE adapter was not reused"),
    )
    monkeypatch.setattr(
        workflow,
        "run_validate",
        lambda received, received_models: events.append("validate"),
    )

    workflow.run(workflow.parse_args(["--stage", "all"]))

    assert events == ["preflight", "visual", "context", "asr", "validate"]


def test_remote_embedding_readiness_stops_before_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail before corpus work when the endpoint lacks the required BGE model."""

    from thundercompute.pipeline import LLMService
    from scripts import build_retrieval_indexes as workflow

    events: list[str] = []
    config = SimpleNamespace(projected_frames_path=tmp_path / "projected.parquet")
    models = _remote_models()
    readiness = _remote_readiness(models).model_copy(
        update={
            "models": {
                "visual_embedding": ModelStatus(
                    loaded=True,
                    checkpoint=models.visual_embedding.model_name,
                    revision=models.visual_embedding.revision,
                )
            }
        }
    )
    service = FakeRemoteService(readiness, events)

    monkeypatch.setattr(workflow, "load_offline_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(workflow, "load_model_config", lambda *args, **kwargs: models)
    monkeypatch.setattr(
        LLMService,
        "remote",
        classmethod(lambda cls, *args: service),
    )
    monkeypatch.setattr(
        workflow,
        "run_preflight",
        lambda received: pytest.fail("preflight must follow remote readiness"),
    )

    with pytest.raises(RuntimeError, match="caption_embedding is not advertised"):
        workflow.run(
            workflow.parse_args(
                ["--stage", "all", "--inference-url", "https://gpu.test"]
            )
        )

    assert events == ["remote-ready", "remote-close"]
    assert service.closed is True


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


def test_custom_projection_uses_canonical_paths_without_btc_mapping(
    tmp_path: Path,
) -> None:
    """Custom extraction needs neither keyframe order nor organizer CSV data."""

    from scripts import build_retrieval_indexes as workflow

    image = tmp_path / "published" / "v1" / "images" / "000000000.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture-image")
    frames = pd.DataFrame(
        [
            {
                "frame_id": "v1_raw1fps_000000000",
                "video_id": "v1",
                "frame_idx": 12,
                "timestamp_ms": 500,
                "keyframe_order": None,
                "image_path": image.relative_to(tmp_path).as_posix(),
            }
        ]
    )

    projected = workflow.project_canonical_images(frames, tmp_path)

    assert projected.iloc[0]["frame_id"] == "v1_raw1fps_000000000"
    assert projected.iloc[0]["frame_idx"] == 12
    assert projected.iloc[0]["timestamp_ms"] == 500
    assert projected.iloc[0]["keyframe_order"] is None
    assert Path(projected.iloc[0]["image_path"]) == image.resolve()


def test_offline_model_config_rejects_mutable_revisions(tmp_path: Path) -> None:
    """Refuse branch aliases that cannot reproduce a published index."""

    from scripts import build_retrieval_indexes as workflow

    models = tmp_path / "models.yaml"
    models.write_text(
        """
visual_embedding:
  backend: siglip
  model_name: fake/visual
  revision: main
evidence_embedding:
  backend: bge_m3
  model_name: fake/text
  revision: 5617a9f61b028005a4858fdac845db406aefb181
""".strip()
    )

    with pytest.raises(ValueError, match="40-character hexadecimal"):
        workflow.load_model_config(models)


def test_offline_preflight_rejects_evidence_with_no_usable_corpus() -> None:
    """Stop before model loading when Context and transcript builders would fail."""

    from scripts import build_retrieval_indexes as workflow
    from hcmai.common.schemas import ProcessingStatus

    context = SimpleNamespace(frame_id="f1")
    data = SimpleNamespace(
        iter_frame_contexts=lambda: iter((context,)),
        get_frame_context_text=lambda frame_id: None,
    )
    with pytest.raises(ValueError, match="no usable context_text"):
        workflow._require_usable_context_ids(data)

    failed_segment = SimpleNamespace(
        status=ProcessingStatus.FAILED,
        text="recognizable words",
    )
    with pytest.raises(ValueError, match="no usable completed segments"):
        workflow._require_usable_completed_segments((failed_segment,))
