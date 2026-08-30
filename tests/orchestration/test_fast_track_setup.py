"""Verify startup composition for fast-track retrieval artifacts.

The tests mock index and encoder boundaries so online startup behavior is
checked without loading model weights or rebuilding offline artifacts.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hcmai.common.config import (
    AppConfig,
    EncoderConfig,
    FusionConfig,
    IndexConfig,
    SearchConfig,
)
from hcmai.corpus import CorpusFrameLoadError
from hcmai.common.schemas import RetrievalSource
from thundercompute.config import LLMServiceConfig
from hcmai.orchestration import setup
from hcmai.retrieval.retriever.pipeline import RetrievalService


class _LoadedService:
    """Small observable stand-in for a composed retrieval service."""

    def __init__(self, sources: tuple[RetrievalSource, ...]) -> None:
        self.active_sources = sources


class _FakeCorpus(SimpleNamespace):
    """Corpus-shaped fake retaining the public length protocol."""

    def __len__(self) -> int:
        return 1


def _metadata(
    *,
    model_name: str,
    model_revision: str | None,
    dimension: int,
    entity_kind: str,
    retrieval_source: str | None,
    dataset_version: str = "dataset-v1",
    schema_version: str = "dense-index-v2",
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_version=dataset_version,
        model_name=model_name,
        model_revision=model_revision,
        embedding_dim=dimension,
        entity_kind=entity_kind,
        retrieval_source=retrieval_source,
        schema_version=schema_version,
    )


def _models() -> LLMServiceConfig:
    return LLMServiceConfig(
        visual_embedding=EncoderConfig(
            model_name="visual/model",
            revision="visual-revision",
        ),
        caption_embedding=EncoderConfig(
            backend="bge_m3",
            model_name="legacy/text",
            revision="legacy-revision",
        ),
        evidence_embedding=EncoderConfig(
            backend="bge_m3",
            model_name="evidence/model",
            revision="evidence-revision",
        ),
    )


def _modern_settings(
    tmp_path: Path,
    *,
    required_sources: set[RetrievalSource] | None = None,
) -> AppConfig:
    visual = tmp_path / "visual"
    context = tmp_path / "context"
    asr = tmp_path / "asr-segments"
    visual.mkdir()
    context.mkdir()
    asr.mkdir()
    return AppConfig(
        index=IndexConfig(
            path=visual,
            context_path=context,
            asr_segment_path=asr,
        ),
        search=SearchConfig(
            fusion=FusionConfig(
                required_sources=required_sources or {RetrievalSource.VISUAL}
            ),
        ),
    )


def _install_modern_loaders(
    monkeypatch: pytest.MonkeyPatch,
    settings: AppConfig,
    *,
    context_metadata: SimpleNamespace | None = None,
    asr_metadata: SimpleNamespace | None = None,
) -> tuple[list[str], dict[str, Any]]:
    visual = SimpleNamespace(metadata=_metadata(
        model_name="visual/model",
        model_revision="visual-revision",
        dimension=768,
        entity_kind="frame",
        retrieval_source="visual",
    ))
    context = SimpleNamespace(metadata=context_metadata or _metadata(
        model_name="evidence/model",
        model_revision="evidence-revision",
        dimension=1024,
        entity_kind="frame",
        retrieval_source="context",
    ))
    asr = SimpleNamespace(metadata=asr_metadata or _metadata(
        model_name="evidence/model",
        model_revision="evidence-revision",
        dimension=1024,
        entity_kind="segment",
        retrieval_source="asr",
    ))
    dense = {
        settings.index.path: visual,
        settings.index.context_path: context,
    }
    monkeypatch.setattr(
        setup.RetrievalService,
        "load_index",
        staticmethod(lambda path, **_: dense[path]),
    )
    monkeypatch.setattr(
        setup.SegmentDenseIndex,
        "load",
        classmethod(lambda cls, path, **_: asr),
    )
    encoder_sources: list[str] = []

    def query_encoder(config, index, llm, source):
        encoder_sources.append(source)
        return object()

    monkeypatch.setattr(setup, "_query_encoder", query_encoder)
    captured: dict[str, Any] = {}

    def compose(**kwargs):
        captured.update(kwargs)
        sources = [RetrievalSource.VISUAL]
        if kwargs["context_index"] is not None:
            sources.append(RetrievalSource.CONTEXT)
        if kwargs["asr_segment_index"] is not None:
            sources.append(RetrievalSource.ASR)
        return _LoadedService(tuple(sources))

    monkeypatch.setattr(
        setup.RetrievalService,
        "from_fast_track_indexes",
        staticmethod(compose),
    )
    for name in (
        "HCMAI_CONTEXT_INDEX_PATH",
        "HCMAI_ASR_SEGMENT_INDEX_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    return encoder_sources, captured


def test_modern_profile_loads_visual_context_and_segment_asr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _modern_settings(tmp_path)
    encoder_sources, captured = _install_modern_loaders(monkeypatch, settings)
    corpus = object()
    messages: list[str] = []

    service = setup._load_retrieval(
        settings,
        _models(),
        settings.index.path,
        None,
        messages,
        corpus=cast(Any, corpus),
    )

    assert cast(_LoadedService, service).active_sources == (
        RetrievalSource.VISUAL,
        RetrievalSource.CONTEXT,
        RetrievalSource.ASR,
    )
    assert encoder_sources == ["visual", "text"]
    assert captured["corpus"] is corpus
    assert captured["max_projection_gap_ms"] == 5_000
    assert messages == []


def test_modern_index_paths_allow_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _modern_settings(tmp_path)
    override_context = tmp_path / "override-context"
    override_asr = tmp_path / "override-asr"
    override_context.mkdir()
    override_asr.mkdir()
    monkeypatch.setenv("HCMAI_CONTEXT_INDEX_PATH", str(override_context))
    monkeypatch.setenv("HCMAI_ASR_SEGMENT_INDEX_PATH", str(override_asr))

    visual = SimpleNamespace(metadata=_metadata(
        model_name="visual/model",
        model_revision="visual-revision",
        dimension=768,
        entity_kind="frame",
        retrieval_source="visual",
    ))
    evidence = _metadata(
        model_name="evidence/model",
        model_revision="evidence-revision",
        dimension=1024,
        entity_kind="frame",
        retrieval_source="context",
    )
    loaded_paths: list[Path] = []

    def load_dense(path, **_):
        loaded_paths.append(path)
        return visual if path == settings.index.path else SimpleNamespace(metadata=evidence)

    monkeypatch.setattr(setup.RetrievalService, "load_index", staticmethod(load_dense))
    monkeypatch.setattr(
        setup.SegmentDenseIndex,
        "load",
        classmethod(lambda cls, path, **_: loaded_paths.append(path) or SimpleNamespace(
            metadata=_metadata(
                model_name="evidence/model",
                model_revision="evidence-revision",
                dimension=1024,
                entity_kind="segment",
                retrieval_source="asr",
            )
        )),
    )
    monkeypatch.setattr(setup, "_query_encoder", lambda *args: object())
    monkeypatch.setattr(
        setup.RetrievalService,
        "from_fast_track_indexes",
        staticmethod(lambda **_: _LoadedService(tuple())),
    )

    setup._load_retrieval(
        settings,
        _models(),
        settings.index.path,
        None,
        [],
        corpus=cast(Any, object()),
    )

    assert loaded_paths == [settings.index.path, override_context, override_asr]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("schema_version", "dense-index-v1", "dense-index-v2"),
        ("entity_kind", "segment", "entity_kind"),
        ("retrieval_source", "caption", "retrieval_source"),
        ("dataset_version", "other-dataset", "dataset version"),
        ("model_name", "other/model", "model"),
        ("model_revision", "other-revision", "revision"),
    ],
)
def test_optional_incompatible_context_is_skipped_with_clear_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: str,
    expected: str,
) -> None:
    settings = _modern_settings(tmp_path)
    metadata = _metadata(
        model_name="evidence/model",
        model_revision="evidence-revision",
        dimension=1024,
        entity_kind="frame",
        retrieval_source="context",
    )
    setattr(metadata, field, value)
    _install_modern_loaders(monkeypatch, settings, context_metadata=metadata)
    messages: list[str] = []

    service = setup._load_retrieval(
        settings,
        _models(),
        settings.index.path,
        None,
        messages,
        corpus=cast(Any, object()),
    )

    assert cast(_LoadedService, service).active_sources == (
        RetrievalSource.VISUAL,
        RetrievalSource.ASR,
    )
    assert any(expected in message for message in messages)


def test_incompatible_asr_dimension_degrades_to_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _modern_settings(tmp_path)
    asr_metadata = _metadata(
        model_name="evidence/model",
        model_revision="evidence-revision",
        dimension=768,
        entity_kind="segment",
        retrieval_source="asr",
    )
    _install_modern_loaders(monkeypatch, settings, asr_metadata=asr_metadata)
    messages: list[str] = []

    service = setup._load_retrieval(
        settings,
        _models(),
        settings.index.path,
        None,
        messages,
        corpus=cast(Any, object()),
    )

    assert cast(_LoadedService, service).active_sources == (
        RetrievalSource.VISUAL,
        RetrievalSource.CONTEXT,
    )
    assert any("embedding dimension" in message for message in messages)


def test_missing_required_context_disables_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _modern_settings(
        tmp_path,
        required_sources={RetrievalSource.VISUAL, RetrievalSource.CONTEXT},
    )
    settings.index.context_path.rmdir()
    _install_modern_loaders(monkeypatch, settings)
    messages: list[str] = []

    service = setup._load_retrieval(
        settings,
        _models(),
        settings.index.path,
        None,
        messages,
        corpus=cast(Any, object()),
    )

    assert service is None
    assert any("CONTEXT index not available" in message for message in messages)


def test_corpus_loader_opens_configured_usable_artifacts_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames.parquet"
    frames.write_bytes(b"frames")
    captions = tmp_path / "captions.parquet"
    captions.write_bytes(b"captions")
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    (transcripts / "video.parquet").write_bytes(b"segments")
    settings = AppConfig.model_validate({
        "dataset": {
            "frames_path": frames,
            "root": tmp_path,
            "media_info_path": None,
            "enrichment": {
                "caption_path": captions,
                "ocr_path": None,
                "object_path": None,
                "context_path": None,
                "transcripts_path": transcripts,
            },
        }
    })
    calls: list[dict[str, Any]] = []
    canonical = _FakeCorpus()

    def open_corpus(*args, **kwargs):
        calls.append(kwargs)
        return canonical

    monkeypatch.setattr(setup.Corpus, "open", staticmethod(open_corpus))

    loaded = setup._load_corpus(
        settings,
        frames,
        tmp_path,
        [],
    )

    assert loaded is canonical
    assert calls == [
        {
            "evidence_paths": {RetrievalSource.CAPTION: captions},
            "dataset_root": tmp_path,
            "object_counts_path": None,
            "transcript_path": transcripts,
            "video_metadata_path": None,
        },
    ]


def test_invalid_corpus_open_keeps_startup_degraded_with_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames.parquet"
    frames.write_bytes(b"frames")
    transcripts = tmp_path / "transcripts.parquet"
    transcripts.write_bytes(b"segments")
    settings = AppConfig.model_validate({
        "dataset": {
            "frames_path": frames,
            "root": tmp_path,
            "enrichment": {
                "transcripts_path": transcripts,
            },
        }
    })
    def open_corpus(*args, **kwargs):
        raise ValueError("lineage mismatch")

    monkeypatch.setattr(setup.Corpus, "open", staticmethod(open_corpus))
    messages: list[str] = []

    loaded = setup._load_corpus(
        settings,
        frames,
        tmp_path,
        messages,
    )

    assert loaded is None
    assert any("Could not load metadata" in message for message in messages)


def test_missing_canonical_frames_fail_fast(tmp_path: Path) -> None:
    """Startup must not run search without its required frame authority."""

    settings = AppConfig.model_validate({"dataset": {"root": tmp_path}})

    with pytest.raises(FileNotFoundError, match="Metadata not available"):
        setup._load_corpus(
            settings,
            tmp_path / "missing-frames.parquet",
            tmp_path,
            [],
        )


def test_malformed_canonical_frames_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A present but unreadable frame artifact remains a startup error."""

    frames = tmp_path / "frames.parquet"
    frames.write_bytes(b"invalid parquet")
    settings = AppConfig.model_validate({"dataset": {"root": tmp_path}})

    def open_invalid_frames(*_args: object, **_kwargs: object) -> None:
        """Simulate Corpus rejecting a malformed required frame artifact."""

        raise CorpusFrameLoadError("invalid frame schema")

    monkeypatch.setattr(
        setup.Corpus,
        "open",
        staticmethod(open_invalid_frames),
    )

    with pytest.raises(CorpusFrameLoadError, match="invalid frame schema"):
        setup._load_corpus(settings, frames, tmp_path, [])


def test_removed_environment_profile_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup, "_load_app_config", lambda: AppConfig())
    monkeypatch.setattr(setup, "_load_model_config", _models)
    monkeypatch.setenv("HCMAI_RETRIEVAL_PROFILE", "legacy_specialists")

    with pytest.raises(ValueError, match="no longer supported"):
        setup.load_search_service([])


def test_public_startup_selects_fast_track_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default startup reaches the sole supported fast-track composer."""

    settings = _modern_settings(tmp_path)
    monkeypatch.setattr(setup, "_load_app_config", lambda: settings)
    monkeypatch.setattr(setup, "_load_model_config", _models)
    monkeypatch.setattr(setup, "_load_remote_llm", lambda *_: None)
    corpus = SimpleNamespace()

    def load_corpus(*args, **kwargs):
        return corpus

    monkeypatch.setattr(setup, "_load_corpus", load_corpus)
    visual = SimpleNamespace(metadata=_metadata(
        model_name="visual/model",
        model_revision="visual-revision",
        dimension=768,
        entity_kind="frame",
        retrieval_source="visual",
    ))
    loaded_index_paths: list[Path] = []

    def load_index(path, **_):
        loaded_index_paths.append(path)
        return visual

    monkeypatch.setattr(setup.RetrievalService, "load_index", staticmethod(load_index))
    monkeypatch.setattr(setup, "_query_encoder", lambda *args: object())
    def modern(*args, **kwargs):
        return _LoadedService((
            RetrievalSource.VISUAL,
            RetrievalSource.CONTEXT,
            RetrievalSource.ASR,
        ))

    monkeypatch.setattr(setup, "_load_fast_track_retrieval", modern)

    service = setup.load_search_service([])

    assert cast(_LoadedService, service.retrieval).active_sources == (
        RetrievalSource.VISUAL,
        RetrievalSource.CONTEXT,
        RetrievalSource.ASR,
    )
    assert loaded_index_paths == [settings.index.path]
