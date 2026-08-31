"""Acceptance tests for resumable S3-first preparation orchestration."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hcmai.retrieval.models import RetrievalSource
from offline.ingestion.corpus_build import (
    DefaultPreparationOperations,
    PreparationPaths,
    S3CorpusPreparationConfig,
    S3CorpusPreparationService,
)

SHA = "a" * 40


class _Paginator:
    def __init__(self, client: _FakeS3) -> None:
        self.client = client

    def paginate(self, *, Bucket: str, Prefix: str):
        assert Bucket == self.client.bucket
        yield {
            "Contents": [
                {
                    "Key": key,
                    "Size": len(value),
                    "ETag": f'"etag-{index}"',
                    "LastModified": datetime(
                        2026, 8, 13, index + 1, tzinfo=UTC
                    ),
                }
                for index, (key, value) in enumerate(self.client.objects.items())
                if key.startswith(Prefix)
            ]
        }


class _FakeS3:
    bucket = "hcmai-dataset"

    def __init__(self) -> None:
        self.objects = {
            "videos/L21_V001.mp4": b"newest-s3-one",
            "videos/L21_V002.mp4": b"newest-s3-two",
        }
        self.downloads: list[str] = []

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        assert bucket == self.bucket
        self.downloads.append(key)
        Path(filename).write_bytes(self.objects[key])


class _Operations:
    def __init__(self, paths: PreparationPaths) -> None:
        self.paths = paths
        self.events: list[str] = []

    def prepare_btc_frame_store(self) -> Path:
        self.events.append("btc_frame_store")
        self.paths.frame_store_root.mkdir(parents=True, exist_ok=True)
        self.paths.frames_path.write_bytes(b"frames")
        (self.paths.frame_store_root / "manifest.json").write_text(
            '{"frame_count":2}', encoding="utf-8"
        )
        return self.paths.frames_path

    def prepare_transcript(self, video: Path) -> Path:
        assert video.is_file()
        assert video.read_bytes().startswith(b"newest-s3-")
        self.events.append(f"transcript:{video.stem}")
        output = self.paths.transcripts_root / f"{video.stem}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"transcript")
        return output

    def materialize_asr(self) -> Path:
        self.events.append("asr")
        self.paths.asr_root.mkdir(parents=True, exist_ok=True)
        self.paths.asr_enrichment_path.write_bytes(b"asr")
        return self.paths.asr_enrichment_path

    def generate_caption(self) -> Path:
        self.events.append("caption")
        root = self.paths.caption_root
        root.mkdir(parents=True, exist_ok=True)
        output = root / "captions.parquet"
        output.write_bytes(b"caption")
        (root / "failures.json").write_text("[]", encoding="utf-8")
        (root / "frame_enrichment.parquet").write_bytes(b"caption-projection")
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        return output

    def generate_ocr(self) -> Path:
        self.events.append("ocr")
        root = self.paths.ocr_root
        root.mkdir(parents=True, exist_ok=True)
        output = root / "frames.parquet"
        output.write_bytes(b"ocr")
        (root / "regions.parquet").write_bytes(b"regions")
        (root / "failures.json").write_text("[]", encoding="utf-8")
        (root / "frame_enrichment.parquet").write_bytes(b"ocr-projection")
        (root / "ocr_report.json").write_text("{}", encoding="utf-8")
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        return output

    def detect_objects(self) -> Path:
        self.events.append("objects")
        root = self.paths.object_root
        root.mkdir(parents=True, exist_ok=True)
        output = root / "frames.parquet"
        output.write_bytes(b"objects")
        (root / "detections.parquet").write_bytes(b"detections")
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        return output

    def build_frame_context(self) -> Path:
        self.events.append("frame_context")
        root = self.paths.context_root
        root.mkdir(parents=True, exist_ok=True)
        output = root / "frame_context_v1.parquet"
        output.write_bytes(b"context")
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        return output

    def build_visual_index(self) -> Path:
        self.events.append("visual_index")
        self.paths.visual_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.visual_embeddings_path.write_bytes(b"embeddings")
        self.paths.visual_mapping_path.write_bytes(b"mapping")
        return self._index(self.paths.visual_index_root)

    def build_text_index(self, source: RetrievalSource) -> Path:
        self.events.append(f"{source.value}_index")
        root = self.paths.index_root(source)
        root.mkdir(parents=True, exist_ok=True)
        self.paths.text_embeddings_path(source).write_bytes(b"embeddings")
        self.paths.text_mapping_path(source).write_bytes(b"mapping")
        return self._index(root)

    @staticmethod
    def _index(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        output = root / "dense.index"
        output.write_bytes(b"index")
        return output


def _config(tmp_path: Path) -> S3CorpusPreparationConfig:
    work = (tmp_path / "run").resolve()
    model_names = {
        name: f"fixture/{name}"
        for name in (
            "caption",
            "ocr",
            "asr",
            "diarization",
            "visual_embedding",
            "text_embedding",
        )
    }
    return S3CorpusPreparationConfig.model_validate({
        "corpus_revision": "hcmai2026-s3-fixture-v1",
        "work_root": work,
        "execution": {"minimum_free_gib_after_cache": 0},
        "models": {
            name: {"model_name": model, "revision": SHA}
            for name, model in model_names.items()
        },
        "preprocessing": {
            "s3": {
                "bucket": "hcmai-dataset",
                "videos_prefix": "videos",
                "artifacts_prefix": "artifacts/production/fixture",
                "smoke_artifacts_prefix": "artifacts/smoke/fixture",
                "staging_root": work / "staging",
                "cache_root": work / "source-cache",
            },
        },
    })


def test_btc_competition_run_uses_detection_then_context_without_preprocessing(
    tmp_path: Path,
) -> None:
    """Route active BTC stages without touching the legacy video frame session."""

    values = _config(tmp_path).model_dump(mode="python")
    values["frame_store_source"] = "btc_keyframes"
    stages = values["stages"]
    assert isinstance(stages, dict)
    stages.update(
        {
            "frame_store": True,
            "caption": True,
            "ocr": True,
            "objects": True,
            "asr": False,
            "frame_context": True,
            "visual_index": False,
            "caption_index": False,
            "ocr_index": False,
            "asr_index": False,
        }
    )
    config = S3CorpusPreparationConfig.model_validate(values)
    client = _FakeS3()
    paths = PreparationPaths.from_config(config, None)

    class _CompetitionOperations(_Operations):
        def generate_caption(self) -> Path:
            self.events.append("caption")
            self.paths.caption_root.mkdir(parents=True, exist_ok=True)
            output = self.paths.caption_root / "captions.parquet"
            output.write_bytes(b"caption")
            (self.paths.caption_root / "failures.json").write_text("[]")
            (self.paths.caption_root / "frame_enrichment.parquet").write_bytes(
                b"caption-projection"
            )
            (self.paths.caption_root / "manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            return output

        def generate_ocr(self) -> Path:
            self.events.append("ocr")
            self.paths.ocr_root.mkdir(parents=True, exist_ok=True)
            output = self.paths.ocr_root / "frames.parquet"
            output.write_bytes(b"ocr")
            (self.paths.ocr_root / "regions.parquet").write_bytes(b"regions")
            (self.paths.ocr_root / "failures.json").write_text("[]")
            (self.paths.ocr_root / "frame_enrichment.parquet").write_bytes(
                b"ocr-projection"
            )
            (self.paths.ocr_root / "ocr_report.json").write_text("{}")
            (self.paths.ocr_root / "manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            return output

        def detect_objects(self) -> Path:
            self.events.append("objects")
            self.paths.object_root.mkdir(parents=True, exist_ok=True)
            output = self.paths.object_root / "frames.parquet"
            output.write_bytes(b"objects")
            (self.paths.object_root / "detections.parquet").write_bytes(
                b"detections"
            )
            (self.paths.object_root / "manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            return output

        def build_frame_context(self) -> Path:
            self.events.append("frame_context")
            self.paths.context_root.mkdir(parents=True, exist_ok=True)
            output = self.paths.context_root / "frame_context_v1.parquet"
            output.write_bytes(b"context")
            (self.paths.context_root / "manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            return output

    operations = _CompetitionOperations(paths)
    result = S3CorpusPreparationService(
        config,
        client=client,
        operations=operations,
        paths=paths,
    ).run()

    assert operations.events == [
        "btc_frame_store",
        "caption",
        "ocr",
        "objects",
        "frame_context",
    ]
    assert result.completed_stages == (
        "frame_store",
        "caption",
        "ocr",
        "objects",
        "frame_context",
    )
    assert not any(event.endswith("index") for event in operations.events)


def test_default_operations_use_public_object_and_context_services_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep object detection and derived context behind public boundaries."""

    from offline.enrichment.context.config import FrameContextConfig
    from offline.enrichment.object_detection import ObjectDetectionConfig
    from offline.enrichment.pipeline import EnrichmentService

    paths = SimpleNamespace(
        frames_path=tmp_path / "frames.parquet",
        caption_root=tmp_path / "captions",
        ocr_root=tmp_path / "ocr",
        object_root=tmp_path / "objects",
        context_root=tmp_path / "context",
    )
    operations = object.__new__(DefaultPreparationOperations)
    operations.paths = paths
    operations.enrichment_job = SimpleNamespace(
        objects=ObjectDetectionConfig(),
        data_root=tmp_path,
        frame_store_id="btc-fixture-v1",
        context=FrameContextConfig(),
    )
    calls: list[str] = []

    def detect_objects(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append("objects")
        return {}

    def build_context(*args: object, **kwargs: object) -> Path:
        calls.append("frame_context")
        return paths.context_root / "frame_context_v1.parquet"

    monkeypatch.setattr(EnrichmentService, "detect_objects", detect_objects)
    monkeypatch.setattr(EnrichmentService, "build_frame_context", build_context)

    operations.detect_objects()
    operations.build_frame_context()

    assert calls == ["objects", "frame_context"]


def test_default_operations_preserve_configured_ocr_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Override runtime placement and pins without discarding OCR policy."""

    from offline.enrichment.ocr.config import OCRConfig
    from offline.enrichment.pipeline import EnrichmentService

    configured = OCRConfig(
        enabled=False,
        backend="configured-backend",
        checkpoint="configured/model",
        revision="configured-revision",
        device="configured-device",
        dtype="float32",
        batch_size=7,
        image_size=512,
        enrichment_version="configured-enrichment-v2",
        dataset_version="configured-dataset",
        min_region_confidence=0.45,
        min_context_quality=0.8,
        artifact_version="configured-ocr-v2",
    )
    expected = replace(
        configured,
        checkpoint="pinned/model",
        revision=SHA,
        dataset_version="corpus-v2",
    )
    operations = object.__new__(DefaultPreparationOperations)
    operations.config = SimpleNamespace(
        models=SimpleNamespace(
            ocr=SimpleNamespace(model_name="pinned/model", revision=SHA)
        ),
        corpus_revision="corpus-v2",
        frame_store_source="btc_keyframes",
    )
    operations.enrichment_job = SimpleNamespace(
        ocr=configured,
        frame_store_id="btc-v2",
    )
    operations.paths = SimpleNamespace(
        frames_path=tmp_path / "frames.parquet",
        frame_store_root=tmp_path,
        ocr_root=tmp_path / "ocr",
    )
    operations._remote_pool = lambda capability: None
    captured: dict[str, object] = {}
    adapter = object()

    monkeypatch.setattr(
        EnrichmentService,
        "create_ocr_adapter",
        lambda config: adapter,
    )

    def capture_generation(
        frames_path: Path,
        output_dir: Path,
        config: OCRConfig,
        adapter: object,
        **kwargs: object,
    ) -> dict[str, object]:
        captured.update(
            config=config,
            adapter=adapter,
            frame_store_id=kwargs["frame_store_id"],
        )
        return {}

    monkeypatch.setattr(EnrichmentService, "generate_ocr", capture_generation)

    operations.generate_ocr()

    assert captured == {
        "config": expected,
        "adapter": adapter,
        "frame_store_id": "btc-v2",
    }


def test_default_context_stage_identity_depends_on_ocr_policy(
    tmp_path: Path,
) -> None:
    """Changing OCR derivation policy must also invalidate FrameContext."""

    from offline.enrichment.caption.config import CaptionConfig
    from offline.enrichment.context.config import FrameContextConfig
    from offline.enrichment.ocr.config import OCRConfig
    from offline.enrichment.object_detection import ObjectDetectionConfig

    operations = object.__new__(DefaultPreparationOperations)
    operations.config = SimpleNamespace(
        models=SimpleNamespace(
            ocr=SimpleNamespace(model_name="pinned/model", revision=SHA)
        ),
        corpus_revision="corpus-v2",
        frame_store_source="btc_keyframes",
    )
    operations.caption_job = SimpleNamespace(
        caption=CaptionConfig(
            model_checkpoint="caption/model",
            revision=SHA,
            prompt="<CAPTION>",
            decoding={},
            device="cpu",
            precision="fp32",
            dtype="float32",
            image_size=16,
            batch_size=2,
            enrichment_version="caption-v1",
            write_interval=2,
            dataset_version="corpus-v2",
        )
    )
    operations.paths = SimpleNamespace(
        frames_path=tmp_path / "frames.parquet",
        caption_root=tmp_path / "caption",
        ocr_root=tmp_path / "ocr",
        object_root=tmp_path / "objects",
        context_root=tmp_path / "context",
    )
    operations.enrichment_job = SimpleNamespace(
        frame_store_id="btc-v2",
        data_root=tmp_path,
        caption=operations.caption_job.caption,
        ocr=OCRConfig(
            checkpoint="configured/model",
            revision="configured-revision",
            min_region_confidence=0.0,
        ),
        objects=ObjectDetectionConfig(),
        context=FrameContextConfig(),
    )
    before_ocr = operations.stage_dependency_identity("ocr")
    before_context = operations.stage_dependency_identity("frame_context")
    operations.enrichment_job.ocr = replace(
        operations.enrichment_job.ocr,
        min_region_confidence=0.5,
    )

    assert operations.stage_dependency_identity("ocr") != before_ocr
    assert (
        operations.stage_dependency_identity("frame_context")
        != before_context
    )


def test_stage_resume_uses_policy_fingerprints_and_manifest_identity(
    tmp_path: Path,
) -> None:
    """Invalidate only policy-dependent stages and reject stale manifests."""

    values = _config(tmp_path).model_dump(mode="python")
    values["frame_store_source"] = "btc_keyframes"
    config = S3CorpusPreparationConfig.model_validate(values)
    paths = PreparationPaths.from_config(config, None)

    class _PolicyOperations(_Operations):
        def __init__(self, active_paths: PreparationPaths) -> None:
            super().__init__(active_paths)
            self.ocr_min_confidence = 0.0

        def stage_dependency_identity(self, stage: str) -> dict[str, object]:
            manifests: dict[str, dict[str, object]] = {
                "caption": {"artifact_version": "caption-v1"},
                "ocr": {"artifact_version": "ocr-v1"},
                "objects": {"artifact_version": "object-v1"},
                "frame_context": {
                    "context_version": "frame-context-v1",
                    "caption_version": "caption-v1",
                    "ocr_version": "ocr-v1",
                    "object_version": "object-v1",
                },
            }
            dependency: dict[str, object] = {"stage": stage}
            if stage in {"ocr", "frame_context"}:
                dependency["ocr_min_confidence"] = self.ocr_min_confidence
            return {
                "dependencies": dependency,
                "manifest": manifests.get(stage, {}),
            }

    operations = _PolicyOperations(paths)
    service = S3CorpusPreparationService(
        config,
        client=_FakeS3(),
        operations=operations,
        paths=paths,
    )

    bundles = {
        "caption": (
            paths.caption_root,
            ("captions.parquet", "failures.json", "frame_enrichment.parquet"),
            {"artifact_version": "caption-v1"},
        ),
        "ocr": (
            paths.ocr_root,
            (
                "frames.parquet",
                "regions.parquet",
                "failures.json",
                "frame_enrichment.parquet",
                "ocr_report.json",
            ),
            {"artifact_version": "ocr-v1"},
        ),
        "objects": (
            paths.object_root,
            ("frames.parquet", "detections.parquet"),
            {"artifact_version": "object-v1"},
        ),
        "frame_context": (
            paths.context_root,
            ("frame_context_v1.parquet",),
            {
                "context_version": "frame-context-v1",
                "caption_version": "caption-v1",
                "ocr_version": "ocr-v1",
                "object_version": "object-v1",
            },
        ),
    }
    for stage, (root, filenames, manifest) in bundles.items():
        root.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            (root / filename).write_bytes(filename.encode())
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        service._complete_stage(stage, "run-v1")

    assert not service._pending("caption", "run-v1", service._stage_outputs("caption"), [])
    assert not service._pending("ocr", "run-v1", service._stage_outputs("ocr"), [])
    assert not service._pending(
        "frame_context",
        "run-v1",
        service._stage_outputs("frame_context"),
        [],
    )

    operations.ocr_min_confidence = 0.5

    assert not service._pending("caption", "run-v1", service._stage_outputs("caption"), [])
    assert service._pending("ocr", "run-v1", service._stage_outputs("ocr"), [])
    assert service._pending(
        "frame_context",
        "run-v1",
        service._stage_outputs("frame_context"),
        [],
    )

    operations.ocr_min_confidence = 0.0
    (paths.caption_root / "manifest.json").write_text(
        '{"artifact_version":"caption-v2"}', encoding="utf-8"
    )
    assert service._pending(
        "caption", "run-v1", service._stage_outputs("caption"), []
    )


@pytest.mark.parametrize(
    ("stage", "missing_name"),
    [("ocr", "regions.parquet"), ("objects", "detections.parquet")],
)
def test_stage_resume_requires_authoritative_sibling_artifacts(
    tmp_path: Path, stage: str, missing_name: str
) -> None:
    """Rerun a specialist stage when its structured sibling disappears."""

    values = _config(tmp_path).model_dump(mode="python")
    values["frame_store_source"] = "btc_keyframes"
    config = S3CorpusPreparationConfig.model_validate(values)
    paths = PreparationPaths.from_config(config, None)
    service = S3CorpusPreparationService(
        config,
        client=_FakeS3(),
        operations=_Operations(paths),
        paths=paths,
    )
    outputs = service._stage_outputs(stage)
    for output in outputs:
        if output.name == "manifest.json":
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}", encoding="utf-8")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(output.name.encode())
    sibling = outputs[0].parent / missing_name
    if not sibling.exists():
        sibling.write_bytes(missing_name.encode())
    service._complete_stage(stage, "run-v1")

    sibling.unlink()

    assert service._pending(stage, "run-v1", outputs, [])


def test_two_video_run_resumes_every_stage_without_legacy_local_reads(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeS3()
    paths = PreparationPaths.from_config(config, None)
    operations = _Operations(paths)
    service = S3CorpusPreparationService(
        config,
        client=client,
        operations=operations,
        paths=paths,
    )

    first = service.run()
    events_after_first = tuple(operations.events)
    second = service.run()

    assert sorted(client.downloads) == [
        "videos/L21_V001.mp4",
        "videos/L21_V002.mp4",
    ]
    assert events_after_first == (
        "btc_frame_store",
        "transcript:L21_V001",
        "transcript:L21_V002",
        "caption",
        "ocr",
        "asr",
        "visual_index",
        "caption_index",
        "ocr_index",
        "asr_index",
    )
    assert tuple(operations.events) == events_after_first
    assert first.completed_stages == (
        "frame_store",
        "caption",
        "ocr",
        "asr",
        "visual_index",
        "caption_index",
        "ocr_index",
        "asr_index",
    )
    assert second.completed_stages == ()
    assert second.skipped_stages == first.completed_stages
    manifest = json.loads(first.inventory_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == first.run_id == second.run_id
    assert [item["key"] for item in manifest["source"]["objects"]] == [
        "videos/L21_V001.mp4",
        "videos/L21_V002.mp4",
    ]
    assert first.artifacts_root.is_relative_to(config.work_root)


def test_cache_only_records_inventory_without_model_work(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.execution.minimum_free_gib_after_cache = 0
    client = _FakeS3()
    paths = PreparationPaths.from_config(config, None)
    operations = _Operations(paths)
    result = S3CorpusPreparationService(
        config,
        client=client,
        operations=operations,
        paths=paths,
    ).cache_sources()

    assert result.source_count == 2
    assert result.downloaded_count == 2
    assert result.reused_count == 0
    assert result.total_bytes == sum(map(len, client.objects.values()))
    assert operations.events == []
    assert result.inventory_path.is_file()


def test_cached_run_can_overlap_frame_and_asr_lanes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.execution.overlap_frame_asr = True
    client = _FakeS3()
    paths = PreparationPaths.from_config(config, None)
    operations = _Operations(paths)

    result = S3CorpusPreparationService(
        config,
        client=client,
        operations=operations,
        paths=paths,
    ).run()

    assert operations.events[:3] == [
        "btc_frame_store",
        "transcript:L21_V001",
        "transcript:L21_V002",
    ]
    assert result.source_count == 2
    assert paths.frames_path.is_file()
    assert paths.asr_enrichment_path.is_file()


def test_changed_s3_inventory_cannot_reuse_an_existing_run(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeS3()
    paths = PreparationPaths.from_config(config, None)
    operations = _Operations(paths)
    service = S3CorpusPreparationService(
        config,
        client=client,
        operations=operations,
        paths=paths,
    )
    service.run()
    client.objects["videos/L21_V003.mp4"] = b"newest-s3-three"

    with pytest.raises(RuntimeError, match="inventory.*changed"):
        service.run()

    assert len(client.downloads) == 2
