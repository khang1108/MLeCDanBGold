"""Acceptance tests for resumable S3-first preparation orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from hcmai.common.schemas import RetrievalSource
from hcmai.data.corpus_build import (
    PreparationCacheRun,
    PreparationPaths,
    PreparationRun,
    S3CorpusPreparationConfig,
    S3CorpusPreparationService,
)
from scripts import prepare_s3_corpus as cli

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
        self.staged_paths: list[Path] = []

    def prepare_frame(self, video: Path, source: Any) -> str:
        assert video.is_file()
        assert video.read_bytes().startswith(b"newest-s3-")
        self.staged_paths.append(video)
        self.events.append(f"frame:{source.video_id}")
        return source.video_id

    def finalize_frames(self, prepared, sources) -> Path:
        assert list(prepared) == [source.video_id for source in sources]
        self.events.append("finalize_frames")
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
        return self._enrichment(self.paths.caption_root)

    def generate_ocr(self) -> Path:
        self.events.append("ocr")
        return self._enrichment(self.paths.ocr_root)

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
    def _enrichment(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        output = root / "frame_enrichment.parquet"
        output.write_bytes(b"enrichment")
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        return output

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
            "dino",
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
            "output_root": work / "artifacts/frame_store",
            "transnet_repo": work / "models/transnet",
            "transnet_weights": work / "models/transnet-weights",
            "efficientgebd_repo": work / "models/gebd",
            "efficientgebd_config": work / "models/gebd.yaml",
            "efficientgebd_checkpoint": work / "models/gebd.pth",
            "dino_model": model_names["dino"],
            "dino_revision": SHA,
        },
    })


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
    )

    first = service.run()
    events_after_first = tuple(operations.events)
    second = service.run()

    assert config.preprocessing.videos_root is None
    assert sorted(client.downloads) == [
        "videos/L21_V001.mp4",
        "videos/L21_V002.mp4",
    ]
    assert events_after_first == (
        "frame:L21_V001",
        "transcript:L21_V001",
        "frame:L21_V002",
        "transcript:L21_V002",
        "finalize_frames",
        "caption",
        "ocr",
        "asr",
        "visual_index",
        "caption_index",
        "ocr_index",
        "asr_index",
    )
    assert tuple(operations.events) == events_after_first
    assert all(path.exists() for path in operations.staged_paths)
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


def test_cli_is_a_thin_service_boundary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config = object()
    expected = PreparationRun(
        run_id="b" * 64,
        inventory_path=tmp_path / "run.json",
        artifacts_root=tmp_path / "artifacts.limit-1",
        source_count=2,
        completed_stages=("frame_store",),
        skipped_stages=("asr",),
    )

    class _Config:
        @staticmethod
        def from_yaml(path: Path):
            assert path == tmp_path / "preparation.yaml"
            return config

    class _Service:
        def __init__(self, active, **options) -> None:
            assert active is config
            assert options["resume"] is False
            assert options["limit"] == 1

        @staticmethod
        def run() -> PreparationRun:
            return expected

    monkeypatch.setattr(cli, "S3CorpusPreparationConfig", _Config)
    monkeypatch.setattr(cli, "S3CorpusPreparationService", _Service)

    result = cli.main([
        "--config",
        str(tmp_path / "preparation.yaml"),
        "--limit",
        "1",
        "--no-resume",
    ])

    assert result == 0
    output = capsys.readouterr().out
    assert f"Run ID: {expected.run_id}" in output
    assert "S3 videos: 2" in output
    assert "Status: PASSED" in output


def test_cache_only_records_inventory_without_model_work(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.execution.minimum_free_gib_after_cache = 0
    client = _FakeS3()
    operations = _Operations(PreparationPaths.from_config(config, None))
    result = S3CorpusPreparationService(
        config,
        client=client,
        operations=operations,
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
    ).run()

    assert operations.events[:2] == [
        "frame:L21_V001",
        "transcript:L21_V001",
    ]
    assert {
        "frame:L21_V002",
        "transcript:L21_V002",
    }.issubset(operations.events)
    assert result.source_count == 2
    assert paths.frames_path.is_file()
    assert paths.asr_enrichment_path.is_file()


def test_cli_cache_only_uses_cache_boundary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config = object()
    expected = PreparationCacheRun(
        run_id="c" * 64,
        inventory_path=tmp_path / "run.json",
        cache_root=tmp_path / "source-cache",
        source_count=2,
        downloaded_count=2,
        reused_count=0,
        total_bytes=30,
        duration_seconds=1.5,
    )

    class _Config:
        @staticmethod
        def from_yaml(path: Path):
            return config

    class _Service:
        def __init__(self, active, **options) -> None:
            assert active is config

        @staticmethod
        def cache_sources() -> PreparationCacheRun:
            return expected

    monkeypatch.setattr(cli, "S3CorpusPreparationConfig", _Config)
    monkeypatch.setattr(cli, "S3CorpusPreparationService", _Service)

    result = cli.main([
        "--config",
        str(tmp_path / "config.yaml"),
        "--cache-only",
    ])

    assert result == 0
    output = capsys.readouterr().out
    assert "Cache downloaded: 2" in output
    assert "Cache reused: 0" in output
    assert "Status: CACHED" in output


def test_changed_s3_inventory_cannot_reuse_an_existing_run(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeS3()
    operations = _Operations(PreparationPaths.from_config(config, None))
    service = S3CorpusPreparationService(
        config,
        client=client,
        operations=operations,
    )
    service.run()
    client.objects["videos/L21_V003.mp4"] = b"newest-s3-three"

    with pytest.raises(RuntimeError, match="inventory.*changed"):
        service.run()

    assert len(client.downloads) == 2
