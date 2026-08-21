from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcmai.common.schemas import RetrievalSource
from hcmai.data.corpus_build import (
    CommittedGroup,
    GroupPreparationService,
    GroupSourceInventory,
    S3CorpusPreparationConfig,
    S3GroupIndexReducer,
    verify_local_group,
)
from hcmai.data.corpus_build.audio import S3AudioReferenceProvider

SHA = "a" * 40


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        assert bucket == "hcmai-dataset"
        self.objects[key] = Path(filename).read_bytes()

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_):
        assert Bucket == "hcmai-dataset"
        self.objects[Key] = bytes(Body)

    def head_object(self, *, Bucket: str, Key: str):
        assert Bucket == "hcmai-dataset"
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, *, Bucket: str, Key: str):
        assert Bucket == "hcmai-dataset"
        return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        assert bucket == "hcmai-dataset"
        Path(filename).write_bytes(self.objects[key])

    def delete_object(self, *, Bucket: str, Key: str):
        assert Bucket == "hcmai-dataset"
        self.deleted.append(Key)
        self.objects.pop(Key, None)

    def generate_presigned_url(self, *_args, **_kwargs):
        return "https://s3.test/audio.flac?signature=test"


def _config(tmp_path: Path) -> S3CorpusPreparationConfig:
    work = (tmp_path / "work").resolve()
    names = {
        "dino": "dino",
        "caption": "caption",
        "ocr": "ocr",
        "asr": "asr",
        "diarization": "diarization",
        "visual_embedding": "visual",
        "text_embedding": "text",
    }
    return S3CorpusPreparationConfig.model_validate({
        "corpus_revision": "fixture-v1",
        "work_root": work,
        "models": {
            name: {"model_name": model, "revision": SHA}
            for name, model in names.items()
        },
        "preprocessing": {
            "s3": {
                "bucket": "hcmai-dataset",
                "videos_prefix": "data",
                "artifacts_prefix": "artifacts/production/v1",
                "smoke_artifacts_prefix": "artifacts/smoke/v1",
                "staging_root": work / "staging",
            },
            "output_root": work / "unused-frame-store",
            "transnet_repo": work / "models/transnet",
            "transnet_weights": work / "models/transnet-weights",
            "efficientgebd_repo": work / "models/gebd",
            "efficientgebd_config": work / "models/gebd.yaml",
            "efficientgebd_checkpoint": work / "models/gebd.pth",
            "dino_model": "dino",
            "dino_revision": SHA,
        },
    })


def _inventory(size: int) -> GroupSourceInventory:
    return GroupSourceInventory.model_validate({
        "group_id": "L21_a",
        "bucket": "hcmai-dataset",
        "prefix": "data/L21_a/videos",
        "objects": [{
            "key": "data/L21_a/videos/L21_V001.mp4",
            "size": size,
            "etag": "etag-1",
            "last_modified_ns": 1,
        }],
    })


class FakeOperations:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.events: list[str] = []

    def prepare_frame(self, video, source):
        self.events.append(f"frame:{source.video_id}")
        return source.video_id

    def finalize_frames(self, prepared, sources):
        root = self.paths.frame_store_root
        root.mkdir(parents=True, exist_ok=True)
        rows = pd.DataFrame([{
            "frame_id": "L21_V001_frame_000000000",
            "video_id": "L21_V001",
            "frame_idx": 0,
            "timestamp_ms": 0,
            "image_path": "images/L21/L21_V001/000000000.jpg",
        }])
        rows.to_parquet(self.paths.frames_path, index=False)
        (root / "manifest.json").write_text("{}")

    def prepare_transcript(self, video):
        self.events.append(f"asr:{video.stem}")
        self.paths.transcripts_root.mkdir(parents=True, exist_ok=True)
        (self.paths.transcripts_root / "segments.json").write_text("[]")

    def generate_caption(self):
        return self._enrichment(RetrievalSource.CAPTION)

    def generate_ocr(self):
        return self._enrichment(RetrievalSource.OCR)

    def materialize_asr(self):
        return self._enrichment(RetrievalSource.ASR)

    def _enrichment(self, source):
        root = self.paths.enrichment_path(source).parent
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "frame_id": "L21_V001_frame_000000000",
            source.value: f"{source.value} text",
        }]).to_parquet(self.paths.enrichment_path(source), index=False)
        if source is not RetrievalSource.ASR:
            (root / "manifest.json").write_text("{}")
        return self.paths.enrichment_path(source)

    def build_visual_artifacts(self):
        self.paths.visual_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.paths.visual_embeddings_path, np.array([[1, 0]], dtype=np.float32))
        self._mapping().to_parquet(self.paths.visual_mapping_path, index=False)
        return self.paths.visual_embeddings_path, self.paths.visual_mapping_path

    def build_text_embeddings(self, source):
        root = self.paths.index_root(source)
        root.mkdir(parents=True, exist_ok=True)
        np.save(self.paths.text_embeddings_path(source), np.array([[0, 1]], dtype=np.float32))
        self._mapping().to_parquet(self.paths.text_mapping_path(source), index=False)
        return self.paths.text_embeddings_path(source), self.paths.text_mapping_path(source)

    @staticmethod
    def _mapping():
        return pd.DataFrame([{
            "embedding_index": 0,
            "frame_id": "L21_V001_frame_000000000",
            "video_id": "L21_V001",
            "frame_idx": 0,
            "timestamp_ms": 0,
        }])


def test_group_input_verification_rejects_extra_video(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    (root / "L21_V001.mp4").write_bytes(b"one")
    inventory = _inventory(3)
    assert verify_local_group(root, inventory)["L21_V001"].is_file()
    (root / "L21_V002.mp4").write_bytes(b"two")
    with pytest.raises(ValueError, match="differs"):
        verify_local_group(root, inventory)


def test_audio_reference_is_reused_and_cleanup_is_exact(tmp_path: Path) -> None:
    client = FakeS3()
    video = tmp_path / "L21_V001.mp4"
    video.write_bytes(b"video")
    extracted: list[Path] = []

    def fake_extract(_video: Path, output: Path, _sample_rate: int) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"lossless-audio")
        extracted.append(output)

    provider = S3AudioReferenceProvider(
        client,
        bucket="hcmai-dataset",
        prefix="artifacts/temporary/run-1",
        work_root=tmp_path / "work",
        extractor=fake_extract,
    )
    first = provider.reference(video, "L21_V001", 16_000)
    second = provider.reference(video, "L21_V001", 16_000)

    assert first.audio_sha256 == second.audio_sha256
    assert len(extracted) == 1
    uploaded = next(iter(client.objects))
    assert uploaded.startswith("artifacts/temporary/run-1/temporary-audio/")

    provider.cleanup()
    assert client.deleted == [uploaded]
    assert uploaded not in client.objects


def test_group_run_commits_then_reducer_builds_global_index(tmp_path: Path) -> None:
    config, client = _config(tmp_path), FakeS3()
    videos = tmp_path / "videos"
    videos.mkdir()
    source = videos / "L21_V001.mp4"
    source.write_bytes(b"video")
    inventory = _inventory(source.stat().st_size)
    from hcmai.data.corpus_build.pipeline import PreparationPaths

    paths = PreparationPaths.for_group(config, "L21_a")
    operations = FakeOperations(paths)
    service = GroupPreparationService(
        config, videos, inventory, client=client, operations=operations
    )

    first = service.run()
    assert first.publication is not None
    commit_key = first.publication.latest_key
    assert commit_key.endswith("/COMMITTED.json")
    assert commit_key in client.objects

    second = service.run()
    assert "frame_store" in second.skipped_stages
    assert operations.events == ["frame:L21_V001", "asr:L21_V001"]

    reducer = S3GroupIndexReducer(
        client, bucket="hcmai-dataset", work_root=tmp_path / "reduce"
    )
    index_path = reducer.reduce(
        [CommittedGroup("L21_a", first.run_id, first.publication.version_prefix)],
        source=RetrievalSource.VISUAL,
        output_dir=tmp_path / "global-index",
        dataset_version=config.corpus_revision,
        model_name="visual",
    )
    assert index_path.is_file()
