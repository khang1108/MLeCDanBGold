"""S3 transport tests for adaptive offline preprocessing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from hcmai.common.schemas.frame import FrameRecord
from hcmai.data.pipeline import DataService
from hcmai.data.preprocessing import PreprocessingConfig, S3PreprocessingConfig
from hcmai.data.preprocessing.prepare import (
    _checkpoint_path,
    _load_checkpoint,
    _write_parquet,
)
from hcmai.data.preprocessing.s3 import (
    S3VideoObject,
    list_video_objects,
    prepare_frame_store_from_s3,
    publish_frame_store,
    staged_video,
)


class FakePaginator:
    def __init__(self, client: "FakeS3") -> None:
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
                        2026, 8, 13, index + 1, tzinfo=timezone.utc
                    ),
                }
                for index, (key, value) in enumerate(self.client.objects.items())
                if key.startswith(Prefix)
            ]
        }


class FakeS3:
    """Small in-memory S3 surface used without boto3 or network access."""

    def __init__(
        self,
        objects: dict[str, bytes] | None = None,
        *,
        fail_upload_suffix: str | None = None,
    ) -> None:
        self.bucket = "hcmai-dataset"
        self.objects = dict(objects or {})
        self.fail_upload_suffix = fail_upload_suffix
        self.calls: list[tuple[str, str]] = []

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator(self)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        assert bucket == self.bucket
        self.calls.append(("download", key))
        Path(filename).write_bytes(self.objects[key])

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        assert bucket == self.bucket
        self.calls.append(("upload", key))
        if self.fail_upload_suffix and key.endswith(self.fail_upload_suffix):
            raise OSError("simulated artifact upload failure")
        self.objects[key] = Path(filename).read_bytes()

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
    ) -> None:
        assert Bucket == self.bucket
        assert ContentType == "application/json"
        self.calls.append(("put", Key))
        self.objects[Key] = Body

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, int]:
        assert Bucket == self.bucket
        self.calls.append(("head", Key))
        return {"ContentLength": len(self.objects[Key])}


def _config(root: Path, **updates: Any) -> PreprocessingConfig:
    values: dict[str, Any] = {
        "s3": {
            "bucket": "hcmai-dataset",
            "videos_prefix": "/raw/videos/",
            "artifacts_prefix": "/artifacts/frame-store/",
            "staging_root": root / "staging",
        },
        "output_root": root / "frame_store",
        "transnet_repo": root / "TransNetV2",
        "transnet_weights": root / "transnetv2-weights",
        "efficientgebd_repo": root / "EfficientGEBD",
        "efficientgebd_config": root / "efficientgebd.yaml",
        "efficientgebd_checkpoint": root / "efficientgebd.pth",
        "dino_revision": "test-revision",
    }
    values.update(updates)
    return PreprocessingConfig.model_validate(values)


def _frame_row(video_id: str) -> dict[str, object]:
    return FrameRecord(
        frame_id=f"{video_id}_frame_000000000",
        video_id=video_id,
        frame_idx=0,
        timestamp_ms=0,
        image_path=f"images/{video_id}.jpg",
        width=8,
        height=8,
        pts=0,
        time_base="1/10",
        selection_reasons=("coverage_anchor",),
    ).model_dump(mode="python")


def _bundle(root: Path) -> Path:
    root.mkdir()
    (root / "images").mkdir()
    (root / "images/L21_V001.jpg").write_bytes(b"jpeg")
    pd.DataFrame([_frame_row("L21_V001")]).to_parquet(
        root / "frames.parquet", index=False
    )
    (root / "manifest.json").write_text(
        json.dumps({"frame_count": 1, "video_count": 1}),
        encoding="utf-8",
    )
    return root


def test_s3_config_requires_exactly_one_source_and_normalizes_prefixes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    assert config.videos_root is None
    assert config.s3 is not None
    assert config.s3.videos_prefix == "raw/videos"
    assert config.s3.artifacts_prefix == "artifacts/frame-store"

    with pytest.raises(ValidationError, match="exactly one"):
        _config(tmp_path, videos_root=tmp_path / "videos")
    with pytest.raises(ValidationError, match="bucket-relative"):
        S3PreprocessingConfig.model_validate({
            "bucket": "hcmai-dataset",
            "videos_prefix": "s3://hcmai-dataset/videos",
        })


def test_s3_listing_and_staging_are_filtered_bounded_and_ephemeral(
    tmp_path: Path,
) -> None:
    client = FakeS3({
        "raw/videos/L21_V002.mp4": b"video-two",
        "raw/videos/L21_V001.mp4": b"video-one",
        "raw/videos/readme.txt": b"ignored",
        "other/L21_V003.mp4": b"ignored",
    })
    storage = _config(tmp_path).s3
    assert storage is not None

    sources = list_video_objects(client, storage, limit=1)

    assert [source.key for source in sources] == [
        "raw/videos/L21_V001.mp4"
    ]
    with staged_video(client, storage, sources[0]) as path:
        staged_path = path
        assert path.read_bytes() == b"video-one"
        assert path.stat().st_size == sources[0].size
    assert not staged_path.exists()


def test_s3_preprocessing_publishes_version_and_advances_latest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3({
        "raw/videos/L21_V001.mp4": b"first",
        "raw/videos/L21_V002.mp4": b"second",
    })
    config = _config(tmp_path)
    seen_versions: list[str] = []

    def fake_prepare(
        path: Path,
        active: PreprocessingConfig,
        *_args: object,
    ) -> pd.DataFrame:
        source_version = str(_args[-1])
        seen_versions.append(source_version)
        image = active.output_root / f"images/{path.stem}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(path.read_bytes())
        return pd.DataFrame([_frame_row(path.stem)])

    monkeypatch.setattr(
        "hcmai.data.preprocessing.s3._prepare_video", fake_prepare
    )

    output = prepare_frame_store_from_s3(
        config,
        shot_detector=object(),
        event_detector=object(),
        encoder=object(),
        client=client,
    )

    assert output == config.output_root / "frames.parquet"
    assert len(pd.read_parquet(output)) == 2
    assert len(seen_versions) == 2
    assert all(len(value) == 64 for value in seen_versions)
    latest_key = "artifacts/frame-store/latest.json"
    latest = json.loads(client.objects[latest_key])
    assert latest["frames_key"].endswith("/frames.parquet")
    assert latest["completion_key"].endswith("/_SUCCESS.json")
    assert client.calls[-2:] == [("put", latest_key), ("head", latest_key)]
    assert not tuple((tmp_path / "staging").iterdir())


def test_s3_checkpoint_rejects_changed_object_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = tmp_path / "L21_V001.mp4"
    source.write_bytes(b"video")
    image = config.output_root / "images/L21_V001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg")
    checkpoint = pd.DataFrame([_frame_row("L21_V001")]).assign(
        _config_hash="config-hash",
        _source_size=source.stat().st_size,
        _source_mtime_ns=source.stat().st_mtime_ns,
        _source_version="object-version-one",
    )
    _write_parquet(checkpoint, _checkpoint_path(config, "L21_V001"))

    assert _load_checkpoint(
        config,
        source,
        True,
        "config-hash",
        "object-version-one",
    ) is not None
    assert _load_checkpoint(
        config,
        source,
        True,
        "config-hash",
        "object-version-two",
    ) is None


def test_limited_s3_run_cannot_advance_full_corpus_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3({"raw/videos/L21_V001.mp4": b"video"})
    config = _config(tmp_path)

    def fake_prepare(
        path: Path,
        active: PreprocessingConfig,
        *_args: object,
    ) -> pd.DataFrame:
        image = active.output_root / f"images/{path.stem}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"jpeg")
        return pd.DataFrame([_frame_row(path.stem)])

    monkeypatch.setattr(
        "hcmai.data.preprocessing.s3._prepare_video", fake_prepare
    )

    output = prepare_frame_store_from_s3(
        config,
        shot_detector=object(),
        event_detector=object(),
        encoder=object(),
        client=client,
        limit=1,
    )

    assert output == tmp_path / "frame_store.limit-1/frames.parquet"
    assert "artifacts/frame-store/latest.json" not in client.objects
    assert (
        "artifacts/frame-store/limited/limit-1/latest.json" in client.objects
    )


def test_failed_bundle_upload_never_changes_latest(tmp_path: Path) -> None:
    client = FakeS3(fail_upload_suffix="manifest.json")
    latest_key = "artifacts/frame-store/latest.json"
    client.objects[latest_key] = b'{"bundle_id":"previous"}\n'
    storage = S3PreprocessingConfig.model_validate({
        "bucket": client.bucket,
        "artifacts_prefix": "artifacts/frame-store",
    })

    with pytest.raises(OSError, match="simulated artifact upload failure"):
        publish_frame_store(client, storage, _bundle(tmp_path / "bundle"))

    assert client.objects[latest_key] == b'{"bundle_id":"previous"}\n'
    assert not any(key.endswith("/_SUCCESS.json") for key in client.objects)


def test_data_service_dispatches_s3_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "preprocessing.yaml"
    values = config.model_dump(mode="json", exclude_none=True)
    import yaml

    config_path.write_text(
        yaml.safe_dump({"preprocessing": values}), encoding="utf-8"
    )
    expected = tmp_path / "dispatched.parquet"

    def fake_s3_prepare(
        active: PreprocessingConfig, *, resume: bool, limit: int | None,
    ) -> Path:
        assert active.s3 is not None
        assert resume is False
        assert limit == 3
        return expected

    monkeypatch.setattr(
        "hcmai.data.preprocessing.prepare_frame_store_from_s3",
        fake_s3_prepare,
    )

    assert DataService.prepare_adaptive(
        config_path, resume=False, limit=3
    ) == expected
