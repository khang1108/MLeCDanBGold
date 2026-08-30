"""Validate resumable S3 transfer for offline retrieval-index artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from offline.ingestion.s3 import (
    download_prefix,
    load_s3_config,
    publish_retrieval_bundle,
)


class FakeS3:
    """Provide the small boto3 surface used by transfer tests."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.uploaded: dict[str, bytes] = {}

    def get_paginator(self, name: str) -> Any:
        """Return this fake as the list-objects paginator."""

        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
        """List matching objects with boto3-compatible fields."""

        del Bucket
        return [
            {
                "Contents": [
                    {"Key": key, "Size": len(value)}
                    for key, value in sorted(self.objects.items())
                    if key.startswith(Prefix)
                ]
            }
        ]

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        """Write one fake remote object to its requested local path."""

        del bucket
        Path(filename).write_bytes(self.objects[key])

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        """Capture one uploaded file."""

        del bucket
        self.uploaded[key] = Path(filename).read_bytes()

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: Any) -> None:
        """Capture a small JSON marker or pointer."""

        del Bucket
        self.uploaded[Key] = bytes(Body)

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, int]:
        """Return the byte count used by post-upload verification."""

        del Bucket
        payload = self.uploaded.get(Key, self.objects.get(Key))
        if payload is None:
            raise KeyError(Key)
        return {"ContentLength": len(payload)}


def test_load_s3_config_accepts_transport_only_yaml_and_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate the active S3 config without loading legacy preparation stages."""

    config_path = tmp_path / "storage.s3.yaml"
    config_path.write_text(
        "s3:\n  bucket: default-bucket\n  region: ap-east-1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HCMAI_S3_BUCKET", "override-bucket")
    monkeypatch.setenv("HCMAI_S3_ENDPOINT_URL", "https://s3.example.test")

    config = load_s3_config(config_path)

    assert config.bucket == "override-bucket"
    assert config.region == "ap-east-1"
    assert config.endpoint_url == "https://s3.example.test"


def test_download_prefix_resumes_and_replaces_atomically(tmp_path: Path) -> None:
    """Keep complete files and download missing files without partial debris."""

    client = FakeS3(
        {
            "data/input/a.txt": b"already complete",
            "data/input/nested/b.txt": b"download me",
        }
    )
    destination = tmp_path / "input"
    destination.mkdir()
    (destination / "a.txt").write_bytes(b"already complete")

    stats = download_prefix(
        client,
        "bucket",
        "data/input",
        destination,
        workers=2,
    )

    assert (stats.files, stats.downloaded, stats.skipped) == (2, 1, 1)
    assert stats.bytes_transferred == len(b"download me")
    assert (destination / "nested/b.txt").read_bytes() == b"download me"
    assert not list(destination.rglob("*.part"))


def test_download_prefix_dry_run_does_not_write_objects(tmp_path: Path) -> None:
    """Inventory remote inputs without downloading their payloads."""

    destination = tmp_path / "input"
    stats = download_prefix(
        FakeS3({"data/input/a.txt": b"payload"}),
        "bucket",
        "data/input",
        destination,
        dry_run=True,
    )

    assert stats.files == 1
    assert stats.downloaded == 0
    assert not (destination / "a.txt").exists()


def test_publish_retrieval_bundle_advances_latest_after_success(tmp_path: Path) -> None:
    """Publish indexes and report before exposing the immutable version."""

    root = tmp_path / "indexes"
    for name in ("visual", "context", "asr_segments"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text("{}", encoding="utf-8")
    (root / "build_report.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    client = FakeS3({})

    publication = publish_retrieval_bundle(
        client,
        "bucket",
        root,
        "data/artifacts/indexes",
        workers=2,
    )

    version = publication.version_prefix
    assert f"{version}/build_report.json" in client.uploaded
    assert f"{version}/_SUCCESS.json" in client.uploaded
    latest = json.loads(client.uploaded[publication.latest_key])
    assert latest["version_prefix"] == version
    assert json.loads(client.uploaded[f"{version}/_SUCCESS.json"])["file_count"] == 4


def test_publish_retrieval_bundle_rejects_unpassed_report(tmp_path: Path) -> None:
    """Refuse to upload a bundle that has not passed local validation."""

    root = tmp_path / "indexes"
    for name in ("visual", "context", "asr_segments"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text("{}", encoding="utf-8")
    (root / "build_report.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="passed build report"):
        publish_retrieval_bundle(FakeS3({}), "bucket", root, "indexes")
