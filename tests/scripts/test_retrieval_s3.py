"""Small deterministic tests for the S3 retrieval-index transfer boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.retrieval_s3 import download_prefix, publish_retrieval_bundle


class FakeS3:
    """Thread-safe-enough in-memory S3 surface used by transfer tests."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.uploaded: dict[str, bytes] = {}

    def get_paginator(self, name: str) -> Any:
        """Return this fake as the only list-objects paginator."""

        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
        """List matching objects using the same shape as boto3."""

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
        """Write one fake object to the requested temporary path."""

        del bucket
        Path(filename).write_bytes(self.objects[key])

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        """Capture one uploaded local file."""

        del bucket
        self.uploaded[key] = Path(filename).read_bytes()

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: Any) -> None:
        """Capture a small pointer/completion JSON object."""

        del Bucket
        self.uploaded[Key] = bytes(Body)

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, int]:
        """Return the remote byte count used by upload verification."""

        del Bucket
        payload = self.uploaded.get(Key, self.objects.get(Key))
        if payload is None:
            raise KeyError(Key)
        return {"ContentLength": len(payload)}


def test_download_prefix_resumes_same_size_files_atomically(tmp_path: Path) -> None:
    """Retain complete files and replace missing files without ``.part`` debris."""

    client = FakeS3(
        {
            "data/input/a.txt": b"already complete",
            "data/input/nested/b.txt": b"download me",
        }
    )
    destination = tmp_path / "input"
    (destination / "a.txt").parent.mkdir(parents=True)
    (destination / "a.txt").write_bytes(b"already complete")

    stats = download_prefix(
        client,
        "bucket",
        "data/input",
        destination,
        workers=2,
    )

    assert stats.files == 2
    assert stats.downloaded == 1
    assert stats.skipped == 1
    assert stats.bytes_transferred == len(b"download me")
    assert (destination / "nested/b.txt").read_bytes() == b"download me"
    assert not list(destination.rglob("*.part"))


def test_download_prefix_dry_run_does_not_write_objects(tmp_path: Path) -> None:
    """Allow checking the remote input inventory before reserving disk space."""

    client = FakeS3({"data/input/a.txt": b"payload"})
    destination = tmp_path / "input"

    stats = download_prefix(
        client,
        "bucket",
        "data/input",
        destination,
        dry_run=True,
    )

    assert stats.files == 1
    assert stats.downloaded == 0
    assert not (destination / "a.txt").exists()


def test_publish_retrieval_bundle_uploads_report_before_latest(tmp_path: Path) -> None:
    """Never advertise ``latest`` until indexes, report, and success exist."""

    root = tmp_path / "indexes"
    for name in ("visual", "context", "asr_segments"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(
            json.dumps({"source": name}), encoding="utf-8"
        )
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
    completion = json.loads(client.uploaded[f"{version}/_SUCCESS.json"])
    assert completion["status"] == "passed"
    assert completion["file_count"] == 4


def test_publish_retrieval_bundle_rejects_unpassed_report(tmp_path: Path) -> None:
    """Keep incomplete local builds out of the remote immutable namespace."""

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


def test_s3_cli_downloads_builds_validates_then_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep S3 transfer outside model work and publish only after validation."""

    from scripts import build_retrieval_indexes as workflow

    events: list[str] = []
    projection = tmp_path / "projected.parquet"
    projection.write_bytes(b"projection")
    config = type(
        "Config",
        (),
        {"projected_frames_path": projection, "output_root": tmp_path / "indexes"},
    )()
    client = object()

    monkeypatch.setattr(
        workflow,
        "load_offline_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(workflow, "load_model_config", lambda *args: object())
    monkeypatch.setattr(
        workflow,
        "_load_s3_transport",
        lambda path: (events.append("open") or client, "bucket"),
    )
    monkeypatch.setattr(
        workflow,
        "_download_s3_inputs",
        lambda received_client, bucket, received_config, args: events.append("download"),
    )
    monkeypatch.setattr(
        workflow,
        "_close_s3_transport",
        lambda received_client: events.append("close"),
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
        lambda: events.append("release"),
    )
    monkeypatch.setattr(
        workflow,
        "create_text_encoder",
        lambda *args: events.append("text-encoder") or object(),
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
        lambda *args, **kwargs: events.append("validate"),
    )
    monkeypatch.setattr(
        workflow,
        "_publish_s3_bundle",
        lambda *args, **kwargs: events.append("publish"),
    )

    workflow.run(workflow.parse_args(["--s3", "--stage", "all"]))

    assert events == [
        "open",
        "download",
        "preflight",
        "visual",
        "release",
        "text-encoder",
        "context",
        "asr",
        "validate",
        "publish",
        "close",
    ]
