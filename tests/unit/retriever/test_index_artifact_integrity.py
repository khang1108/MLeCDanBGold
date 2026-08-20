"""Integrity and publication tests for persisted dense-index bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from hcmai.retrieval.retriever.artifacts import (
    fingerprint_files,
    publish_directory,
    sha256_file,
)
from hcmai.retrieval.retriever.dense.index import DenseIndex, IndexArtifactError
from hcmai.retrieval.retriever.models.metadata import IndexMetadata


@pytest.fixture
def tiny_dense_index() -> DenseIndex:
    """Return a deterministic frame-native dense index for persistence tests."""
    vectors = np.eye(3, dtype=np.float32)
    mapping = pd.DataFrame(
        {
            "frame_id": ["f0", "f1", "f2"],
            "video_id": ["v0", "v0", "v1"],
            "frame_idx": [10, 20, 30],
            "embedding_index": [0, 1, 2],
            "timestamp_ms": [100, 200, 300],
        }
    )
    return DenseIndex.build(
        vectors,
        mapping,
        dataset_version="test",
        model_name="test-model",
    )


def test_sha256_file_and_fingerprint_files_are_content_sensitive(tmp_path: Path) -> None:
    """File helper digests must change when any indexed input changes."""
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert sha256_file(first) == hashlib.sha256(b"first").hexdigest()
    fingerprint = fingerprint_files([second, first])
    assert fingerprint == fingerprint_files([first, second])

    second.write_bytes(b"changed")
    assert fingerprint_files([first, second]) != fingerprint


def test_publish_directory_replaces_destination_and_removes_backup(tmp_path: Path) -> None:
    """Publication atomically exposes staged contents and clears its backup."""
    destination = tmp_path / "published"
    staged = tmp_path / "staged"
    destination.mkdir()
    staged.mkdir()
    (destination / "old.txt").write_text("old")
    (staged / "new.txt").write_text("new")

    assert publish_directory(staged, destination) == destination.resolve()
    assert (destination / "new.txt").read_text() == "new"
    assert not (destination / "old.txt").exists()
    assert not destination.with_name("published.backup").exists()


def test_publish_directory_refuses_stale_backup(tmp_path: Path) -> None:
    """A leftover backup blocks publication rather than risking data loss."""
    destination = tmp_path / "published"
    staged = tmp_path / "staged"
    backup = tmp_path / "published.backup"
    staged.mkdir()
    backup.mkdir()

    with pytest.raises(RuntimeError, match="Stale publication backup exists"):
        publish_directory(staged, destination)

    assert staged.exists()
    assert backup.exists()


def test_publish_directory_restores_previous_destination_on_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed staged rename must restore the previously published bundle."""
    destination = tmp_path / "published"
    staged = tmp_path / "staged"
    destination.mkdir()
    staged.mkdir()
    (destination / "old.txt").write_text("old")
    (staged / "new.txt").write_text("new")
    resolved_staged = staged.resolve()
    original_replace = Path.replace

    def fail_staged_rename(path: Path, target: Path) -> Path:
        """Simulate an OS failure while replacing the published directory."""
        if path == resolved_staged:
            raise OSError("simulated staged rename failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staged_rename)

    with pytest.raises(OSError, match="simulated staged rename failure"):
        publish_directory(staged, destination)

    assert (destination / "old.txt").read_text() == "old"
    assert not destination.with_name("published.backup").exists()
    assert staged.exists()


def test_v2_dense_index_rejects_tampered_vectors(
    tmp_path: Path, tiny_dense_index: DenseIndex
) -> None:
    """A v2 bundle must reject a checksum mismatch before serving retrieval."""
    output = tiny_dense_index.save(tmp_path / "index")
    vectors = output / "vectors.npy"
    with vectors.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(IndexArtifactError, match="checksum"):
        DenseIndex.load(output)


def test_old_metadata_without_v2_fields_remains_readable(
    tmp_path: Path, tiny_dense_index: DenseIndex
) -> None:
    """V1 JSON must remain loadable for rollback-safe artifact migrations."""
    output = tiny_dense_index.save(tmp_path / "index")
    metadata_path = output / "metadata.json"
    metadata = IndexMetadata.from_dict(json.loads(metadata_path.read_text())).to_dict()
    for field in (
        "schema_version",
        "entity_kind",
        "retrieval_source",
        "model_revision",
        "source_fingerprint",
        "config_fingerprint",
        "checksums",
    ):
        metadata.pop(field, None)
    metadata["unexpected_future_field"] = "ignored"
    metadata_path.write_text(json.dumps(metadata))

    loaded = DenseIndex.load(output)
    assert loaded.metadata.schema_version == "dense-index-v1"
    assert loaded.metadata.checksums is None
