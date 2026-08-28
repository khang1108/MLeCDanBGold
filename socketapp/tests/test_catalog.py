"""Tests for safe local video catalog construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socketapp.catalog import CatalogError, VideoCatalog


def test_manifest_maps_canonical_id_to_a_local_file(tmp_path: Path) -> None:
    """A manifest can preserve a canonical ID independent of its filename."""

    root = tmp_path / "videos"
    root.mkdir()
    source = root / "downloaded-source.mp4"
    source.write_bytes(b"video")
    manifest = tmp_path / "videos.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "videos": [
                    {
                        "video_id": "folder.one.L21_V001",
                        "path": "downloaded-source.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    catalog = VideoCatalog(root, manifest)

    entry = catalog.get("folder.one.L21_V001")
    assert entry is not None
    assert entry.path == source.resolve()
    assert entry.media_type == "video/mp4"


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    """A manifest cannot turn the origin into an arbitrary file server."""

    root = tmp_path / "videos"
    root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"not allowed")
    manifest = tmp_path / "videos.json"
    manifest.write_text(
        json.dumps({"videos": [{"video_id": "escape", "path": "../outside.mp4"}]}),
        encoding="utf-8",
    )

    with pytest.raises(CatalogError, match="escapes"):
        VideoCatalog(root, manifest)


def test_discovery_rejects_duplicate_filename_stems(tmp_path: Path) -> None:
    """Ambiguous automatic IDs require an explicit manifest."""

    root = tmp_path / "videos"
    (root / "one").mkdir(parents=True)
    (root / "two").mkdir()
    (root / "one" / "same.mp4").write_bytes(b"one")
    (root / "two" / "same.webm").write_bytes(b"two")

    with pytest.raises(CatalogError, match="duplicate discovered"):
        VideoCatalog(root)
