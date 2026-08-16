from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from hcmai.data.corpus_build.source_cache import SourceVideoCache
from hcmai.data.s3 import S3VideoObject


class _Client:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.downloads: list[str] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        assert bucket == "bucket"
        self.downloads.append(key)
        Path(filename).write_bytes(self.payloads[key])


def _source(key: str, payload: bytes, second: int) -> S3VideoObject:
    return S3VideoObject(
        key=key,
        size=len(payload),
        etag=f"etag-{second}",
        last_modified_ns=round(
            datetime(2026, 8, 16, 1, 0, second, tzinfo=UTC).timestamp()
            * 1_000_000_000
        ),
    )


def test_population_is_atomic_and_reuses_valid_files(tmp_path: Path) -> None:
    payloads = {"data/V001.mp4": b"one", "data/V002.mp4": b"two"}
    sources = [
        _source("data/V001.mp4", payloads["data/V001.mp4"], 1),
        _source("data/V002.mp4", payloads["data/V002.mp4"], 2),
    ]
    client = _Client(payloads)
    cache = SourceVideoCache(client, "bucket", tmp_path / "cache")

    first = cache.populate(sources, max_workers=2, minimum_free_bytes=0)
    second = cache.populate(sources, max_workers=2, minimum_free_bytes=0)

    assert first.downloaded_count == 2
    assert second.reused_count == 2
    assert sorted(client.downloads) == ["data/V001.mp4", "data/V002.mp4"]
    assert [path.read_bytes() for path in first.paths] == [b"one", b"two"]
    assert not list((tmp_path / "cache").rglob("*.partial"))


def test_invalid_cached_size_is_downloaded_again(tmp_path: Path) -> None:
    payload = b"correct"
    source = _source("data/V001.mp4", payload, 1)
    client = _Client({source.key: payload})
    cache = SourceVideoCache(client, "bucket", tmp_path / "cache")
    target = cache.path_for(source)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bad")

    result = cache.populate([source], max_workers=1, minimum_free_bytes=0)

    assert result.downloaded_count == 1
    assert target.read_bytes() == payload


def test_population_rejects_insufficient_disk_before_downloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"video"
    source = _source("data/V001.mp4", payload, 1)
    client = _Client({source.key: payload})
    cache = SourceVideoCache(client, "bucket", tmp_path / "cache")
    monkeypatch.setattr(
        "hcmai.data.corpus_build.source_cache.shutil.disk_usage",
        lambda _: SimpleNamespace(total=100, used=90, free=10),
    )

    with pytest.raises(OSError, match="Insufficient disk"):
        cache.populate([source], max_workers=1, minimum_free_bytes=8)

    assert client.downloads == []
